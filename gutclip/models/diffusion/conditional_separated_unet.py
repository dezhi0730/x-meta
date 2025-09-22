import torch
import torch.nn as nn
import torch.nn.functional as F
from gutclip.models.diffusion.separated_unet import SeparatedDiffusionModel
from gutclip.models.condition_encoder import SimpleConditionEncoder


class ConditionalSeparatedDiffusionModel(SeparatedDiffusionModel):
    """
    条件分离扩散模型：在SeparatedDiffusionModel基础上添加条件控制
    
    新增输入：
    - condition_embedding: (B, condition_dim) 条件嵌入向量
    
    输出：
    - eps_hat: (ΣN,) 预测的噪声
    - pres_logit: (ΣN,) presence 的 logits
    """
    
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 128,
                 out_dim: int = 256,
                 num_layers: int = 4,
                 dropout_rate: float = 0.25,
                 condition_dim: int = 256,
                 model_cfg: dict = None,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            model_cfg=model_cfg,
            **kwargs
        )
        
        # Prompt投影层：3072 → 256
        self.prompt_projection = nn.Sequential(
            nn.Linear(3072, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(condition_dim, condition_dim),
            nn.LayerNorm(condition_dim)
        )
        
        # 条件处理模块
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 条件融合：将条件信息融合到树特征中
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # 时间条件融合（可选）
        self.time_condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def _compute_dna_masked_mean(self, dna_tensor, dna_pad_mask):
        """
        计算DNA的masked mean，避免padding零值影响
        
        Args:
            dna_tensor: (B, L_max, 768) DNA嵌入
            dna_pad_mask: (B, L_max) True表示有效位置
            
        Returns:
            dna_masked_mean: (B, 768) 有效位置的均值
        """
        # 将padding位置设为0
        masked_dna = dna_tensor * dna_pad_mask.unsqueeze(-1).float()
        
        # 计算有效位置的数量
        valid_lengths = dna_pad_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        
        # 避免除零
        valid_lengths = torch.clamp(valid_lengths, min=1.0)
        
        # 计算masked mean
        dna_masked_mean = masked_dna.sum(dim=1) / valid_lengths  # (B, 768)
        
        return dna_masked_mean
    
    def set_external_condition_encoder(self, condition_encoder):
        """设置外部条件编码器"""
        self.external_condition_encoder = condition_encoder
    
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 包含以下字段的批次数据
                - x_t: (ΣN, 1) 加噪后的 log_abun
                - x_t_pres: (ΣN, 1) 加噪后的 presence
                - x_static: (ΣN, 2) 静态特征
                - mask_feat: (ΣN, 1) 显式 mask 特征
                - edge_index: (2, E) 边索引
                - pos: (ΣN, 3) 位置特征
                - batch: (ΣN,) batch 索引
                - t_idx: (B,) 时间步索引
                - condition_embedding: (B, condition_dim) 条件嵌入向量
        """
        # 获取批次大小
        batch_size = batch.t_idx.size(0)
        
        # 处理条件信息（使用raw prompt embeddings）
        if hasattr(batch, 'prompt_embeddings') and batch.prompt_embeddings is not None:
            # 确保prompt_embeddings在正确设备上
            prompt_embeddings = batch.prompt_embeddings.to(batch.x_t.device)
            # 投影prompt embeddings：3072 → 256
            prompt_projected = self.prompt_projection(prompt_embeddings)  # (B, 256)
            
            # 处理无条件情况（更稳定的mask处理）
            unconditional_mask = batch.unconditional_mask.to(batch.x_t.device).float().unsqueeze(-1)  # (B, 1) float类型
            prompt_projected = prompt_projected * (1 - unconditional_mask)  # 无条件时置零
            
            # 通过条件编码器
            condition_emb = self.condition_encoder(prompt_projected)  # (B, hidden_dim)
            condition_proj = self.condition_fusion(condition_emb)  # (B, out_dim)
        else:
            # 无条件情况
            condition_proj = torch.zeros(batch_size, self.out_dim, 
                                       device=batch.x_t.device)
        
        # 获取树特征（使用父类的方法）
        # 注意：树编码器期望4维输入，不包含mask_feat
        tree_features = self.tree_encoder(
            torch.cat([batch.x_t, batch.x_t_pres, batch.x_static], dim=1),
            batch.edge_index,
            batch.pos,
            batch.batch
        )  # (ΣN, out_dim)
        
        # 时间嵌入
        time_emb = self.time_embed(batch.t_idx.unsqueeze(-1).float())  # (B, hidden_dim)
        time_proj = self.time_condition_fusion(time_emb)  # (B, out_dim)
        
        # 将时间条件扩展到节点级别
        batch_indices = batch.batch
        time_proj_expanded = time_proj[batch_indices]  # (ΣN, out_dim)
        condition_proj_expanded = condition_proj[batch_indices]  # (ΣN, out_dim)
        
        # 融合所有条件信息
        conditioned_features = tree_features + condition_proj_expanded + time_proj_expanded
        
        # 使用分离头进行预测
        eps_hat = self.abun_head(conditioned_features).squeeze(-1)  # (ΣN,)
        pres_logit = self.pres_head(conditioned_features).squeeze(-1)  # (ΣN,)
        
        return {
            "eps_hat": eps_hat,
            "pres_logit": pres_logit
        }


class ConditionalSeparatedDiffusionModelWithDNA(ConditionalSeparatedDiffusionModel):
    """
    带DNA条件的条件分离扩散模型
    """
    
    def __init__(self,
                 input_dim: int = 4,
                 dna_dim: int = 768,
                 hidden_dim: int = 128,
                 out_dim: int = 256,
                 num_layers: int = 4,
                 dropout_rate: float = 0.25,
                 condition_dim: int = 256,
                 pretrained_dna_encoder: bool = True,
                 dna_output_dim: int = None,
                 model_cfg: dict = None,
                 **kwargs):
        
        # 先调用父类初始化（不包含DNA相关）
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            condition_dim=condition_dim,
            model_cfg=model_cfg,
            **kwargs
        )
        
        # 添加DNA编码器（从SeparatedDiffusionModelWithDNA复制）
        from gutclip.models.dna_encoder import DNAEncoder
        
        self.dna_encoder = DNAEncoder(
            input_dim=dna_dim,
            output_dim=dna_output_dim or hidden_dim
        )
        
        # DNA条件融合
        self.dna_condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # 如果使用预训练DNA编码器
        if pretrained_dna_encoder:
            self.pretrained_dna_encoder = True
        else:
            self.pretrained_dna_encoder = False
    
    def forward(self, batch):
        """
        前向传播（包含DNA条件）
        """
        # 获取批次大小
        batch_size = batch.t_idx.size(0)
        
        # 处理DNA条件
        if hasattr(batch, 'dna') and batch.dna is not None:
            dna_emb = self.dna_encoder(
                batch.dna,
                getattr(batch, 'dna_pad_mask', None),
                getattr(batch, 'dna_rand_mask', None)
            )  # (B, hidden_dim)
            dna_proj = self.dna_condition_fusion(dna_emb)  # (B, out_dim)
        else:
            dna_proj = torch.zeros(batch_size, self.tree_encoder.out_dim, 
                                 device=batch.x_t.device)
        
        # 处理干预条件（使用raw prompt embeddings）
        if hasattr(batch, 'prompt_embeddings') and batch.prompt_embeddings is not None:
            # 投影prompt embeddings：3072 → 256
            prompt_projected = self.prompt_projection(batch.prompt_embeddings)  # (B, 256)
            
            # 处理无条件情况（更稳定的mask处理）
            unconditional_mask = batch.unconditional_mask.to(batch.x_t.device).float().unsqueeze(-1)  # (B, 1) float类型
            prompt_projected = prompt_projected * (1 - unconditional_mask)  # 无条件时置零
            
            # 通过条件编码器
            condition_emb = self.condition_encoder(prompt_projected)
            condition_proj = self.condition_fusion(condition_emb)
        else:
            condition_proj = torch.zeros(batch_size, self.tree_encoder.out_dim, 
                                       device=batch.x_t.device)
        
        # 获取树特征
        # 注意：树编码器期望4维输入，不包含mask_feat
        tree_features = self.tree_encoder(
            torch.cat([batch.x_t, batch.x_t_pres, batch.x_static], dim=1),
            batch.edge_index,
            batch.pos,
            batch.batch
        )
        
        # 时间嵌入
        time_emb = self.time_embed(batch.t_idx.unsqueeze(-1).float())
        time_proj = self.time_condition_fusion(time_emb)
        
        # 扩展到节点级别
        batch_indices = batch.batch
        time_proj_expanded = time_proj[batch_indices]
        condition_proj_expanded = condition_proj[batch_indices]
        dna_proj_expanded = dna_proj[batch_indices]
        
        # 融合所有条件信息
        conditioned_features = (tree_features + 
                              condition_proj_expanded + 
                              time_proj_expanded + 
                              dna_proj_expanded)
        
        # 预测
        eps_hat = self.abun_head(conditioned_features).squeeze(-1)
        pres_logit = self.pres_head(conditioned_features).squeeze(-1)
        
        return {
            "eps_hat": eps_hat,
            "pres_logit": pres_logit
        }
    
    def load_pretrained_encoders(self, gutclip_checkpoint_path: str, load_tree_encoder: bool = False):
        """加载预训练的编码器（从SeparatedDiffusionModelWithDNA复制）"""
        # 这里可以复用原有的加载逻辑
        pass 
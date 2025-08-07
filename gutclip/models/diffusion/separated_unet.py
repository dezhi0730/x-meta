import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet1DModel
from gutclip.models.tree_encoder import TreeEncoder
from gutclip.models.dna_encoder import DNAEncoder


class SeparatedDiffusionModel(nn.Module):
    """
    分离建模扩散模型：支持 Bernoulli + Gaussian 分离建模
    
    输入：
    - x_t: (ΣN, 1) 加噪后的 log_abun
    - x_t_pres: (ΣN, 1) 加噪后的 presence
    - x_static: (ΣN, 2) 静态特征
    - edge_index: (2, E) 边索引
    - pos: (ΣN, 3) 位置特征
    - batch: (ΣN,) batch 索引
    - t_idx: (B,) 时间步索引
    
    输出：
    - eps_hat: (ΣN,) 预测的噪声
    - pres_logit: (ΣN,) presence 的 logits
    """
    
    def __init__(self,
                 input_dim: int = 4,      # x_t + x_static + mask_feat
                 hidden_dim: int = 128,
                 out_dim: int = 256,
                 num_layers: int = 4,
                 dropout_rate: float = 0.25,
                 model_cfg: dict = None,  # 模型配置字典
                 **kwargs):
        super().__init__()
        
        # 使用配置或默认值
        if model_cfg is None:
            model_cfg = {}
        
        # 树编码器配置
        tree_cfg = model_cfg.get('tree_encoder', {})
        self.tree_encoder = TreeEncoder(
            input_dim=input_dim,
            hidden_dim=tree_cfg.get('hidden_dim', hidden_dim),
            out_dim=tree_cfg.get('out_dim', out_dim),
            num_layers=tree_cfg.get('num_layers', num_layers),
            dropout_rate=tree_cfg.get('dropout_rate', dropout_rate),
            return_node_emb=tree_cfg.get('return_node_emb', True)
        )
        
        # 分离头配置
        heads_cfg = model_cfg.get('heads', {})
        abun_cfg = heads_cfg.get('abundance', {})
        pres_cfg = heads_cfg.get('presence', {})
        
        self.abun_head = nn.Sequential(
            nn.Linear(out_dim, abun_cfg.get('hidden_dim', 64)),
            nn.ReLU(),
            nn.Dropout(abun_cfg.get('dropout_rate', dropout_rate)),
            nn.Linear(abun_cfg.get('hidden_dim', 64), abun_cfg.get('output_dim', 1))
        )
        
        self.pres_head = nn.Sequential(
            nn.Linear(out_dim, pres_cfg.get('hidden_dim', 64)),
            nn.ReLU(),
            nn.Dropout(pres_cfg.get('dropout_rate', dropout_rate)),
            nn.Linear(pres_cfg.get('hidden_dim', 64), pres_cfg.get('output_dim', 1))
        )
        
        # 时间嵌入配置
        time_cfg = model_cfg.get('time_embed', {})
        self.time_embed = nn.Sequential(
            nn.Linear(time_cfg.get('input_dim', 1), time_cfg.get('hidden_dim', hidden_dim)),
            nn.ReLU(),
            nn.Linear(time_cfg.get('hidden_dim', hidden_dim), time_cfg.get('hidden_dim', hidden_dim))
        )
        
        # 时间条件化
        self.time_condition = nn.Sequential(
            nn.Linear(time_cfg.get('hidden_dim', hidden_dim), time_cfg.get('output_dim', out_dim)),
            nn.ReLU(),
            nn.Linear(time_cfg.get('output_dim', out_dim), time_cfg.get('output_dim', out_dim))
        )
    
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
        
        Returns:
            dict: 包含 eps_hat 和 pres_logit
        """
        # 构建输入特征
        x_comb = torch.cat([
            batch.x_t,           # (ΣN, 1) log_abun
            batch.x_static,      # (ΣN, 2) 静态特征
            batch.mask_feat      # (ΣN, 1) mask 特征
        ], dim=1)               # (ΣN, 4)
        
        # 通过树编码器
        h = self.tree_encoder(
            x=x_comb,
            edge_index=batch.edge_index,
            pos=batch.pos,
            batch=batch.batch
        )  # (ΣN, out_dim)
        
        # 时间条件化
        t_emb = self.time_embed(batch.t_idx.float().unsqueeze(-1))  # (B, hidden_dim)
        t_cond = self.time_condition(t_emb)  # (B, out_dim)
        
        # 将时间条件扩展到节点级别
        t_cond_nodes = t_cond[batch.batch]  # (ΣN, out_dim)
        h = h + t_cond_nodes  # 时间条件化
        
        # 分离头预测
        eps_hat = self.abun_head(h).squeeze(-1)      # (ΣN,) 连续 ε̂
        pres_logit = self.pres_head(h).squeeze(-1)   # (ΣN,) 对 presence=1 的 logits
        
        return {
            "eps_hat": eps_hat,
            "pres_logit": pres_logit
        }


class SeparatedDiffusionModelWithDNA(SeparatedDiffusionModel):
    """
    带DNA条件的分离建模扩散模型
    使用预训练的DNA encoder
    """
    
    def __init__(self,
                 input_dim: int = 4,
                 dna_dim: int = 768,
                 hidden_dim: int = 128,
                 out_dim: int = 256,
                 num_layers: int = 4,
                 dropout_rate: float = 0.25,
                 pretrained_dna_encoder: bool = True,
                 dna_output_dim: int = None,  # 动态DNA输出维度
                 model_cfg: dict = None,  # 模型配置字典
                 **kwargs):
        
        # 使用配置或默认值
        if model_cfg is None:
            model_cfg = {}
        
        # 如果没有指定DNA输出维度，使用hidden_dim作为默认值
        if dna_output_dim is None:
            dna_output_dim = hidden_dim
        
        super().__init__(
            input_dim=input_dim + dna_output_dim,  # 动态计算输入维度
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            model_cfg=model_cfg,  # 传递配置给父类
            **kwargs
        )
        
        # DNA编码器配置
        dna_cfg = model_cfg.get('dna_encoder', {})
        self.dna_encoder = DNAEncoder(
            input_dim=dna_cfg.get('input_dim', dna_dim), 
            output_dim=dna_output_dim,  # 动态输出维度
            dropout_rate=dna_cfg.get('dropout_rate', dropout_rate)
        )
        
        # 保存DNA输出维度，供后续使用
        self.dna_output_dim = dna_output_dim
        
        # 标记是否使用预训练权重
        self.pretrained_dna_encoder = pretrained_dna_encoder
        
        # 保存配置
        self.model_cfg = model_cfg
    
    def load_pretrained_encoders(self, gutclip_checkpoint_path: str, load_tree_encoder: bool = False):
        """
        从预训练的GutCLIP模型中加载DNA encoder和tree encoder权重
        动态检测预训练模型的维度并适配
        
        Args:
            gutclip_checkpoint_path: GutCLIP模型检查点路径
            load_tree_encoder: 是否也加载tree encoder权重
        """
        print(f"[INFO] 加载预训练编码器: {gutclip_checkpoint_path}")
        
        # 加载GutCLIP检查点
        ckpt_data = torch.load(gutclip_checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt_data.get("model", ckpt_data.get("state_dict", ckpt_data))
        
        # 清理状态字典前缀
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):   # Lightning
                k = k[len("model."):]
            if k.startswith("module."):  # DDP
                k = k[len("module."):]
            clean_state[k] = v
        
        # 动态检测预训练模型的DNA encoder输出维度
        dna_output_dim = self._detect_dna_output_dim(clean_state)
        print(f"[INFO] 检测到预训练DNA encoder输出维度: {dna_output_dim}")
        
        # 如果当前模型的DNA输出维度与预训练模型不匹配，需要重新创建DNA encoder
        if dna_output_dim != self.dna_output_dim:
            print(f"[INFO] 重新创建DNA encoder以匹配预训练维度: {self.dna_output_dim} -> {dna_output_dim}")
            
            # 重新创建DNA encoder
            self.dna_encoder = DNAEncoder(
                input_dim=768,  # DNA-BERT嵌入维度
                output_dim=dna_output_dim,
                dropout_rate=0.25
            )
            
            # 更新DNA输出维度
            self.dna_output_dim = dna_output_dim
            
            # 重新创建树编码器以匹配新的输入维度
            old_tree_encoder = self.tree_encoder
            self.tree_encoder = TreeEncoder(
                input_dim=4 + dna_output_dim,  # 基础特征 + DNA特征
                hidden_dim=128,  # 使用默认值
                out_dim=256,     # 使用默认值
                num_layers=4,    # 使用默认值
                dropout_rate=0.25,  # 使用默认值
                return_node_emb=True
            )
            print(f"[INFO] 重新创建树编码器，输入维度: {4 + dna_output_dim}")
        
        # 提取DNA encoder的权重
        dna_encoder_state = {}
        for k, v in clean_state.items():
            if k.startswith("dna_encoder."):
                # 移除dna_encoder前缀
                new_key = k[len("dna_encoder."):]
                dna_encoder_state[new_key] = v
        
        if not dna_encoder_state:
            raise ValueError("在检查点中未找到DNA encoder权重")
        
        # 加载DNA encoder权重
        missing_keys, unexpected_keys = self.dna_encoder.load_state_dict(dna_encoder_state, strict=False)
        
        print(f"[INFO] DNA encoder加载完成")
        print(f"[INFO] Missing keys: {missing_keys}")
        print(f"[INFO] Unexpected keys: {unexpected_keys}")
        
        # 冻结DNA encoder参数（可选）
        if self.pretrained_dna_encoder:
            for param in self.dna_encoder.parameters():
                param.requires_grad = False
            print("[INFO] DNA encoder参数已冻结")
        
        # 加载tree encoder权重（可选）
        if load_tree_encoder:
            tree_encoder_state = {}
            for k, v in clean_state.items():
                if k.startswith("tree_encoder."):
                    # 移除tree_encoder前缀
                    new_key = k[len("tree_encoder."):]
                    tree_encoder_state[new_key] = v
            
            if tree_encoder_state:
                # 检查tree encoder的输入维度是否匹配
                expected_input_dim = self._detect_tree_input_dim(tree_encoder_state)
                current_input_dim = 4 + dna_output_dim
                
                print(f"[INFO] Tree encoder期望输入维度: {expected_input_dim}")
                print(f"[INFO] 当前模型输入维度: {current_input_dim}")
                
                if expected_input_dim != current_input_dim:
                    print(f"[WARN] Tree encoder输入维度不匹配，跳过tree encoder加载")
                    print(f"[WARN] 预训练tree encoder期望: {expected_input_dim}，当前模型: {current_input_dim}")
                    print(f"[WARN] 建议：只使用DNA encoder，或者调整模型架构以匹配预训练维度")
                else:
                    # 加载tree encoder权重
                    missing_keys, unexpected_keys = self.tree_encoder.load_state_dict(tree_encoder_state, strict=False)
                    
                    print(f"[INFO] Tree encoder加载完成")
                    print(f"[INFO] Missing keys: {missing_keys}")
                    print(f"[INFO] Unexpected keys: {unexpected_keys}")
                    
                    # 冻结tree encoder参数（可选）
                    for param in self.tree_encoder.parameters():
                        param.requires_grad = False
                    print("[INFO] Tree encoder参数已冻结")
            else:
                print("[WARN] 在检查点中未找到tree encoder权重")
    
    def _detect_tree_input_dim(self, state_dict: dict) -> int:
        """
        从预训练模型的状态字典中检测tree encoder的输入维度
        
        Args:
            state_dict: 预训练模型的状态字典
            
        Returns:
            Tree encoder的输入维度
        """
        # 查找tree encoder的第一个投影层权重
        tree_keys = [k for k in state_dict.keys() if k.startswith("egnn.proj.0.weight")]
        
        if tree_keys:
            # 获取权重形状，第二个维度是输入维度
            weight_shape = state_dict[tree_keys[0]].shape
            input_dim = weight_shape[1]
            print(f"[INFO] 从权重形状检测到Tree输入维度: {weight_shape} -> {input_dim}")
            return input_dim
        else:
            # 如果找不到投影层，使用默认值
            print("[WARN] 无法从权重中检测Tree输入维度，使用默认值9")
            return 9
    
    def _detect_dna_output_dim(self, state_dict: dict) -> int:
        """
        从预训练模型的状态字典中检测DNA encoder的输出维度
        
        Args:
            state_dict: 预训练模型的状态字典
            
        Returns:
            DNA encoder的输出维度
        """
        # 查找DNA encoder的最终输出层权重
        dna_keys = [k for k in state_dict.keys() if k.startswith("dna_encoder.")]
        
        # 查找transform.6.weight（最终输出层）
        final_output_key = None
        for key in dna_keys:
            if "transform.6.weight" in key:
                final_output_key = key
                break
        
        if final_output_key:
            # 获取权重形状，第一个维度是输出维度
            weight_shape = state_dict[final_output_key].shape
            output_dim = weight_shape[0]
            print(f"[INFO] 从权重形状检测到DNA输出维度: {weight_shape} -> {output_dim}")
            return output_dim
        else:
            # 如果找不到最终输出层，尝试从配置中获取
            print("[WARN] 无法从权重中检测DNA输出维度，使用默认值256")
            return 256
    
    def load_pretrained_dna_encoder(self, gutclip_checkpoint_path: str):
        """
        向后兼容的方法，只加载DNA encoder
        """
        self.load_pretrained_encoders(gutclip_checkpoint_path, load_tree_encoder=False)
    
    def forward(self, batch):
        """
        带DNA条件的前向传播
        """
        x_t       = batch.x_t
        x_static  = batch.x_static
        mask_feat = batch.mask_feat

        if hasattr(batch, 'dna') and batch.dna is not None:
            # 使用预训练的DNA encoder处理DNA数据
            # DNA数据格式: (B, L_max, 768)
            dna_pad_mask = getattr(batch, 'dna_pad_mask', None)
            dna_rand_mask = getattr(batch, 'dna_rand_mask', None)
            
            # 使用DNAEncoder进行编码
            dna_encoded = self.dna_encoder(batch.dna, dna_pad_mask, dna_rand_mask)  # (B, hidden_dim)
            
            # 将DNA编码扩展到节点级别
            batch_indices = batch.batch.long().to(x_t.device)
            dna_nodes = dna_encoded[batch_indices]  # (ΣN, hidden_dim)
        else:
            # 无 DNA -> 用 0 向量占位，保持列数一致
            zeros = torch.zeros(
                x_t.size(0),              # ΣN
                self.dna_output_dim,      # 动态DNA输出维度
                device=x_t.device, dtype=x_t.dtype
            )
            dna_nodes = zeros

        # 特征拼接
        x_comb = torch.cat([x_t, x_static, mask_feat, dna_nodes], dim=1)  # (ΣN, 4+hidden_dim)
        
        # 通过树编码器
        h = self.tree_encoder(
            x=x_comb,
            edge_index=batch.edge_index,
            pos=batch.pos,
            batch=batch.batch
        )
        
        # 时间条件化
        t_emb = self.time_embed(batch.t_idx.float().unsqueeze(-1))
        t_cond = self.time_condition(t_emb)
        t_cond_nodes = t_cond[batch.batch]
        h = h + t_cond_nodes
        
        # 分离头预测
        eps_hat = self.abun_head(h).squeeze(-1)
        pres_logit = self.pres_head(h).squeeze(-1)
        
        return {
            "eps_hat": eps_hat,
            "pres_logit": pres_logit
        } 
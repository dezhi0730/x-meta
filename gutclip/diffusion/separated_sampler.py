import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from tqdm import tqdm


class SeparatedDiffusionSampler:
    """
    分离建模采样器：支持 Bernoulli + Gaussian 分离建模
    
    两阶段采样：
    1. Stage-1: 离散 reverse step (presence)
    2. Stage-2: 连续流 (log_abun)
    """
    
    def __init__(self, betas: torch.Tensor, device: torch.device):
        self.betas = betas.to(device)
        self.device = device
        
        # 预计算
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod)
    
    @torch.no_grad()
    def sample(self, 
               model, 
               batch,
               num_steps: int = 50,
               temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        分离建模采样
        
        Args:
            model: 分离建模模型
            batch: 批次数据 (Data对象)
            num_steps: 采样步数
            temperature: 温度参数（用于presence采样）
        
        Returns:
            dict: 包含采样结果
        """
        device = next(model.parameters()).device
        
        # 初始化噪声
        B = batch.t_idx.size(0)
        N = batch.x0_abun.size(0)
        
        # 初始化 x_T
        x_t_abun = torch.randn(N, 1, device=device)  # 连续流
        x_t_pres = torch.randint(0, 2, (N, 1), device=device, dtype=torch.long)  # 离散流
        
        # 时间步
        timesteps = torch.linspace(len(self.betas) - 1, 0, num_steps + 1, dtype=torch.long, device=device)
        
        for i in tqdm(range(num_steps), desc="Sampling"):
            t = timesteps[i]
            t_next = timesteps[i + 1] if i < num_steps - 1 else torch.tensor(0, device=device)
            
            # 构建当前批次
            current_batch = type(batch)()  # 创建新的Data对象
            current_batch.x_t = x_t_abun
            current_batch.x_t_pres = x_t_pres
            current_batch.mask_feat = x_t_pres.float()
            current_batch.t_idx = t.expand(B)
            current_batch.x_static = batch.x_static
            current_batch.edge_index = batch.edge_index
            current_batch.pos = batch.pos
            current_batch.batch = batch.batch
            if hasattr(batch, 'dna'):
                current_batch.dna = batch.dna
            
            # 模型预测
            model_output = model(current_batch)
            eps_hat = model_output["eps_hat"]
            pres_logit = model_output["pres_logit"]
            
            # Stage-1: 离散 reverse step (presence)
            logp_1 = pres_logit
            p_1 = torch.sigmoid(logp_1 / temperature)
            prob_keep = 1.0 / (1.0 + torch.exp(-logp_1 / temperature))
            x_t_minus_1_pres = torch.bernoulli(prob_keep).long()
            
            # Stage-2: 连续流 (log_abun)
            # 反推 log_abun_{t-1}
            alpha_t = self.alpha_t[t].view(-1, 1)
            sigma_t = self.sigma_t[t].view(-1, 1)
            
            # 只在 presence=1 的节点更新 abundance
            mask = x_t_minus_1_pres.squeeze(-1).float()
            
            # 预测 x0
            x0_hat_abun = (x_t_abun - sigma_t * eps_hat) / alpha_t
            
            # 只在 presence=1 的节点应用预测
            x0_hat_abun = x0_hat_abun * mask.unsqueeze(-1)
            
            # 计算 x_{t-1}
            if t_next > 0:
                alpha_t_next = self.alpha_t[t_next].view(-1, 1)
                sigma_t_next = self.sigma_t[t_next].view(-1, 1)
                
                # 添加噪声
                noise = torch.randn_like(x_t_abun)
                x_t_minus_1_abun = alpha_t_next * x0_hat_abun + sigma_t_next * noise
                
                # 只在 presence=1 的节点更新
                x_t_minus_1_abun = x_t_minus_1_abun * mask.unsqueeze(-1) + \
                                  x_t_abun * (1 - mask.unsqueeze(-1))
            else:
                x_t_minus_1_abun = x0_hat_abun
            
            # 更新状态
            x_t_abun = x_t_minus_1_abun
            x_t_pres = x_t_minus_1_pres
        
        return {
            "x0_abun": x_t_abun,
            "x0_pres": x_t_pres,
            "presence_mask": x_t_pres.squeeze(-1).float()
        }
    
    @torch.no_grad()
    def sample_with_guidance(self, 
                            model, 
                            batch,
                            guidance_scale: float = 7.5,
                            num_steps: int = 50,
                            temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        带引导的分离建模采样
        
        Args:
            model: 分离建模模型
            batch: 批次数据 (Data对象)
            guidance_scale: 引导强度
            num_steps: 采样步数
            temperature: 温度参数
        
        Returns:
            dict: 包含采样结果
        """
        device = next(model.parameters()).device
        
        # 初始化噪声
        B = batch.t_idx.size(0)
        N = batch.x0_abun.size(0)
        
        # 初始化 x_T
        x_t_abun = torch.randn(N, 1, device=device)
        x_t_pres = torch.randint(0, 2, (N, 1), device=device, dtype=torch.long)
        
        # 时间步
        timesteps = torch.linspace(len(self.betas) - 1, 0, num_steps + 1, dtype=torch.long, device=device)
        
        for i in tqdm(range(num_steps), desc="Guided Sampling"):
            t = timesteps[i]
            t_next = timesteps[i + 1] if i < num_steps - 1 else torch.tensor(0, device=device)
            
            # 构建当前批次
            current_batch = type(batch)()  # 创建新的Data对象
            current_batch.x_t = x_t_abun
            current_batch.x_t_pres = x_t_pres
            current_batch.mask_feat = x_t_pres.float()
            current_batch.t_idx = t.expand(B)
            current_batch.x_static = batch.x_static
            current_batch.edge_index = batch.edge_index
            current_batch.pos = batch.pos
            current_batch.batch = batch.batch
            if hasattr(batch, 'dna'):
                current_batch.dna = batch.dna
            
            # 模型预测
            model_output = model(current_batch)
            eps_hat = model_output["eps_hat"]
            pres_logit = model_output["pres_logit"]
            
            # 引导：使用原始数据作为条件
            if hasattr(batch, 'x0_abun') and hasattr(batch, 'x0_pres'):
                # 计算引导梯度
                x0_abun_gt = batch.x0_abun
                x0_pres_gt = batch.x0_pres
                
                # Abundance 引导
                alpha_t = self.alpha_t[t].view(-1, 1)
                sigma_t = self.sigma_t[t].view(-1, 1)
                x0_hat_abun = (x_t_abun - sigma_t * eps_hat) / alpha_t
                
                # 计算引导梯度
                abun_grad = guidance_scale * (x0_hat_abun - x0_abun_gt)
                eps_hat = eps_hat - sigma_t * abun_grad / alpha_t
                
                # Presence 引导
                pres_grad = guidance_scale * (pres_logit - x0_pres_gt.squeeze(-1).float())
                pres_logit = pres_logit - pres_grad
            
            # Stage-1: 离散 reverse step (presence)
            logp_1 = pres_logit
            p_1 = torch.sigmoid(logp_1 / temperature)
            prob_keep = 1.0 / (1.0 + torch.exp(-logp_1 / temperature))
            x_t_minus_1_pres = torch.bernoulli(prob_keep).long()
            
            # Stage-2: 连续流 (log_abun)
            mask = x_t_minus_1_pres.squeeze(-1).float()
            
            # 预测 x0
            x0_hat_abun = (x_t_abun - sigma_t * eps_hat) / alpha_t
            x0_hat_abun = x0_hat_abun * mask.unsqueeze(-1)
            
            # 计算 x_{t-1}
            if t_next > 0:
                alpha_t_next = self.alpha_t[t_next].view(-1, 1)
                sigma_t_next = self.sigma_t[t_next].view(-1, 1)
                
                noise = torch.randn_like(x_t_abun)
                x_t_minus_1_abun = alpha_t_next * x0_hat_abun + sigma_t_next * noise
                x_t_minus_1_abun = x_t_minus_1_abun * mask.unsqueeze(-1) + \
                                  x_t_abun * (1 - mask.unsqueeze(-1))
            else:
                x_t_minus_1_abun = x0_hat_abun
            
            # 更新状态
            x_t_abun = x_t_minus_1_abun
            x_t_pres = x_t_minus_1_pres
        
        return {
            "x0_abun": x_t_abun,
            "x0_pres": x_t_pres,
            "presence_mask": x_t_pres.squeeze(-1).float()
        } 
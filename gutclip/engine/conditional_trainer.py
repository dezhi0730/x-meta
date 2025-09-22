import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from gutclip.engine.trainer_tree_diffusion import TreeDiffusionTrainer


class ConditionalTreeDiffusionTrainer(TreeDiffusionTrainer):
    """条件树扩散训练器，扩展原有的TreeDiffusionTrainer"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, 
                 condition_encoder=None, unconditional_prob=0.1, **kwargs):
        super().__init__(model, train_loader, val_loader, optimizer, **kwargs)
        
        self.condition_encoder = condition_encoder
        self.unconditional_prob = unconditional_prob  # Classifier-free guidance概率
        
        # Epoch级别的统计变量
        self.epoch_unconditional_count = 0
        self.epoch_total_count = 0
        self.epoch_batch_count = 0
        
        if self.is_main:
            print(f"[INFO] 条件训练器初始化完成，无条件概率: {unconditional_prob}")
    
    def _prepare_batch_with_conditions(self, batch):
        """为batch添加条件编码"""
        # 确保整个batch在正确设备上
        batch = batch.to(self.device, non_blocking=True)
        
        if hasattr(batch, 'condition_embedding') and batch.condition_embedding is not None:
            # 如果batch已经有条件嵌入，直接使用
            condition_embeddings = batch.condition_embedding
            batch_size = condition_embeddings.size(0)
            
            # Classifier-free guidance: 逐样本随机掩码条件
            if self.model.training:  # 修复：使用model.training而不是self.training
                # 逐样本Bernoulli掩码，更稳定
                mask = (torch.rand(batch_size, device=condition_embeddings.device) 
                       < self.unconditional_prob).float().unsqueeze(1)  # (B,1)
                condition_embeddings = condition_embeddings * (1.0 - mask)  # 被丢弃样本→0向量
                
                # 记录哪些样本被丢弃
                batch.is_uncond = (mask.squeeze(1) > 0.5).float()  # (B,)
                
                # 统计epoch级别的数据（不打印）
                self.epoch_unconditional_count += mask.sum().item()
                self.epoch_total_count += batch_size
                self.epoch_batch_count += 1
            else:
                batch.is_uncond = torch.zeros(batch_size, device=condition_embeddings.device)
            
            # 确保设备类型一致
            batch.condition_embedding = condition_embeddings.to(
                self.device, 
                dtype=self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
            )
        else:
            # 如果没有条件嵌入，创建零向量
            batch_size = batch.x_t.size(0) if hasattr(batch, 'x_t') else 1
            condition_dim = 256  # 默认条件维度
            batch.condition_embedding = torch.zeros(
                batch_size, condition_dim, device=self.device
            )
        
        return batch
    
    def train_one_epoch(self, epoch: int, tb_writer=None):
        """重写训练epoch方法，添加无条件训练统计"""
        # 重置epoch统计
        self.epoch_unconditional_count = 0
        self.epoch_total_count = 0
        self.epoch_batch_count = 0
        
        # 调用父类的训练方法
        result = super().train_one_epoch(epoch, tb_writer)
        
        # 打印epoch级别的统计信息
        if self.is_main and self.epoch_total_count > 0:
            unconditional_ratio = self.epoch_unconditional_count / self.epoch_total_count
            print(f"[INFO] Epoch {epoch} 无条件训练统计:")
            print(f"  - 无条件样本: {self.epoch_unconditional_count}/{self.epoch_total_count}")
            print(f"  - 无条件比例: {unconditional_ratio:.1%}")
            print(f"  - 预期比例: {self.unconditional_prob:.1%}")
            print(f"  - 总batch数: {self.epoch_batch_count}")
        
        return result
    
    def _separated_loss_fn(self, batch, model_output):
        """条件分离损失函数"""
        # 首先准备条件信息
        batch = self._prepare_batch_with_conditions(batch)
        
        # 使用父类的损失函数
        loss, loss_details = super()._separated_loss_fn(batch, model_output)
        
        # 多任务损失分解（如果父类返回了逐样本损失）
        if hasattr(batch, 'is_uncond') and hasattr(batch, 'is_paired') and hasattr(batch, 'has_cond'):
            # 获取逐样本损失（这里需要根据父类的实际返回格式调整）
            # 假设父类返回的loss_details中有per_sample_loss
            if 'per_sample_loss' in loss_details:
                per_sample_loss = loss_details['per_sample_loss']  # (B,)
                
                # 获取样本标记
                is_uncond = batch.is_uncond  # (B,)
                is_paired = batch.is_paired  # (B,)
                has_cond = batch.has_cond    # (B,)
                
                # 定义样本子集掩码
                mask_cond_full = (is_paired == 1) & (has_cond == 1) & (is_uncond == 0)
                mask_cond_only = (is_paired == 0) & (has_cond == 1) & (is_uncond == 0)
                mask_uncond = (is_uncond == 1)
                
                def safe_mean(x, m):
                    """安全平均，避免除零"""
                    denom = m.float().sum().clamp_min(1.0)
                    return (x * m.float()).sum() / denom
                
                # 计算各子集损失
                L_cond = safe_mean(per_sample_loss, mask_cond_full)
                L_cond_only = safe_mean(per_sample_loss, mask_cond_only)
                L_uncond = safe_mean(per_sample_loss, mask_uncond)
                
                # 从配置读取权重
                lam1 = self.cfg.get("condition", {}).get("lam_cond", 1.0)
                lam2 = self.cfg.get("condition", {}).get("lam_cond_only", 0.5)
                lam3 = self.cfg.get("condition", {}).get("lam_uncond", 0.1)
                
                # 加权求和
                total_loss = lam1 * L_cond + lam2 * L_cond_only + lam3 * L_uncond
                
                # 更新损失详情
                loss_details.update({
                    'L_cond': L_cond.item(),
                    'L_cond_only': L_cond_only.item(),
                    'L_uncond': L_uncond.item(),
                    'mask_cond_full_sum': mask_cond_full.sum().item(),
                    'mask_cond_only_sum': mask_cond_only.sum().item(),
                    'mask_uncond_sum': mask_uncond.sum().item()
                })
                
                return total_loss, loss_details
        
        return loss, loss_details
    
    def _prepare_batch_with_conditions(self, batch):
        """为batch添加条件编码"""
        if hasattr(batch, 'condition_embedding') and batch.condition_embedding is not None:
            # 如果batch已经有条件嵌入，直接使用
            condition_embeddings = batch.condition_embedding
            batch_size = condition_embeddings.size(0)
            
            # Classifier-free guidance: 逐样本随机掩码条件
            if self.model.training:  # 修复：使用model.training而不是self.training
                # 逐样本Bernoulli掩码，更稳定
                mask = (torch.rand(batch_size, device=condition_embeddings.device) 
                       < self.unconditional_prob).float().unsqueeze(1)  # (B,1)
                condition_embeddings = condition_embeddings * (1.0 - mask)  # 被丢弃样本→0向量
                
                # 记录哪些样本被丢弃
                batch.is_uncond = (mask.squeeze(1) > 0.5).float()  # (B,)
                
                if self.is_main and mask.sum() > 0:
                    print(f"[DEBUG] {mask.sum().item()}/{batch_size} 样本使用无条件训练")
            else:
                batch.is_uncond = torch.zeros(batch_size, device=condition_embeddings.device)
            
            # 确保设备类型一致
            batch.condition_embedding = condition_embeddings.to(
                self.device, 
                dtype=self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
            )
        else:
            # 如果没有条件嵌入，创建零向量
            batch_size = batch.x_t.size(0) if hasattr(batch, 'x_t') else 1
            condition_dim = 256  # 默认条件维度
            batch.condition_embedding = torch.zeros(
                batch_size, condition_dim, 
                device=self.device,
                dtype=self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
            )
            batch.is_uncond = torch.zeros(batch_size, device=self.device)
        
        return batch
    
    def fit(self):
        """覆盖训练循环，添加条件处理"""
        self.model.to(self.device)
        global_step = 0
        
        # 余弦插值函数（从父类复制）
        def _cosine_interp(v_start: float, v_end: float, t: float) -> float:
            import math
            return v_end + 0.5 * (v_start - v_end) * (1 + math.cos(math.pi * t))
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # 动态调整lambda权重（从父类复制）
            progress = (epoch + 1) / self.epochs
            if progress < 0.4:
                t = progress / 0.4
                self.lambda_pres = _cosine_interp(5.0, 4.0, t)
                self.lambda_abun = _cosine_interp(1.0, 1.5, t)
            elif progress < 0.7:
                t = (progress - 0.4) / 0.3
                self.lambda_pres = _cosine_interp(4.0, 3.0, t)
                self.lambda_abun = _cosine_interp(1.5, 3.0, t)
            elif progress < 0.9:
                t = (progress - 0.7) / 0.2
                self.lambda_pres = _cosine_interp(3.0, 2.0, t)
                self.lambda_abun = _cosine_interp(3.0, 5.0, t)
            else:
                t = (progress - 0.9) / 0.1
                self.lambda_pres = _cosine_interp(2.0, 1.5, t)
                self.lambda_abun = _cosine_interp(5.0, 8.0, t)
            
            from tqdm import tqdm
            bar = tqdm(self.train_loader, desc="Batch", dynamic_ncols=True, disable=not self.is_main)
            
            running = 0.0
            running_loss_main = 0.0
            running_pres_accuracy = 0.0
            running_pres1_accuracy = 0.0
            running_pres_prob_mean = 0.0
            
            for i, batch in enumerate(bar):
                batch = batch.to(self.device, non_blocking=True)
                
                # 添加条件处理
                batch = self._prepare_batch_with_conditions(batch)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.separated_modeling:
                        model_output = self.model(batch)
                        loss, loss_details = self._separated_loss_fn(batch, model_output)
                        running_loss_main += loss_details["abun_loss"]
                        running_pres_accuracy += loss_details.get("pres_accuracy", 0.0)
                        running_pres1_accuracy += loss_details.get("pres1_accuracy", 0.0)
                        running_pres_prob_mean += loss_details.get("pres_prob_mean", 0.0)
                    else:
                        # 标准diffusion分支（如果需要的话）
                        B = batch.x0_abun.size(0) if hasattr(batch, 'x0_abun') else batch.noise.size(0)
                        t = torch.randint(0, len(self.betas), (B,), device=self.device)
                        alpha_t = self.alpha_t[t].view(-1, 1)
                        sigma_t = self.sigma_t[t].view(-1, 1)
                        
                        x0 = batch.x0_abun if hasattr(batch, 'x0_abun') else batch.noise
                        noise = torch.randn_like(x0)
                        xt = alpha_t * x0 + sigma_t * noise
                        snr = (alpha_t * alpha_t) / (sigma_t * sigma_t + 1e-12)
                        
                        if self.prediction_type == "v":
                            v_pred = self.model(xt, t).float()
                        else:
                            eps_hat = self.model(xt, t).float()
                            v_pred = self._v_from_eps_x0(eps_hat, x0, alpha_t, sigma_t)
                        
                        v_true = self._v_from_eps_x0(noise, x0, alpha_t, sigma_t).float()
                        
                        loss_dict = self._v_pred_loss_fn(v_pred=v_pred, v_true=v_true, snr=snr.view(-1))
                        loss = loss_dict["total"]
                        running_loss_main += loss_dict["loss_main"].item()
                
                # 反向传播和优化（与父类相同）
                self.scaler.scale(loss).backward()
                
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self._unwrap(self.model).parameters(), self.max_grad_norm)
                
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                
                self.scheduler.step()
                self._step_ema()
                
                running += loss.item()
                global_step += 1
                
                if self.is_main:
                    if self.separated_modeling:
                        bar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            abun=f"{running_loss_main/(i+1):.4f}",
                            acc=f"{running_pres_accuracy/(i+1):.3f}",
                            acc1=f"{running_pres1_accuracy/(i+1):.3f}",
                            prob=f"{running_pres_prob_mean/(i+1):.3f}",
                            lr=f"{self.opt.param_groups[0]['lr']:.2e}"
                        )
                    else:
                        bar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            loss_main=f"{running_loss_main/(i+1):.4f}",
                            lr=f"{self.opt.param_groups[0]['lr']:.2e}"
                        )
            
            # epoch平均值（与父类相同）
            avg_loss = running / self.length_train_loader
            avg_loss_main = running_loss_main / self.length_train_loader
            
            if self.is_main:
                if self.separated_modeling:
                    avg_pres_accuracy = running_pres_accuracy / self.length_train_loader
                    avg_pres1_accuracy = running_pres1_accuracy / self.length_train_loader
                    avg_pres_prob_mean = running_pres_prob_mean / self.length_train_loader
                    from tqdm import tqdm
                    tqdm.write(f"[Epoch {epoch:03d}] train: mean_loss={avg_loss:.5f} abun={avg_loss_main:.5f}")
                    tqdm.write(f"[Epoch {epoch:03d}] train: pres_acc={avg_pres_accuracy:.3f} pres1_acc={avg_pres1_accuracy:.3f} pres_prob={avg_pres_prob_mean:.3f}")
                    
                    self._tb_add_scalar('Train/Loss_Total', avg_loss, epoch)
                    self._tb_add_scalar('Train/Loss_Abundance', avg_loss_main, epoch)
                    self._tb_add_scalar('Train/Presence_Accuracy', avg_pres_accuracy, epoch)
                    self._tb_add_scalar('Train/Pres1_Accuracy', avg_pres1_accuracy, epoch)
                    self._tb_add_scalar('Train/Presence_Prob_Mean', avg_pres_prob_mean, epoch)
                else:
                    from tqdm import tqdm
                    tqdm.write(f"[Epoch {epoch:03d}] train: mean_loss={avg_loss:.5f} loss_main={avg_loss_main:.5f}")
                    self._tb_add_scalar('Train/Loss_Total', avg_loss, epoch)
                    self._tb_add_scalar('Train/Loss_Main', avg_loss_main, epoch)
            
            # 验证（与父类相同）
            val_results = None
            if self.val_loader is not None:
                val_batches = int(self.cfg.get("val_num_batches", 20))
                val_results = self.evaluate_noise_matching_gate(loader=self.val_loader, num_batches=val_batches)
                
                if self.is_main:
                    self._tb_add_scalar('Val/Spearman', 0.0 if self._is_nan(val_results["spearman"]) else val_results["spearman"], epoch)
                    self._tb_add_scalar('Val/Pearson', 0.0 if self._is_nan(val_results["correlation"]) else val_results["correlation"], epoch)
                    if self.separated_modeling:
                        self._tb_add_scalar('Val/Pres_Accuracy', 0.0 if self._is_nan(val_results["pres_accuracy"]) else val_results["pres_accuracy"], epoch)
                        self._tb_add_scalar('Val/Pres1_Accuracy', 0.0 if self._is_nan(val_results.get("pres1_accuracy", float('nan'))) else val_results["pres1_accuracy"], epoch)
                        self._tb_add_scalar('Val/Abun_MSE', 0.0 if self._is_nan(val_results["abun_mse"]) else val_results["abun_mse"], epoch)
                    else:
                        self._tb_add_scalar('Val/V_MSE', 0.0 if self._is_nan(val_results["v_mse"]) else val_results["v_mse"], epoch)
                    
                    from tqdm import tqdm
                    tqdm.write(f"[Epoch {epoch:03d}] val: Spearman={val_results['spearman']:.4f}, Pearson={val_results['correlation']:.4f}")
            
            # 保存检查点（与父类相同）
            if self.is_main:
                if self.separated_modeling:
                    if val_results is not None:
                        current_metrics = {
                            'abun_mse': val_results['abun_mse'],
                            'pres1_accuracy': val_results.get('pres1_accuracy', 0.0)
                        }
                    else:
                        current_metrics = None
                    self._save_ckpt(epoch, avg_loss, current_metrics)
                else:
                    self._save_ckpt(epoch, avg_loss, {})
        
        # 训练结束后的评估（与父类相同）
        if self.is_main:
            print("\n" + "="*60)
            print("条件扩散训练结束，开始最终评估...")
        _ = self.evaluate_training_stage()
        if self.is_main:
            print("="*60 + "\n")
        self.close()
    
    def _is_nan(self, value):
        """检查值是否为NaN"""
        import math
        return math.isnan(value) if isinstance(value, float) else False 
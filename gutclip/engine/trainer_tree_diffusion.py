import math, time, json, pathlib, torch
import torch.nn.functional as F
from torch.cuda.amp      import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional, Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def build_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_ratio: float = 0.02,  # 最低学习率占初始 lr 的比例
):
    """
    构建带 warmup 和 min_lr 的 cosine 衰减调度器

    Args:
        optimizer: torch.optim 优化器
        num_training_steps: 总共的 step 数
        num_warmup_steps: warmup 步数
        min_lr_ratio: 衰减到的最低学习率相对于初始 lr 的比例

    Returns:
        LambdaLR 调度器
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1 - min_lr_ratio) * (current_step / num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class TreeDiffusionTrainer:
    """Simple trainer with AMP, cosine LR, optional EMA."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        betas: torch.Tensor,
        cfg: Dict[str, Any],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.model, self.loader, self.opt = model, loader, optimizer
        self.device = device
        self.epochs = epochs
        self.cfg = cfg
        
        # 分离建模参数
        train_cfg = cfg.get("train", {})
        self.separated_modeling = train_cfg.get("separated_modeling", False)
        self.lambda_abun = train_cfg.get("lambda_abun", 1.0)
        self.lambda_pres = train_cfg.get("lambda_pres", 1.0)
        

        # LR scheduler (linear warm-up + cosine) - 按step调用

        # 最低学习率比例 - 保持初始 lr 的 2%
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.02)
        
        # warmup 参数
        warmup_ratio = train_cfg.get("warmup_ratio", 0.1)

        # 打印学习率信息
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] 初始学习率: {initial_lr}")
        print(f"[INFO] 最低学习率比例: {min_lr_ratio}")
        print(f"[INFO] 最低学习率: {initial_lr * min_lr_ratio}")
        
        # 计算总步数和warmup步数
        total_steps = len(loader) * epochs
        warmup_steps = int(warmup_ratio * total_steps)
        print(f"[INFO] 总步数: {total_steps}")
        print(f"[INFO] Warmup步数: {warmup_steps}")

        self.scheduler = build_cosine_schedule_with_warmup(
            optimizer=optimizer,  # 修复：使用传入的optimizer
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio  # 修复：使用局部变量
        )

        # ----- optional EMA -----
        self.use_ema   = train_cfg.get("use_ema", False)  # 暂时关闭EMA
        self.ema_decay = train_cfg.get("ema_decay", 0.9999)
        if self.use_ema:
            self.ema_model = type(model)()  # fresh weights
            self.ema_model.load_state_dict(model.state_dict())
            self.ema_model.to(device).eval()

        # output dir
        self.ckpt_dir = pathlib.Path("checkpoints/tree_diffusion")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP 开关
        use_amp = train_cfg.get("amp", True)
        self.scaler = GradScaler(enabled=use_amp)
        
        # 跟踪最佳模型
        self.best_loss = float('inf')
        self.best_ckpt_path = None
        
        # Diffusion训练参数
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alpha_t = torch.sqrt(self.alphas_cumprod).to(device)
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        
        # 简化的训练策略参数 - 先关闭所有校准项
        self.gamma_min_snr = train_cfg.get("gamma_min_snr", 6.0)
        
        # 校准项权重 - 全部设为0，逐步开启
        self.lambda_high = 0.0  # 暂时关闭高SNR校准
        self.lambda_mag = 0.0   # 暂时关闭幅度校准
        self.lambda_ang = 0.0   # 暂时关闭角度校准
        self.lambda_norm = 0.0  # 暂时关闭范数校准
        self.lambda_ortho = 0.0 # 暂时关闭正交校准
        
        # 门函数参数
        self.high_center = train_cfg.get("high_snr_center", 5.9)
        self.high_tau = train_cfg.get("high_snr_tau", 0.6)
        
        print(f"[INFO] 简化训练策略：只保留主损失+SNR权重")
        print(f"[INFO] 校准项权重：high={self.lambda_high}, mag={self.lambda_mag}, ang={self.lambda_ang}, norm={self.lambda_norm}, ortho={self.lambda_ortho}")
        
        # TensorBoard writer - 使用实验名称和时间戳
        experiment_name = cfg.get("experiment_name", "tree_diffusion")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.tb_log_dir = f"runs/{experiment_name}_{timestamp}"
        self.tb_writer = SummaryWriter(self.tb_log_dir)
        print(f"[INFO] TensorBoard日志保存在: {self.tb_log_dir}")
        print(f"[INFO] 实验名称: {experiment_name}")
        print(f"[INFO] 启动TensorBoard: tensorboard --logdir=runs")
        print(f"[INFO] 对比多次训练: 在TensorBoard中选择不同的运行进行对比")
        
        # 记录实验配置到TensorBoard
        self._log_experiment_config()

    # --------------------------------------------------
    def _step_ema(self):
        if not self.use_ema:
            return
        with torch.no_grad():
            for p_ema, p in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    # --------------------------------------------------
    def _v_from_eps_x0(self, eps: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """从ε和x0计算v"""
        return alpha * eps - sigma * x0
    
    def _eps_from_v_xt(self, v: torch.Tensor, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """从v和xt计算ε"""
        denom = alpha.pow(2) + sigma.pow(2)
        return (sigma * xt + alpha * v) / (denom + 1e-12)
    
    def _x0_from_v_xt(self, v: torch.Tensor, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """从v和xt计算x0"""
        denom = alpha.pow(2) + sigma.pow(2)
        return (alpha * xt - sigma * v) / (denom + 1e-12)
    
    def _cos_sim(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """计算余弦相似度"""
        a_2d = a.view(a.size(0), -1) if a.dim() > 2 else a
        b_2d = b.view(b.size(0), -1) if b.dim() > 2 else b
        num = (a_2d * b_2d).sum(-1)
        den = a_2d.norm(dim=-1) * b_2d.norm(dim=-1) + eps
        return num / den
    
    def _k_scale(self, eps_hat: torch.Tensor, eps_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """计算缩放因子k"""
        a_2d = eps_hat.view(eps_hat.size(0), -1) if eps_hat.dim() > 2 else eps_hat
        b_2d = eps_true.view(eps_true.size(0), -1) if eps_true.dim() > 2 else eps_true
        dot = (a_2d * b_2d).sum(-1)
        den = (b_2d * b_2d).sum(-1) + eps
        return dot / den

    # --------------------------------------------------
    def _loss_fn(self, eps_hat: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """简单的L2 loss（用于对比）"""
        return torch.mean((eps_hat - eps) ** 2)
    
    def _v_pred_loss_fn(self, v_pred: torch.Tensor, v_true: torch.Tensor, snr: torch.Tensor, 
                        xt: torch.Tensor, eps_true: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor,
                        epoch: int, enable_high_cal: bool = False) -> Dict[str, torch.Tensor]:
        """简化的v-prediction loss，只保留主损失+SNR权重"""
        
        # ---- SNR加权策略 ----
        if epoch < 2:
            # Ep<2: 统一权重，先把高段样本量"喂饱"
            w_main = torch.ones_like(snr)
        else:
            # Ep>=2: 温和上权高段
            w_main = snr / (snr + self.gamma_min_snr)  # 单调递增，饱和于1
            w_main = w_main / (w_main.mean() + 1e-8)  # 归一到均值≈1

        # 主损失：v-prediction MSE
        mse_v = ((v_pred - v_true) ** 2).mean(dim=tuple(range(1, v_pred.ndim)))
        loss_main = (w_main * mse_v).mean()

        # 计算 ε̂ 与 x̂0（用于监控）
        eps_hat = self._eps_from_v_xt(v_pred, xt, alpha, sigma)
        x0_hat = self._x0_from_v_xt(v_pred, xt, alpha, sigma)

        # 高 SNR 校准（暂时关闭）
        loss_cal = v_pred.new_tensor(0.0)
        cal_details = {
            "mag_loss": v_pred.new_tensor(0.0),
            "ang_loss": v_pred.new_tensor(0.0),
            "norm_loss": v_pred.new_tensor(0.0),
            "orth_loss": v_pred.new_tensor(0.0),
            "gate_mean": v_pred.new_tensor(0.0)
        }
        
        if enable_high_cal and self.lambda_high > 0:
            logsnr = torch.log((alpha * alpha) / (sigma * sigma + 1e-12) + 1e-12).view(-1)
            gate = torch.sigmoid((logsnr - self.high_center) / self.high_tau)
            gate = torch.clamp((gate ** 1.5) * 1.3, 0.0, 1.0)

            # 幅度校准
            k = self._k_scale(eps_hat, eps_true)
            mag_loss = (k - 1.0) ** 2
            
            # 角度校准
            cos = self._cos_sim(eps_hat, eps_true)
            ang_loss = (1.0 - cos).pow(2)
            
            # 范数校准
            eps_ = 1e-12
            n_hat = eps_hat.view(eps_hat.size(0), -1).norm(dim=-1) + eps_
            n_true = eps_true.view(eps_true.size(0), -1).norm(dim=-1) + eps_
            norm_loss = (torch.log(n_hat) - torch.log(n_true)) ** 2

            # 方向正交项
            a2d = eps_hat.view(eps_hat.size(0), -1)
            b2d = eps_true.view(eps_true.size(0), -1)
            dot = (a2d * b2d).sum(-1, keepdim=True)
            den = (b2d * b2d).sum(-1, keepdim=True).clamp_min(1e-12)
            proj = (dot / den) * b2d
            orth = a2d - proj
            orth_loss = (orth.pow(2).sum(-1) / (b2d.pow(2).sum(-1).clamp_min(1e-12))).mean()

            cal = (gate * (self.lambda_mag * mag_loss +
                          self.lambda_ang * ang_loss +
                          self.lambda_norm * norm_loss +
                          self.lambda_ortho * orth_loss)).mean()
            loss_cal = self.lambda_high * cal
            
            cal_details = {
                "mag_loss": mag_loss.detach(),
                "ang_loss": ang_loss.detach(),
                "norm_loss": norm_loss.detach(),
                "orth_loss": orth_loss.detach(),
                "gate_mean": gate.mean().detach()
            }

        total = loss_main + loss_cal
        
        return {
            "total": total,
            "loss_main": loss_main.detach(),
            "loss_cal": loss_cal.detach(),
            "w_main_mean": w_main.mean().detach(),
            "snr_mean": snr.mean().detach(),
            "cal_details": cal_details
        }
    
    def _separated_loss_fn(self, batch, model_output):
        """分离建模损失：Bernoulli + Gaussian"""
        # 提取数据
        x0_abun = batch.x0_abun           # (ΣN, 1) log_abun
        x0_pres = batch.x0_pres           # (ΣN, 1) presence
        noise = batch.noise               # (ΣN, 1) 高斯噪声
        
        # 模型输出
        eps_hat = model_output["eps_hat"]      # 连续 ε̂
        pres_logit = model_output["pres_logit"]   # 对 presence=1 的 logits

        # Presence流程校验 - 确保在同一计算图上
        pres_prob = torch.sigmoid(pres_logit)  # logits → 概率
        pres_sampled = (pres_prob > 0.5).float()  # 概率 → 采样 (用于监控)
        
        # Mask for presence == 1 nodes (positive class)
        mask = x0_pres.squeeze(-1).float()

        # Presence BCE with positive-class weighting to alleviate imbalance
        pos_weight = ((mask.numel() - mask.sum()) / mask.sum().clamp_min(1)).to(pres_logit.device)
        loss_pres = F.binary_cross_entropy_with_logits(
            pres_logit, mask, pos_weight=pos_weight)

        # Abundance MSE (只在 presence==1 的节点)
        loss_abun = ((eps_hat - noise.squeeze(-1))**2 * mask).sum() \
                    / mask.sum().clamp_min(1)

        # 总损失 - 调整权重平衡
        loss = self.lambda_pres * loss_pres + self.lambda_abun * loss_abun
        
        # 监控指标 - 修复accuracy计算
        # 计算所有节点的presence准确率，不只是presence=1的节点
        all_pres_accuracy = (pres_sampled == x0_pres.squeeze(-1)).float().mean()
        # 计算presence=1节点的准确率（用于调试）
        pres1_accuracy = ((pres_sampled == x0_pres.squeeze(-1)).float() * mask).sum() / mask.sum().clamp_min(1)
        
        return loss, {
            "abun_loss": loss_abun.item(), 
            "pres_loss": loss_pres.item(),
            "pres_accuracy": all_pres_accuracy.item(),  # 使用所有节点的准确率
            "pres1_accuracy": pres1_accuracy.item(),   # presence=1节点的准确率
            "pres_prob_mean": pres_prob.mean().item(),
            "pres1_ratio": mask.mean().item()  # presence=1的比例
        }

    # --------------------------------------------------
    def fit(self):
        self.model.to(self.device)
        global_step = 0
        
        # 记录训练历史
        train_history = {
            "epoch": [],
            "loss_main": [],
            "loss_cal": [],
            "w_main_mean": [],
            "snr_mean": [],
            "cal_mag": [],
            "cal_ang": [],
            "cal_norm": [],
            "cal_ortho": [],
            "gate_mean": []
        }
        
        for epoch in tqdm(range(self.epochs), desc="Epoch", dynamic_ncols=True):
            self.model.train()
            running = 0.0
            running_loss_main = 0.0
            running_loss_cal = 0.0
            running_w_main = 0.0
            running_snr = 0.0
            running_cal_mag = 0.0
            running_cal_ang = 0.0
            running_cal_norm = 0.0
            running_cal_ortho = 0.0
            running_gate_mean = 0.0
            
            # Presence监控指标
            running_pres_accuracy = 0.0
            running_pres_prob_mean = 0.0
            
            bar = tqdm(self.loader, desc="Batch", dynamic_ncols=True)
            
            # 启用高SNR校准的epoch（暂时关闭）
            enable_high_cal = False  # epoch >= 2
            
            for i, batch in enumerate(bar):
                # 确保数据在正确的设备上
                batch = batch.to(self.device, non_blocking=True)

                with autocast(enabled=self.cfg.get("amp", True)):
                    if self.separated_modeling:
                        # 分离建模
                        model_output = self.model(batch)
                        loss, loss_details = self._separated_loss_fn(batch, model_output)
                        running_loss_main += loss_details["abun_loss"]
                        running_loss_cal += loss_details["pres_loss"]
                        
                        # 记录presence监控指标
                        if "pres_accuracy" in loss_details:
                            running_pres_accuracy += loss_details["pres_accuracy"]
                        if "pres_prob_mean" in loss_details:
                            running_pres_prob_mean += loss_details["pres_prob_mean"]
                    else:
                        # 改进的diffusion建模
                        # 获取时间步和噪声
                        B = batch.x0_abun.size(0) if hasattr(batch, 'x0_abun') else batch.noise.size(0)
                        t = torch.randint(0, len(self.betas), (B,), device=self.device)
                        alpha_t = self.alpha_t[t].view(-1, 1)
                        sigma_t = self.sigma_t[t].view(-1, 1)
                        
                        # 获取原始数据（假设是x0）
                        if hasattr(batch, 'x0_abun'):
                            x0 = batch.x0_abun  # 使用丰度数据
                        else:
                            x0 = batch.noise  # 回退到噪声数据
                        
                        # 添加噪声
                        noise = torch.randn_like(x0)
                        xt = alpha_t * x0 + sigma_t * noise
                        
                        # 计算SNR
                        snr = (alpha_t * alpha_t) / (sigma_t * sigma_t + 1e-12)
                        
                        # 模型预测
                        eps_hat = self.model(xt, t)  # 假设模型输出eps_hat
                        
                        # 计算v_true和v_pred
                        v_true = self._v_from_eps_x0(noise, x0, alpha_t, sigma_t)
                        v_pred = self._v_from_eps_x0(eps_hat, x0, alpha_t, sigma_t)
                        
                        # 使用改进的loss
                        loss_dict = self._v_pred_loss_fn(
                            v_pred=v_pred,
                            v_true=v_true,
                            snr=snr.view(-1),
                            xt=xt,
                            eps_true=noise,
                            alpha=alpha_t,
                            sigma=sigma_t,
                            epoch=epoch,
                            enable_high_cal=enable_high_cal
                        )
                        
                        loss = loss_dict["total"]
                        running_loss_main += loss_dict["loss_main"].item()
                        running_loss_cal += loss_dict["loss_cal"].item()
                        running_w_main += loss_dict["w_main_mean"].item()
                        running_snr += loss_dict["snr_mean"].item()
                        
                        # 校准详情
                        cal_details = loss_dict["cal_details"]
                        running_cal_mag += cal_details["mag_loss"].item()
                        running_cal_ang += cal_details["ang_loss"].item()
                        running_cal_norm += cal_details["norm_loss"].item()
                        running_cal_ortho += cal_details["orth_loss"].item()
                        running_gate_mean += cal_details["gate_mean"].item()

                self.scaler.scale(loss).backward()
                
                # 记录梯度范数（用于监控训练稳定性）
                if i % 50 == 0:  # 每50步记录一次
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    self.tb_writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
                    
                    # 记录训练进度
                    progress = (epoch * len(self.loader) + i) / (self.epochs * len(self.loader))
                    self.tb_writer.add_scalar('Training/Progress', progress, global_step)
                
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

                # 按step调用调度器
                self.scheduler.step()
                self._step_ema()

                running += loss.item()
                global_step += 1
                
                if self.separated_modeling:
                    bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        abun=f"{loss_details['abun_loss']:.4f}",
                        pres=f"{loss_details['pres_loss']:.4f}",
                        acc=f"{loss_details.get('pres_accuracy', 0):.3f}",
                        acc1=f"{loss_details.get('pres1_accuracy', 0):.3f}",
                        ratio=f"{loss_details.get('pres1_ratio', 0):.3f}",
                        prob=f"{loss_details.get('pres_prob_mean', 0):.3f}",
                        lr=f"{self.opt.param_groups[0]['lr']:.2e}"
                    )
                else:
                    bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        loss_main=f"{loss_dict['loss_main']:.4f}",
                        loss_cal=f"{loss_dict['loss_cal']:.4f}",
                        w_main=f"{loss_dict['w_main_mean']:.3f}",
                        snr=f"{loss_dict['snr_mean']:.2f}",
                        lr=f"{self.opt.param_groups[0]['lr']:.2e}"
                    )

            # 计算平均值
            avg_loss = running / len(self.loader)
            avg_loss_main = running_loss_main / len(self.loader)
            avg_loss_cal = running_loss_cal / len(self.loader)
            avg_w_main = running_w_main / len(self.loader)
            avg_snr = running_snr / len(self.loader)
            avg_cal_mag = running_cal_mag / len(self.loader)
            avg_cal_ang = running_cal_ang / len(self.loader)
            avg_cal_norm = running_cal_norm / len(self.loader)
            avg_cal_ortho = running_cal_ortho / len(self.loader)
            avg_gate_mean = running_gate_mean / len(self.loader)
            
            # 记录历史
            train_history["epoch"].append(epoch)
            train_history["loss_main"].append(avg_loss_main)
            train_history["loss_cal"].append(avg_loss_cal)
            train_history["w_main_mean"].append(avg_w_main)
            train_history["snr_mean"].append(avg_snr)
            train_history["cal_mag"].append(avg_cal_mag)
            train_history["cal_ang"].append(avg_cal_ang)
            train_history["cal_norm"].append(avg_cal_norm)
            train_history["cal_ortho"].append(avg_cal_ortho)
            train_history["gate_mean"].append(avg_gate_mean)
            
            if self.separated_modeling:
                avg_pres_accuracy = running_pres_accuracy / len(self.loader)
                avg_pres_prob_mean = running_pres_prob_mean / len(self.loader)
                tqdm.write(f"[Epoch {epoch:03d}] mean_loss={avg_loss:.5f} abun={avg_loss_main:.5f} pres={avg_loss_cal:.5f}")
                tqdm.write(f"[Epoch {epoch:03d}] pres_acc={avg_pres_accuracy:.3f} pres_prob={avg_pres_prob_mean:.3f}")
                
                # TensorBoard日志 - 分离建模
                self.tb_writer.add_scalar('Loss/Total', avg_loss, epoch)
                self.tb_writer.add_scalar('Loss/Abundance', avg_loss_main, epoch)
                self.tb_writer.add_scalar('Loss/Presence', avg_loss_cal, epoch)
                self.tb_writer.add_scalar('Metrics/Presence_Accuracy', avg_pres_accuracy, epoch)
                self.tb_writer.add_scalar('Metrics/Presence_Prob_Mean', avg_pres_prob_mean, epoch)
                self.tb_writer.add_scalar('Learning_Rate', self.opt.param_groups[0]['lr'], epoch)
                
            else:
                tqdm.write(f"[Epoch {epoch:03d}] mean_loss={avg_loss:.5f} loss_main={avg_loss_main:.5f} loss_cal={avg_loss_cal:.5f} w_main={avg_w_main:.3f} snr={avg_snr:.2f}")
                if avg_gate_mean > 0:
                    tqdm.write(f"[Epoch {epoch:03d}] cal_details: mag={avg_cal_mag:.4f} ang={avg_cal_ang:.4f} norm={avg_cal_norm:.4f} ortho={avg_cal_ortho:.4f} gate={avg_gate_mean:.3f}")
                
                # TensorBoard日志 - 标准diffusion建模
                self.tb_writer.add_scalar('Loss/Total', avg_loss, epoch)
                self.tb_writer.add_scalar('Loss/Main', avg_loss_main, epoch)
                self.tb_writer.add_scalar('Loss/Calibration', avg_loss_cal, epoch)
                self.tb_writer.add_scalar('Training/Weight_Mean', avg_w_main, epoch)
                self.tb_writer.add_scalar('Training/SNR_Mean', avg_snr, epoch)
                self.tb_writer.add_scalar('Learning_Rate', self.opt.param_groups[0]['lr'], epoch)
                
                # 校准详情（如果启用）
                if avg_gate_mean > 0:
                    self.tb_writer.add_scalar('Calibration/Magnitude', avg_cal_mag, epoch)
                    self.tb_writer.add_scalar('Calibration/Angle', avg_cal_ang, epoch)
                    self.tb_writer.add_scalar('Calibration/Norm', avg_cal_norm, epoch)
                    self.tb_writer.add_scalar('Calibration/Orthogonal', avg_cal_ortho, epoch)
                    self.tb_writer.add_scalar('Calibration/Gate_Mean', avg_gate_mean, epoch)

            # 保存checkpoint（借鉴其他trainer的最佳实践）
            self._save_ckpt(epoch, avg_loss)
            
            # 每10个epoch保存训练历史
            if (epoch + 1) % 10 == 0:
                self._save_training_history(train_history)
        
        # 训练结束，评估两个关卡
        print("\n" + "="*60)
        print("训练结束，开始评估两个关卡...")
        evaluation_results = self.evaluate_training_stage()
        print("="*60 + "\n")
        
        # 关闭TensorBoard
        self.close()
    
    # --------------------------------------------------
    def _save_training_history(self, history: Dict[str, list]):
        """保存训练历史到JSON文件"""
        history_path = self.ckpt_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[INFO] 训练历史已保存到: {history_path}")
    
    def _save_ckpt(self, epoch: int, loss: float):
        """保存checkpoint，借鉴其他trainer的最佳实践：
        1. 总是保存latest checkpoint
        2. 只有当loss改善时才保存best checkpoint，并删除旧的best文件
        """
        obj = {
            "epoch": epoch + 1,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss": loss,
            "cfg": self.cfg,
        }
        if self.use_ema:
            obj["ema"] = self.ema_model.state_dict()
        
        # 保存latest checkpoint
        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(obj, latest_path)
        
        # 更新latest.json指针
        (self.ckpt_dir / "latest.json").write_text(json.dumps({"path": str(latest_path)}))
        
        # 检查是否需要保存best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            
            # 删除旧的best checkpoint
            if self.best_ckpt_path and self.best_ckpt_path.exists():
                self.best_ckpt_path.unlink(missing_ok=True)
                print(f"[INFO] 删除旧的best checkpoint: {self.best_ckpt_path.name}")
            
            # 保存新的best checkpoint
            best_path = self.ckpt_dir / f"best_loss{loss:.4f}.pt"
            torch.save(obj, best_path)
            self.best_ckpt_path = best_path
            print(f"[✓] 新的best checkpoint: {best_path.name} (loss={loss:.4f})")
            
            # 记录最佳模型信息到TensorBoard
            self.tb_writer.add_scalar('Model/Best_Loss', loss, epoch)
            self.tb_writer.add_text('Model/Best_Checkpoint', str(best_path), epoch)
    
    def _log_experiment_config(self):
        """记录实验配置到TensorBoard"""
        config_text = f"""
            实验配置:
            - 分离建模: {self.separated_modeling}
            - λ_abun: {self.lambda_abun}
            - λ_pres: {self.lambda_pres}
            - 训练轮数: {self.epochs}
            - 设备: {self.device}
            - 校准项权重: high={self.lambda_high}, mag={self.lambda_mag}, ang={self.lambda_ang}, norm={self.lambda_norm}, ortho={self.lambda_ortho}
            - 门函数参数: center={self.high_center}, tau={self.high_tau}
        """
        self.tb_writer.add_text('Experiment/Config', config_text, 0)
        print("[INFO] 实验配置已记录到TensorBoard")
    
    def close(self):
        """关闭TensorBoard writer"""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
            print("[INFO] TensorBoard writer已关闭")

    def evaluate_noise_matching_gate(self, num_batches: int = 20) -> Dict[str, float]:
        """评估噪声匹配关卡 - 判断训练阶段是否成功"""
        self.model.eval()
        
        total_noise_mse = 0.0
        total_v_mse = 0.0
        total_corr = 0.0
        total_pres_accuracy = 0.0
        total_abun_mse = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                if i >= num_batches:
                    break
                    
                batch = batch.to(self.device, non_blocking=True)
                
                if self.separated_modeling:
                    # 分离建模的噪声匹配评估
                    model_output = self.model(batch)
                    eps_hat = model_output["eps_hat"]
                    noise = batch.noise
                    
                    # 计算噪声MSE
                    noise_mse = F.mse_loss(eps_hat, noise.squeeze(-1))
                    total_noise_mse += noise_mse.item()
                    
                    # 计算相关系数
                    corr = torch.corrcoef(torch.stack([eps_hat.flatten(), noise.squeeze(-1).flatten()]))[0, 1]
                    total_corr += corr.item()
                    
                    # 计算presence准确率
                    pres_logit = model_output["pres_logit"]
                    pres_prob = torch.sigmoid(pres_logit)
                    pres_accuracy = ((pres_prob > 0.5).float() == batch.x0_pres.squeeze(-1).float()).float().mean()
                    total_pres_accuracy += pres_accuracy.item()
                    
                    # 计算abundance MSE（只在presence=1的节点）
                    mask = batch.x0_pres.squeeze(-1).float()
                    if mask.sum() > 0:
                        abun_mse = ((eps_hat - noise.squeeze(-1))**2 * mask).sum() / mask.sum()
                        total_abun_mse += abun_mse.item()
                    
                else:
                    # 标准diffusion建模的噪声匹配评估
                    B = batch.x0_abun.size(0) if hasattr(batch, 'x0_abun') else batch.noise.size(0)
                    t = torch.randint(0, len(self.betas), (B,), device=self.device)
                    alpha_t = self.alpha_t[t].view(-1, 1)
                    sigma_t = self.sigma_t[t].view(-1, 1)
                    
                    if hasattr(batch, 'x0_abun'):
                        x0 = batch.x0_abun
                    else:
                        x0 = batch.noise
                    
                    noise = torch.randn_like(x0)
                    xt = alpha_t * x0 + sigma_t * noise
                    
                    # 模型预测
                    eps_hat = self.model(xt, t)
                    v_pred = self._v_from_eps_x0(eps_hat, x0, alpha_t, sigma_t)
                    v_true = self._v_from_eps_x0(noise, x0, alpha_t, sigma_t)
                    
                    # 计算v-prediction MSE
                    v_mse = F.mse_loss(v_pred, v_true)
                    total_v_mse += v_mse.item()
                    
                    # 计算相关系数
                    corr = torch.corrcoef(torch.stack([v_pred.flatten(), v_true.flatten()]))[0, 1]
                    total_corr += corr.item()
                
                num_samples += 1
        
        # 计算平均指标
        avg_noise_mse = total_noise_mse / num_samples
        avg_v_mse = total_v_mse / num_samples
        avg_corr = total_corr / num_samples
        avg_pres_accuracy = total_pres_accuracy / num_samples
        avg_abun_mse = total_abun_mse / num_samples
        
        # 评估结果 - 更严格的阈值
        noise_mse_passed = avg_noise_mse <= 0.05  # 更严格的噪声MSE阈值
        v_mse_passed = avg_v_mse <= 0.05  # 更严格的v-prediction MSE阈值
        corr_passed = avg_corr >= 0.98  # 相关系数阈值
        pres_accuracy_passed = avg_pres_accuracy >= 0.9  # presence准确率阈值
        abun_mse_passed = avg_abun_mse <= 0.05  # abundance MSE阈值
        
        # 综合评估
        if self.separated_modeling:
            overall_passed = (noise_mse_passed and pres_accuracy_passed and abun_mse_passed)
        else:
            overall_passed = (noise_mse_passed and v_mse_passed and corr_passed)
        
        return {
            "noise_mse": avg_noise_mse,
            "v_mse": avg_v_mse,
            "correlation": avg_corr,
            "pres_accuracy": avg_pres_accuracy,
            "abun_mse": avg_abun_mse,
            "noise_mse_passed": noise_mse_passed,
            "v_mse_passed": v_mse_passed,
            "corr_passed": corr_passed,
            "pres_accuracy_passed": pres_accuracy_passed,
            "abun_mse_passed": abun_mse_passed,
            "overall_passed": overall_passed
        }
    
    def evaluate_shannon_fluctuation_gate(self, num_chains: int = 10, num_steps: int = None) -> Dict[str, float]:
        """评估香农波动关卡 - 使用相对波动率和完整反向链"""
        self.model.eval()
        
        if num_steps is None:
            num_steps = int(0.8 * len(self.betas))  # 至少80%的步数
        
        all_rel_fluctuations = []
        all_richness_diffs = []
        all_bray_curtis_dists = []
        all_pearson_corrs = []
        numerical_stable = True
        
        with torch.no_grad():
            for chain_idx in range(num_chains):
                try:
                    # 生成随机初始噪声
                    B = 1
                    x_t = torch.randn(B, self.y_dim if hasattr(self, 'y_dim') else 256, device=self.device)
                    
                    # 记录完整反向链的香农熵
                    shannon_entropies = []
                    
                    # 逐步去噪 - 完整反向链
                    for step in range(num_steps):
                        t = torch.tensor([step], device=self.device)
                        alpha_t = self.alpha_t[t].view(-1, 1)
                        sigma_t = self.sigma_t[t].view(-1, 1)
                        
                        # 模型预测
                        eps_hat = self.model(x_t, t)
                        
                        # 去噪步骤
                        x_t = (x_t - sigma_t * eps_hat) / alpha_t
                        
                        # 计算香农熵（数值稳定版）
                        x_clip = torch.clamp(x_t, -30, 30)
                        probs = torch.softmax(x_clip, dim=-1)
                        probs = torch.clamp(probs, min=1e-12)
                        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
                        shannon_entropies.append(entropy.item())
                    
                    # 计算相对波动率
                    if len(shannon_entropies) > 1:
                        max_entropy = max(shannon_entropies)
                        min_entropy = min(shannon_entropies)
                        mean_entropy = sum(shannon_entropies) / len(shannon_entropies)
                        
                        if mean_entropy > 0:
                            rel_fluctuation = (max_entropy - min_entropy) / mean_entropy
                            all_rel_fluctuations.append(rel_fluctuation)
                    
                    # 计算richness（非零物种数）
                    final_sample = x_t.squeeze(0)
                    richness = (final_sample > 0.01).sum().item()  # 阈值0.01
                    
                    # 与真实样本比较（这里需要真实样本数据）
                    # 暂时使用随机生成的参考样本
                    ref_sample = torch.randn_like(final_sample)
                    ref_richness = (ref_sample > 0.01).sum().item()
                    richness_diff = abs(richness - ref_richness) / max(ref_richness, 1)
                    all_richness_diffs.append(richness_diff)
                    
                    # 计算Bray-Curtis距离
                    bc_dist = self._bray_curtis_distance(final_sample, ref_sample)
                    all_bray_curtis_dists.append(bc_dist)
                    
                    # 计算Pearson相关系数
                    pearson_corr = torch.corrcoef(torch.stack([final_sample, ref_sample]))[0, 1].item()
                    all_pearson_corrs.append(pearson_corr)
                    
                    # 检查数值稳定性
                    if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                        numerical_stable = False
                    
                except Exception as e:
                    print(f"[WARN] 链 {chain_idx} 采样过程中出现错误: {e}")
                    numerical_stable = False
                    continue
        
        # 计算统计指标
        if all_rel_fluctuations:
            mean_rel_fluctuation = sum(all_rel_fluctuations) / len(all_rel_fluctuations)
            max_rel_fluctuation = max(all_rel_fluctuations)
        else:
            mean_rel_fluctuation = float('inf')
            max_rel_fluctuation = float('inf')
        
        if all_richness_diffs:
            mean_richness_diff = sum(all_richness_diffs) / len(all_richness_diffs)
        else:
            mean_richness_diff = float('inf')
        
        if all_bray_curtis_dists:
            mean_bc_dist = sum(all_bray_curtis_dists) / len(all_bray_curtis_dists)
        else:
            mean_bc_dist = float('inf')
        
        if all_pearson_corrs:
            mean_pearson_corr = sum(all_pearson_corrs) / len(all_pearson_corrs)
        else:
            mean_pearson_corr = -1.0
        
        # 评估结果 - 更严格的阈值
        rel_fluctuation_passed = mean_rel_fluctuation < 0.05  # 5%相对波动率
        richness_passed = mean_richness_diff < 0.1  # 10% richness差异
        bc_dist_passed = mean_bc_dist < 0.1  # Bray-Curtis距离阈值
        pearson_passed = mean_pearson_corr > 0.8  # Pearson相关系数阈值
        numerical_passed = numerical_stable
        
        overall_passed = (rel_fluctuation_passed and richness_passed and 
                         bc_dist_passed and pearson_passed and numerical_passed)
        
        return {
            "mean_rel_fluctuation": mean_rel_fluctuation,
            "max_rel_fluctuation": max_rel_fluctuation,
            "mean_richness_diff": mean_richness_diff,
            "mean_bc_dist": mean_bc_dist,
            "mean_pearson_corr": mean_pearson_corr,
            "numerical_stable": numerical_stable,
            "rel_fluctuation_passed": rel_fluctuation_passed,
            "richness_passed": richness_passed,
            "bc_dist_passed": bc_dist_passed,
            "pearson_passed": pearson_passed,
            "numerical_passed": numerical_passed,
            "overall_passed": overall_passed
        }
    
    def _bray_curtis_distance(self, sample1: torch.Tensor, sample2: torch.Tensor) -> float:
        """计算Bray-Curtis距离"""
        # 确保非负
        sample1 = torch.abs(sample1)
        sample2 = torch.abs(sample2)
        
        # 计算Bray-Curtis距离
        numerator = torch.sum(torch.abs(sample1 - sample2))
        denominator = torch.sum(sample1 + sample2)
        
        if denominator > 0:
            return (numerator / denominator).item()
        else:
            return 0.0
    
    def evaluate_training_stage(self) -> Dict[str, Any]:
        """评估训练阶段是否成功通过两个关卡"""
        print("[INFO] 开始评估训练阶段...")
        
        # 关卡1: 噪声匹配
        print("[INFO] 评估噪声匹配关卡...")
        noise_results = self.evaluate_noise_matching_gate()
        
        # 关卡2: 香农波动
        print("[INFO] 评估香农波动关卡...")
        shannon_results = self.evaluate_shannon_fluctuation_gate()
        
        # 综合评估
        stage1_passed = noise_results["overall_passed"] and shannon_results["overall_passed"]
        
        # 记录到TensorBoard
        self.tb_writer.add_scalar('Gates/Noise_MSE', noise_results["noise_mse"], 0)
        self.tb_writer.add_scalar('Gates/V_MSE', noise_results["v_mse"], 0)
        self.tb_writer.add_scalar('Gates/Correlation', noise_results["correlation"], 0)
        self.tb_writer.add_scalar('Gates/Pres_Accuracy', noise_results["pres_accuracy"], 0)
        self.tb_writer.add_scalar('Gates/Abun_MSE', noise_results["abun_mse"], 0)
        self.tb_writer.add_scalar('Gates/Rel_Fluctuation', shannon_results["mean_rel_fluctuation"], 0)
        self.tb_writer.add_scalar('Gates/Richness_Diff', shannon_results["mean_richness_diff"], 0)
        self.tb_writer.add_scalar('Gates/Bray_Curtis_Dist', shannon_results["mean_bc_dist"], 0)
        self.tb_writer.add_scalar('Gates/Pearson_Corr', shannon_results["mean_pearson_corr"], 0)
        self.tb_writer.add_scalar('Gates/Noise_Matching_Passed', int(noise_results["overall_passed"]), 0)
        self.tb_writer.add_scalar('Gates/Shannon_Passed', int(shannon_results["overall_passed"]), 0)
        self.tb_writer.add_scalar('Gates/Stage1_Passed', int(stage1_passed), 0)
        
        # 打印详细结果
        print(f"[关卡1] 噪声匹配: {'✓' if noise_results['overall_passed'] else '✗'}")
        print(f"  - 噪声MSE: {noise_results['noise_mse']:.4f} (阈值: ≤0.05)")
        if not self.separated_modeling:
            print(f"  - V-MSE: {noise_results['v_mse']:.4f} (阈值: ≤0.05)")
        print(f"  - 相关系数: {noise_results['correlation']:.4f} (阈值: ≥0.98)")
        if self.separated_modeling:
            print(f"  - Presence准确率: {noise_results['pres_accuracy']:.4f} (阈值: ≥0.9)")
            print(f"  - Abundance MSE: {noise_results['abun_mse']:.4f} (阈值: ≤0.05)")
        
        print(f"[关卡2] 香农波动: {'✓' if shannon_results['overall_passed'] else '✗'}")
        print(f"  - 相对波动率: {shannon_results['mean_rel_fluctuation']:.4f} (阈值: <0.05)")
        print(f"  - Richness差异: {shannon_results['mean_richness_diff']:.4f} (阈值: <0.1)")
        print(f"  - Bray-Curtis距离: {shannon_results['mean_bc_dist']:.4f} (阈值: <0.1)")
        print(f"  - Pearson相关系数: {shannon_results['mean_pearson_corr']:.4f} (阈值: >0.8)")
        print(f"  - 数值稳定: {'✓' if shannon_results['numerical_stable'] else '✗'}")
        
        print(f"[综合] 第一阶段训练: {'✓ 成功' if stage1_passed else '✗ 失败'}")
        
        return {
            "stage1_passed": stage1_passed,
            "noise_results": noise_results,
            "shannon_results": shannon_results
        }
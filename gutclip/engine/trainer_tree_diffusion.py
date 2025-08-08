#!/usr/bin/env python3
import math, time, json, pathlib, torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional, Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def build_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_ratio: float = 0.02,
):
    """Cosine decay with linear warmup, floored at (min_lr_ratio * initial_lr)."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1 - min_lr_ratio) * (current_step / max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


class TreeDiffusionTrainer:
    """Trainer with AMP, cosine LR, optional EMA. (No Shannon; eval uses val_loader)"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        betas: torch.Tensor,
        cfg: Dict[str, Any],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.length_train_loader = len(train_loader)
        self.length_val_loader = len(val_loader)
        self.opt = optimizer
        self.device = device
        self.epochs = epochs
        self.cfg = cfg

        # ---- training cfg ----
        train_cfg = cfg.get("train", {})
        self.separated_modeling = train_cfg.get("separated_modeling", False)
        self.lambda_abun = train_cfg.get("lambda_abun", 1.0)
        self.lambda_pres = train_cfg.get("lambda_pres", 1.0)

        # 明确 head 类型：'eps' | 'v'
        self.prediction_type = train_cfg.get("prediction_type", "eps").lower()
        assert self.prediction_type in ("eps", "v"), "prediction_type 必须是 'eps' 或 'v'"

        # ---- AMP ----
        self.use_amp = train_cfg.get("amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # ---- LR scheduler ----
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.05)
        warmup_ratio = train_cfg.get("warmup_ratio", 0.2)
        initial_lr = self.opt.param_groups[0]['lr']
        print(f"[INFO] 初始lr: {initial_lr} | min_lr_ratio: {min_lr_ratio} -> floor: {initial_lr*min_lr_ratio}")

        total_steps = self.length_train_loader * epochs
        warmup_steps = int(warmup_ratio * total_steps)
        print(f"[INFO] 总步数: {total_steps} | warmup步数: {warmup_steps}")

        self.scheduler = build_cosine_schedule_with_warmup(
            optimizer=self.opt,
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio
        )

        # ---- optional EMA ----
        self.use_ema = train_cfg.get("use_ema", False)
        self.ema_decay = train_cfg.get("ema_decay", 0.9999)
        if self.use_ema:
            self.ema_model = type(model)()
            self.ema_model.load_state_dict(model.state_dict())
            self.ema_model.to(device).eval()

        # ---- ckpt dir ----
        self.ckpt_dir = pathlib.Path("checkpoints/tree_diffusion")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---- grad clip ----
        self.max_grad_norm = self.cfg.get("max_grad_norm", 0.5)

        # ---- best trackers ----
        self.best_loss = float('inf')
        self.best_ckpt_path = None
        self.pres1_acc_threshold = train_cfg.get("pres1_acc_threshold", 0.95)
        self.best_abun_mse_at_pres1 = float("inf")

        # ---- diffusion schedule consts ----
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alpha_t = torch.sqrt(self.alphas_cumprod).to(device)
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        # ---- simple SNR weighting ----
        self.gamma_min_snr = train_cfg.get("gamma_min_snr", 6.0)

        # ---- TB ----
        experiment_name = cfg.get("experiment_name", "tree_diffusion")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.tb_log_dir = f"runs/{experiment_name}_{timestamp}"
        self.tb_writer = SummaryWriter(self.tb_log_dir)
        print(f"[INFO] TensorBoard: {self.tb_log_dir}")

        self._log_experiment_config()

    # ---------------------- EMA ----------------------
    def _step_ema(self):
        if not self.use_ema: return
        with torch.no_grad():
            for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    # ----------------- conversions -------------------
    def _v_from_eps_x0(self, eps: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return alpha * eps - sigma * x0

    # ------------------- losses ----------------------
    def _v_pred_loss_fn(self, v_pred: torch.Tensor, v_true: torch.Tensor, snr: torch.Tensor) -> Dict[str, torch.Tensor]:
        if v_pred.dim() > 2:
            mse_v = ((v_pred - v_true) ** 2).mean(dim=tuple(range(1, v_pred.ndim)))
        else:
            mse_v = (v_pred - v_true).pow(2).mean(dim=1)
        w = snr / (snr + self.gamma_min_snr)
        w = w / (w.mean() + 1e-8)
        loss_main = (w * mse_v).mean()
        return {
            "total": loss_main,
            "loss_main": loss_main.detach(),
            "w_main_mean": w.mean().detach(),
            "snr_mean": snr.mean().detach(),
        }

    def _separated_loss_fn(self, batch, model_output):
        x0_pres = batch.x0_pres           # (ΣN, 1)
        noise   = batch.noise             # (ΣN, 1)  # target for eps_hat

        eps_hat    = model_output["eps_hat"]       # continuous head
        pres_logit = model_output["pres_logit"]    # logits for presence

        pres_prob    = torch.sigmoid(pres_logit)
        pres_sampled = (pres_prob > 0.5).float()
        mask = x0_pres.squeeze(-1).float()

        pos_weight = ((mask.numel() - mask.sum()) / mask.sum().clamp_min(1)).to(pres_logit.device)
        loss_pres = F.binary_cross_entropy_with_logits(pres_logit, mask, pos_weight=pos_weight)

        loss_abun = ((eps_hat - noise.squeeze(-1))**2 * mask).sum() / mask.sum().clamp_min(1)
        loss = self.lambda_pres * loss_pres + self.lambda_abun * loss_abun

        all_pres_accuracy = (pres_sampled == x0_pres.squeeze(-1)).float().mean()
        pres1_accuracy = ((pres_sampled == x0_pres.squeeze(-1)).float() * mask).sum() / mask.sum().clamp_min(1)

        return loss, {
            "abun_loss": loss_abun.item(),
            "pres_loss": loss_pres.item(),
            "pres_accuracy": all_pres_accuracy.item(),
            "pres1_accuracy": pres1_accuracy.item(),
            "pres_prob_mean": pres_prob.mean().item(),
            "pres1_ratio": mask.mean().item()
        }

    # -------------------- train ----------------------
    def fit(self):
        self.model.to(self.device)
        global_step = 0

        # cosine interpolation for lambda
        def _cosine_interp(v_start: float, v_end: float, t: float) -> float:
            return v_end + 0.5 * (v_start - v_end) * (1 + math.cos(math.pi * t))

        for epoch in tqdm(range(self.epochs), desc="Epoch", dynamic_ncols=True):
            self.model.train()

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

            running = 0.0
            running_loss_main = 0.0
            running_w_main = 0.0
            running_snr = 0.0

            running_pres_accuracy = 0.0
            running_pres1_accuracy = 0.0
            running_pres_prob_mean = 0.0

            bar = tqdm(self.train_loader, desc="Batch", dynamic_ncols=True)

            for i, batch in enumerate(bar):
                batch = batch.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    if self.separated_modeling:
                        model_output = self.model(batch)
                        loss, loss_details = self._separated_loss_fn(batch, model_output)
                        running_loss_main += loss_details["abun_loss"]
                        running_pres_accuracy += loss_details.get("pres_accuracy", 0.0)
                        running_pres1_accuracy += loss_details.get("pres1_accuracy", 0.0)
                        running_pres_prob_mean += loss_details.get("pres_prob_mean", 0.0)
                    else:
                        # diffusion branch
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
                        else:  # 'eps'
                            eps_hat = self.model(xt, t).float()
                            v_pred = self._v_from_eps_x0(eps_hat, x0, alpha_t, sigma_t)

                        v_true = self._v_from_eps_x0(noise, x0, alpha_t, sigma_t).float()

                        loss_dict = self._v_pred_loss_fn(v_pred=v_pred, v_true=v_true, snr=snr.view(-1))
                        loss = loss_dict["total"]
                        running_loss_main += loss_dict["loss_main"].item()
                        running_w_main += loss_dict["w_main_mean"].item()
                        running_snr += loss_dict["snr_mean"].item()

                self.scaler.scale(loss).backward()

                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if i % 50 == 0:
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            pn = p.grad.data.norm(2)
                            total_norm += pn.item() ** 2
                    total_norm = total_norm ** 0.5
                    self.tb_writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
                    if total_norm > 10.0:
                        print(f"[WARN] 梯度范数过高: {total_norm:.2f} at step {global_step}")
                    progress_bar = (epoch * self.length_train_loader + i) / (self.epochs * self.length_train_loader)
                    self.tb_writer.add_scalar('Training/Progress', progress_bar, global_step)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

                self.scheduler.step()
                self._step_ema()

                running += loss.item()
                global_step += 1

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
                        w_main=f"{running_w_main/(i+1):.3f}",
                        snr=f"{running_snr/(i+1):.2f}",
                        lr=f"{self.opt.param_groups[0]['lr']:.2e}"
                    )

            # epoch averages (train)
            avg_loss = running / self.length_train_loader
            avg_loss_main = running_loss_main / self.length_train_loader

            if self.separated_modeling:
                avg_pres_accuracy = running_pres_accuracy / self.length_train_loader
                avg_pres1_accuracy = running_pres1_accuracy / self.length_train_loader
                avg_pres_prob_mean = running_pres_prob_mean / self.length_train_loader
                tqdm.write(f"[Epoch {epoch:03d}] train: mean_loss={avg_loss:.5f} abun={avg_loss_main:.5f}")
                tqdm.write(f"[Epoch {epoch:03d}] train: pres_acc={avg_pres_accuracy:.3f} pres1_acc={avg_pres1_accuracy:.3f} pres_prob={avg_pres_prob_mean:.3f}")

                self.tb_writer.add_scalar('Train/Loss_Total', avg_loss, epoch)
                self.tb_writer.add_scalar('Train/Loss_Abundance', avg_loss_main, epoch)
                self.tb_writer.add_scalar('Train/Presence_Accuracy', avg_pres_accuracy, epoch)
                self.tb_writer.add_scalar('Train/Pres1_Accuracy', avg_pres1_accuracy, epoch)
                self.tb_writer.add_scalar('Train/Presence_Prob_Mean', avg_pres_prob_mean, epoch)
            else:
                avg_w_main = running_w_main / max(1, self.length_train_loader)
                avg_snr = running_snr / max(1, self.length_train_loader)
                tqdm.write(f"[Epoch {epoch:03d}] train: mean_loss={avg_loss:.5f} loss_main={avg_loss_main:.5f} w_main={avg_w_main:.3f} snr={avg_snr:.2f}")
                self.tb_writer.add_scalar('Train/Loss_Total', avg_loss, epoch)
                self.tb_writer.add_scalar('Train/Loss_Main', avg_loss_main, epoch)
                self.tb_writer.add_scalar('Train/Weight_Mean', avg_w_main, epoch)
                self.tb_writer.add_scalar('Train/SNR_Mean', avg_snr, epoch)

            # -------- validation (use val_loader if provided) --------
            val_results = None
            if self.val_loader is not None:
                val_batches = int(self.cfg.get("val_num_batches", 20))  # 可在 cfg 里改
                val_results = self.evaluate_noise_matching_gate(
                    loader=self.val_loader, num_batches=val_batches
                )
                # log to TB
                self.tb_writer.add_scalar('Val/Spearman', 0.0 if math.isnan(val_results["spearman"]) else val_results["spearman"], epoch)
                self.tb_writer.add_scalar('Val/Pearson',  0.0 if math.isnan(val_results["correlation"]) else val_results["correlation"], epoch)
                if self.separated_modeling:
                    self.tb_writer.add_scalar('Val/Pres_Accuracy', 0.0 if math.isnan(val_results["pres_accuracy"]) else val_results["pres_accuracy"], epoch)
                    self.tb_writer.add_scalar('Val/Pres1_Accuracy', 0.0 if math.isnan(val_results.get("pres1_accuracy", float('nan'))) else val_results["pres1_accuracy"], epoch)
                    self.tb_writer.add_scalar('Val/Abun_MSE', 0.0 if math.isnan(val_results["abun_mse"]) else val_results["abun_mse"], epoch)
                else:
                    self.tb_writer.add_scalar('Val/V_MSE', 0.0 if math.isnan(val_results["v_mse"]) else val_results["v_mse"], epoch)

                tqdm.write(f"[Epoch {epoch:03d}] val: Spearman={val_results['spearman']:.4f}, Pearson={val_results['correlation']:.4f}")

            # -------- checkpointing --------
            if self.separated_modeling:
                # 优先用 val 指标保存 best
                if val_results is not None:
                    current_metrics = {
                        'abun_mse': val_results['abun_mse'],
                        'pres1_accuracy': val_results.get('pres1_accuracy', 0.0)
                    }
                else:
                    assert False, "val_results is None"
                self._save_ckpt(epoch, avg_loss, current_metrics)
            else:
                # 非分离：仍按 train loss 最优（保持你的原逻辑）
                self._save_ckpt(epoch, avg_loss, {})

        print("\n" + "="*60)
        print("训练结束，开始最终评估 (用 val_loader 如有) ...")
        _ = self.evaluate_training_stage()
        print("="*60 + "\n")
        self.close()

    # ----------------- checkpoints -------------------
    def _save_training_history(self, history: Dict[str, list]):
        history_path = self.ckpt_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[INFO] 训练历史已保存到: {history_path}")

    def _save_ckpt(self, epoch: int, loss: float, metrics: Dict[str, float] = None):
        """Always save latest; best rule:
           - separated_modeling: save best when pres1_acc >= threshold AND abun_mse improved (基于验证集优先)
           - else: best by train loss
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

        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(obj, latest_path)
        (self.ckpt_dir / "latest.json").write_text(json.dumps({"path": str(latest_path)}))
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        if getattr(self, "separated_modeling", False):
            pres1_acc = (metrics or {}).get('pres1_accuracy', 0.0)
            abun_mse  = (metrics or {}).get('abun_mse', float('inf'))
            if pres1_acc >= self.pres1_acc_threshold and abun_mse < self.best_abun_mse_at_pres1:
                self.best_abun_mse_at_pres1 = abun_mse
                best_sep_path = self.ckpt_dir / f"best_sep_{timestamp}_abun{abun_mse:.4f}_pres1{pres1_acc:.4f}.pt"
                torch.save(obj, best_sep_path)
                self.best_ckpt_path = best_sep_path
                print(f"[✓] 新 best(分离建模, 基于验证): {best_sep_path.name} (abun_mse={abun_mse:.4f}, pres1_acc={pres1_acc:.4f})")
                self.tb_writer.add_scalar('Model/Best_AbunMSE_at_Pres1', abun_mse, epoch)
                self.tb_writer.add_text('Model/Best_Sep_Checkpoint', str(best_sep_path), epoch)
            return

        # non-separated: best by train loss
        if loss < self.best_loss:
            if self.best_ckpt_path and self.best_ckpt_path.exists():
                self.best_ckpt_path.unlink(missing_ok=True)
                print(f"[INFO] 删除旧的best: {self.best_ckpt_path.name}")
            best_path = self.ckpt_dir / f"best_{timestamp}_loss{loss:.4f}.pt"
            torch.save(obj, best_path)
            self.best_ckpt_path = best_path
            self.best_loss = loss
            print(f"[✓] 新 best: {best_path.name} (loss={loss:.4f})")
            self.tb_writer.add_scalar('Model/Best_Loss', loss, epoch)
            self.tb_writer.add_text('Model/Best_Checkpoint', str(best_path), epoch)

    # ---------------- logging ------------------------
    def _log_experiment_config(self):
        config_text = f"""
            实验配置:
            - 分离建模: {self.separated_modeling}
            - 预测头: {self.prediction_type}
            - λ_abun: {self.lambda_abun}
            - λ_pres: {self.lambda_pres}
            - 训练轮数: {self.epochs}
            - 设备: {self.device}
            - pres1保存阈值: {self.pres1_acc_threshold}
            - 使用验证集: {self.val_loader is not None}
            """
        self.tb_writer.add_text('Experiment/Config', config_text, 0)
        print("[INFO] 实验配置已记录到TensorBoard")

    def close(self):
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
            print("[INFO] TensorBoard writer已关闭")

    # --------------- Spearman (eval only) ------------
    def _spearman_rho(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Spearman ρ，仅在 evaluate 时调用。返回标量 Tensor（batch 平均）。"""
        a2 = a.view(a.size(0), -1).float()
        b2 = b.view(b.size(0), -1).float()
        ar = a2.argsort(dim=1).argsort(dim=1).float()
        br = b2.argsort(dim=1).argsort(dim=1).float()
        ar = (ar - ar.mean(dim=1, keepdim=True)) / (ar.std(dim=1, unbiased=False, keepdim=True) + 1e-12)
        br = (br - br.mean(dim=1, keepdim=True)) / (br.std(dim=1, unbiased=False, keepdim=True) + 1e-12)
        rho = (ar * br).mean(dim=1)  # per-sample rho
        return rho.mean()

    # ----------------- evaluations -------------------
    def evaluate_noise_matching_gate(self, loader: Optional[torch.utils.data.DataLoader] = None, num_batches: int = 20) -> Dict[str, float]:
        """评估噪声/方向匹配 + Spearman。默认优先用 val_loader，其次 train_loader。"""
        self.model.eval()
        loader = loader or self.val_loader or self.train_loader

        total_noise_mse = 0.0; count_noise = 0
        total_v_mse = 0.0;     count_v = 0
        total_corr = 0.0;      count_corr = 0
        total_pres_accuracy = 0.0; count_pres = 0
        total_pres1_accuracy = 0.0; count_pres1 = 0
        total_abun_mse = 0.0;  count_abun = 0
        total_spearman = 0.0;  count_spear = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_batches: break
                batch = batch.to(self.device, non_blocking=True)

                if self.separated_modeling:
                    out = self.model(batch)
                    eps_hat = out["eps_hat"].float()
                    noise = batch.noise.squeeze(-1).float()

                    noise_mse = F.mse_loss(eps_hat, noise)
                    total_noise_mse += noise_mse.item(); count_noise += 1

                    corr = torch.corrcoef(torch.stack([eps_hat.flatten(), noise.flatten()]))[0, 1]
                    total_corr += float(corr.item()); count_corr += 1

                    rho = self._spearman_rho(eps_hat, noise)
                    total_spearman += float(rho.item()); count_spear += 1

                    pres_logit = out["pres_logit"]
                    pres_prob = torch.sigmoid(pres_logit)
                    pred = (pres_prob > 0.5).float()
                    true = batch.x0_pres.squeeze(-1).float()
                    pres_acc = (pred == true).float().mean()
                    total_pres_accuracy += pres_acc.item(); count_pres += 1

                    mask = true
                    pres1_acc = ((pred == true).float() * mask).sum() / mask.sum().clamp_min(1)
                    total_pres1_accuracy += pres1_acc.item(); count_pres1 += 1

                    if mask.sum() > 0:
                        abun_mse = ((eps_hat - noise)**2 * mask).sum() / mask.sum()
                        total_abun_mse += abun_mse.item(); count_abun += 1

                else:
                    B = batch.x0_abun.size(0) if hasattr(batch, 'x0_abun') else batch.noise.size(0)
                    t = torch.randint(0, len(self.betas), (B,), device=self.device)
                    alpha_t = self.alpha_t[t].view(-1, 1)
                    sigma_t = self.sigma_t[t].view(-1, 1)

                    x0 = batch.x0_abun if hasattr(batch, 'x0_abun') else batch.noise
                    noise = torch.randn_like(x0)
                    xt = alpha_t * x0 + sigma_t * noise

                    if self.prediction_type == "v":
                        v_pred = self.model(xt, t).float()
                    else:
                        eps_hat = self.model(xt, t).float()
                        v_pred = self._v_from_eps_x0(eps_hat, x0, alpha_t, sigma_t)

                    v_true = self._v_from_eps_x0(noise, x0, alpha_t, sigma_t).float()

                    v_mse = F.mse_loss(v_pred, v_true)
                    total_v_mse += v_mse.item(); count_v += 1

                    corr = torch.corrcoef(torch.stack([v_pred.flatten(), v_true.flatten()]))[0, 1]
                    total_corr += float(corr.item()); count_corr += 1

                    rho = self._spearman_rho(v_pred, v_true)
                    total_spearman += float(rho.item()); count_spear += 1

        avg_noise_mse   = (total_noise_mse / count_noise) if count_noise > 0 else float('nan')
        avg_v_mse       = (total_v_mse / count_v) if count_v > 0 else float('nan')
        avg_corr        = (total_corr / count_corr) if count_corr > 0 else float('nan')
        avg_pres_acc    = (total_pres_accuracy / count_pres) if count_pres > 0 else float('nan')
        avg_pres1_acc   = (total_pres1_accuracy / count_pres1) if count_pres1 > 0 else float('nan')
        avg_abun_mse    = (total_abun_mse / count_abun) if count_abun > 0 else float('nan')
        avg_spearman    = (total_spearman / count_spear) if count_spear > 0 else float('nan')

        # pass rules（分离：noise/pres1/abun；非分离：v_mse/pearson）
        if self.separated_modeling:
            noise_mse_passed = (avg_noise_mse <= 0.05) if not math.isnan(avg_noise_mse) else False
            pres1_passed     = (avg_pres1_acc >= 0.95) if not math.isnan(avg_pres1_acc) else False
            abun_mse_passed  = (avg_abun_mse <= 0.05) if not math.isnan(avg_abun_mse) else False
            overall_passed = (noise_mse_passed and pres1_passed and abun_mse_passed)
        else:
            v_mse_passed = (avg_v_mse <= 0.05) if not math.isnan(avg_v_mse) else False
            corr_passed  = (avg_corr >= 0.98) if not math.isnan(avg_corr) else False
            overall_passed = (v_mse_passed and corr_passed)

        return {
            "noise_mse": avg_noise_mse,
            "v_mse": avg_v_mse,
            "correlation": avg_corr,
            "spearman": avg_spearman,
            "pres_accuracy": avg_pres_acc,
            "pres1_accuracy": avg_pres1_acc,
            "abun_mse": avg_abun_mse,
            "overall_passed": overall_passed
        }

    def evaluate_training_stage(self) -> Dict[str, Any]:
        print("[INFO] 开始评估 (优先使用 val_loader) ...")

        if self.best_ckpt_path and self.best_ckpt_path.exists():
            print(f"[INFO] 加载最优模型: {self.best_ckpt_path.name}")
            checkpoint = torch.load(self.best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f"[INFO] 最优模型来自 epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")
        else:
            print("[WARN] 未找到最优模型，使用当前模型权重评估")

        results = self.evaluate_noise_matching_gate(loader=self.val_loader, num_batches=self.cfg.get("val_num_batches", 20))

        # log to TB
        self.tb_writer.add_scalar('Final/Noise_MSE', 0.0 if math.isnan(results["noise_mse"]) else results["noise_mse"], 0)
        self.tb_writer.add_scalar('Final/V_MSE',     0.0 if math.isnan(results["v_mse"])     else results["v_mse"], 0)
        self.tb_writer.add_scalar('Final/Pearson',   0.0 if math.isnan(results["correlation"]) else results["correlation"], 0)
        self.tb_writer.add_scalar('Final/Spearman',  0.0 if math.isnan(results["spearman"])    else results["spearman"], 0)
        if self.separated_modeling:
            self.tb_writer.add_scalar('Final/Pres_Acc',   0.0 if math.isnan(results["pres_accuracy"]) else results["pres_accuracy"], 0)
            self.tb_writer.add_scalar('Final/Pres1_Acc',  0.0 if math.isnan(results["pres1_accuracy"]) else results["pres1_accuracy"], 0)
            self.tb_writer.add_scalar('Final/Abun_MSE',   0.0 if math.isnan(results["abun_mse"]) else results["abun_mse"], 0)

        print(f"[评估] overall_passed: {'✓' if results['overall_passed'] else '✗'}")
        if self.separated_modeling:
            print(f"  - Noise MSE: {results['noise_mse']:.4f} (≤0.05)")
            print(f"  - Pres1 Acc: {results['pres1_accuracy']:.4f} (≥0.95)")
            print(f"  - Abun MSE:  {results['abun_mse']:.4f} (≤0.05)")
            print(f"  - Spearman:  {results['spearman']:.4f}  (报告用)")
        else:
            print(f"  - V-MSE:     {results['v_mse']:.4f} (≤0.05)")
            print(f"  - Pearson:   {results['correlation']:.4f} (≥0.98)")
            print(f"  - Spearman:  {results['spearman']:.4f}  (报告用)")

        return {"noise_results": results, "stage1_passed": results["overall_passed"]}
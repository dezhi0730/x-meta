import os
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from gutclip.engine.trainer_diffusion import TrainerDiffusion, RetrievalIndex
from gutclip.data import DiffusionDataModule
from gutclip.utils.seed import set_seed
from gutclip.evaluate.eval_gateA import plot_gateA_curves, save_json
from gutclip.evaluate.eval_gateA import log_tail_worst


def setup_ddp() -> int:
    """初始化分布式，返回当前 rank。"""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0
    return rank


def build_retrieval(cfg, rank: int) -> RetrievalIndex:
    """构建带温度/阈值控制的检索索引。"""
    ret = RetrievalIndex(
        index_file=cfg.retrieval.index,
        y_file=cfg.retrieval.y,
        ids_file=getattr(cfg.retrieval, "ids", None),
        gpu=rank,
        k=cfg.retrieval.k,
        # 新增可选参数（若类签名不含这些参数，请在类中添加形参并存到 self）
        metric=getattr(cfg.retrieval, "metric", "l2"),
        softmax_temp=float(getattr(cfg.retrieval, "softmax_temp", 0.1)),
        sim_thresh=getattr(cfg.retrieval, "sim_thresh", None),
    )
    if rank == 0:
        print(
            "[RET] metric={}, temp={}, sim_thresh={}, k={}".format(
                getattr(cfg.retrieval, "metric", "l2"),
                getattr(cfg.retrieval, "softmax_temp", 0.1),
                getattr(cfg.retrieval, "sim_thresh", None),
                cfg.retrieval.k,
            )
        )
    return ret


def log_to_tb(tb: SummaryWriter, ep: int, train_loss: Dict[str, float], val_stats: Dict[str, float], verdict: Dict[str, Any]):
    tb.add_scalar("train/total", train_loss["total"], ep)
    tb.add_scalar("train/v_mse", train_loss["v_mse"], ep)
    tb.add_scalar("train/cal", train_loss["cal"], ep)
    tb.add_scalar("train/rank", train_loss["rank"], ep)
    if "v_mse" in val_stats:
        tb.add_scalar("val/v_mse", val_stats["v_mse"], ep)
    if "y0_mix_mse" in val_stats:
        tb.add_scalar("val/y0_mix_mse", val_stats["y0_mix_mse"], ep)
    if "y0_mix_spr" in val_stats:
        tb.add_scalar("val/y0_mix_spr", val_stats["y0_mix_spr"], ep)

    tb.add_scalar("gateA/pass", int(verdict["pass"]), ep)
    tb.add_scalar("gateA/nmse_main_mean", verdict["details"]["nmse_main_mean"], ep)
    tb.add_scalar("gateA/cos_main_mean", verdict["details"]["cos_main_mean"], ep)


def log_to_wandb(cfg, ep: int, train_loss: Dict[str, float], val_stats: Dict[str, float], verdict: Dict[str, Any]):
    log_dict = {
        "epoch": ep,
        "train/total": train_loss["total"],
        "train/v_mse": train_loss["v_mse"],
        "train/cal": train_loss["cal"],
        "train/rank": train_loss["rank"],
        "gateA/pass": int(verdict["pass"]),
        "gateA/nmse_main_mean": verdict["details"]["nmse_main_mean"],
        "gateA/cos_main_mean": verdict["details"]["cos_main_mean"],
        "gateA/nmse_tail_max": verdict["details"]["nmse_tail_max"],
        "gateA/cos_tail_min": verdict["details"]["cos_tail_min"],
    }
    if "v_mse" in val_stats:
        log_dict["val/v_mse"] = val_stats["v_mse"]
    if "y0_mix_mse" in val_stats:
        log_dict["val/y0_mix_mse"] = val_stats["y0_mix_mse"]
    if "y0_mix_spr" in val_stats:
        log_dict["val/y0_mix_spr"] = val_stats["y0_mix_spr"]

    if cfg.wandb.mode != "disabled":
        wandb.log(log_dict)


def as_val_stats(v) -> Dict[str, float]:
    """统一把 evaluate 的返回转成字典。"""
    if isinstance(v, dict):
        return v
    return {"v_mse": float(v)}


@hydra.main(config_path="../configs", config_name="train_gen_model", version_base="1.3")
def main(cfg):
    # DEBUG 打印配置
    if os.environ.get("DEBUG_CFG") == "1":
        print(OmegaConf.to_yaml(cfg))

    rank = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)

    # 数据
    dm = DiffusionDataModule(cfg)
    dm.setup(stage="fit")
    z_dim = dm.train_ds.z.shape[1]
    y_dim = dm.train_ds.y.shape[1]
    cfg.model.z_dna_dim = getattr(cfg.model, "z_dna_dim", z_dim)
    cfg.model.y_dim = getattr(cfg.model, "y_dim", y_dim)
    cfg.model.proj_dim = getattr(cfg.model, "proj_dim", min(256, y_dim))
    cfg.model.cond_dim = cfg.model.z_dna_dim + cfg.model.proj_dim

    # 记录工具
    tb = SummaryWriter(f"runs/{cfg.run_name}") if rank == 0 else None
    if rank == 0 and cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.run_name,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # 检索
    ret = build_retrieval(cfg, rank)

    # 训练/验证集 ID 重叠检查
    if rank == 0:
        train_ids = set(np.load(cfg.retrieval.index.replace(".faiss", ".ids.npy")).tolist())
        val_ids = set(dm.val_ds.sample_ids)
        intersection = len(train_ids & val_ids)
        print(f"[CHECK] #train_ids: {len(train_ids)}, #val_ids: {len(val_ids)}, intersection: {intersection}")
        if intersection > 0:
            print(f"[WARN] 训练集与验证集有 {intersection} 个重叠样本！")
        else:
            print("[✓] 训练集与验证集无重叠")

    # Trainer
    trainer = TrainerDiffusion(
        cfg,
        dataloader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        retrieval=ret,
        device=device,
    )

    # 早停 & 保存
    best_vmse = float("inf")
    patience = 0
    best_gateA_score = float("inf")
    best_gateA_epoch = -1
    best_tail = float("inf")  # nmse_tail_max 最优
    ckpt_dir = Path("checkpoints/diffusion")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # —— Booster 自动切换（可选）——
    booster_off_epoch = int(os.environ.get("BOOSTER_OFF_EPOCH", "2"))

    for ep in range(cfg.train.epochs):
        # Booster: 训练前在第 ep==booster_off_epoch 时下调权重/概率
        if ep == booster_off_epoch:
            # 只在 rank0 打印，但对所有进程都要改
            if rank == 0:
                print("[Booster] switch OFF -> set p_high_snr=0.65, lambda_high=0.30")
            # 运行时同步修改
            cfg.train.p_high_snr = 0.65
            cfg.train.lambda_high = 0.30
            # Trainer 内部对象也要更新
            trainer.p_high_snr = 0.65
            trainer.criterion.lambda_high = 0.30

        if dist.is_initialized():
            # 分布式采样器设 epoch
            dl = dm.train_dataloader()
            if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(ep)

        # 训练 / 验证
        train_loss = trainer.train_one_epoch(ep, tb)
        val_stats = as_val_stats(trainer.evaluate(tb, ep))

        if rank == 0:
            # 保存最新
            latest_ckpt_path = trainer.save_ckpt("latest", ep, {"v_mse": val_stats["v_mse"]}, remove_old=True)

            # 更新最优 v_mse
            if val_stats["v_mse"] < best_vmse - cfg.train.min_delta:
                best_ckpt_path = trainer.save_ckpt("best", ep, {"v_mse": val_stats["v_mse"]}, remove_old=True)
                best_vmse = val_stats["v_mse"]
                patience = 0
                print(f"[✓] new best v_mse={best_vmse:.6f} -> {best_ckpt_path.name}")
            else:
                patience += 1
                if patience >= cfg.train.patience:
                    print(f"[EarlyStop] no improv >{cfg.train.patience} epochs")
                    break

            # GateA 快评
            verdict, buckets, raw, suggestions = trainer.eval_gateA_once(
                max_batches=50,  # 小样本
                num_bins=12,
                use_autocast=True,
            )
            print(f"[GateA-Fast][Ep {ep}] PASS={verdict['pass']} details={verdict['details']}")
            save_json(verdict, f"{ckpt_dir}/gateA_fast_epoch{ep}.json")
            save_json(buckets, f"{ckpt_dir}/gateA_fast_epoch{ep}_buckets.json")
            save_json({"verdict": verdict, "suggestions": suggestions}, f"{ckpt_dir}/gateA_fast_epoch{ep}_summary.json")
            log_tail_worst(buckets)

            # 自适应 lambda_high 调整
            if ep == 2 and verdict["details"]["nmse_tail_max"] > 0.32:
                trainer.criterion.lambda_high = 0.26
                print("[Tune] raise lambda_high to 0.26 due to high tail nmse")

            # GateB 残差校准与采样稳定性评估（每10个epoch）
            if (ep + 1) % 10 == 0:
                try:
                    verdict_gateB, raw_metrics_gateB, suggestions_gateB = trainer.eval_gateB_once(
                        max_batches=20,
                        num_ddim_steps=50,
                        use_autocast=True,
                    )
                    print(f"[GateB][Ep {ep}] PASS={verdict_gateB['passed']} scores={verdict_gateB['scores']}")
                    save_json(verdict_gateB, f"{ckpt_dir}/gateB_epoch{ep}.json")
                    save_json({"verdict": verdict_gateB, "suggestions": suggestions_gateB}, f"{ckpt_dir}/gateB_epoch{ep}_summary.json")
                    
                    # 记录到wandb
                    if cfg.wandb.mode != "disabled":
                        wandb.log({
                            "gateB/pass": int(verdict_gateB["passed"]),
                            "gateB/overall_score": verdict_gateB["scores"]["overall"],
                            "gateB/residual_calibration": verdict_gateB["scores"]["residual_calibration"],
                            "gateB/trajectory_stability": verdict_gateB["scores"]["trajectory_stability"],
                        })
                    
                    # 如果关卡B通过，可以考虑进入下一阶段
                    if verdict_gateB["passed"]:
                        print(f"[GateB] ✅ 通过关卡B！模型在数值上可稳定采样，可考虑进入关卡C（条件生成测试）")
                    
                except Exception as e:
                    print(f"[GateB] 评估失败: {e}")
            
            # 记录日志
            log_to_wandb(cfg, ep, train_loss, val_stats, verdict)
            if tb is not None:
                log_to_tb(tb, ep, train_loss, val_stats, verdict)

            # GateA 早停指标（主区 mean NMSE）
            score = verdict["details"]["nmse_main_mean"]
            if np.isfinite(score) and score < best_gateA_score:
                best_gateA_score = score
                best_gateA_epoch = ep

            # 维护“尾部最优”模型（nmse_tail_max 越小越好）
            tail_score = verdict["details"]["nmse_tail_max"]
            if np.isfinite(tail_score) and tail_score < best_tail:
                best_tail = tail_score
                trainer.save_ckpt("best_tail", ep, {"v_mse": val_stats["v_mse"]}, remove_old=True)
                print(f"[✓] new best_tail nmse_tail_max={tail_score:.6f} at epoch {ep}")

            # 每 5 个 epoch 全量评估 + 曲线
            if (ep + 1) % 5 == 0:
                verdict_full, buckets_full, raw_full, suggestions_full = trainer.eval_gateA_once(
                    max_batches=None,  # 全量
                    num_bins=12,
                    use_autocast=True,
                )
                print(f"[GateA-Full][Ep {ep}] PASS={verdict_full['pass']} details={verdict_full['details']}")
                out_prefix = f"{ckpt_dir}/gateA_full_epoch{ep}"
                save_json(verdict_full, f"{out_prefix}.json")
                save_json(buckets_full, f"{out_prefix}_buckets.json")
                save_json({"verdict": verdict_full, "suggestions": suggestions_full}, f"{out_prefix}_summary.json")
                plot_gateA_curves(buckets_full, out_png=f"{out_prefix}.png")
                log_tail_worst(buckets_full)

                # 连续通过且改善不足提醒
                if verdict_full["pass"] and best_gateA_score < float("inf"):
                    improvement = (best_gateA_score - score) / best_gateA_score
                    if improvement < 0.05:
                        print(f"[GateA] 连续通过且改善<5% ({improvement:.1%})，可考虑进入下一阶段。")

    # 结束清理
    if rank == 0:
        if tb is not None:
            tb.close()
        if cfg.wandb.mode != "disabled":
            wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
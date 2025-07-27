import torch, torch.nn.functional as F
from .utils import add_noise

@torch.no_grad()
def generate(model, scheduler, cond_vec, steps=50, shape=(1,)):
    device = next(model.parameters()).device
    y = torch.randn(shape, device=device)        # y_T
    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        # 1) 预测噪声 ε̂
        noise_pred = model(y, t, cond_vec)

        # 2) 由调度器一步反推 y_{t-1}
        y = scheduler.step(noise_pred, t, y)["prev_sample"]
    return y
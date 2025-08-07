import math, torch
from typing import Tuple

# --------------------------------------------------
# 1. Beta schedule helpers
# --------------------------------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Return linearly spaced betas (torch.Tensor, shape [T])."""
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Better cosine schedule from Nichol & Dhariwal 2021."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    x = (steps / T) + s
    alphas_bar = torch.cos((x / (1 + s)) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-8, 0.999)


# --------------------------------------------------
# 2. Helper tensor extractor (borrowed from DDPM impl)
# --------------------------------------------------

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]):
    """Extract per-sample coefficients from a 1-D tensor a at indices t."""
    out = a.gather(0, t)
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


# --------------------------------------------------
# 3. Forward process (q_sample)
# --------------------------------------------------

def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor):
    """Compute x_t = √{ᾱ} x0 + √{1-ᾱ} ε . sqrt_ab & sqrt_1mab are 1-D length-T tensors."""
    return extract(sqrt_ab, t, x0.shape) * x0 + extract(sqrt_1mab, t, x0.shape) * noise


# --------------------------------------------------
# 4. Reverse helpers (predict x0 from eps)
# --------------------------------------------------

def predict_x0_from_eps(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor):
    """x0 ≈ (x_t − √{1-ᾱ} ε) / √{ᾱ}"""
    return (x_t - extract(sqrt_1mab, t, x_t.shape) * eps) / extract(sqrt_ab, t, x_t.shape)


# --------------------------------------------------
# 5. DDIM deterministic step (eta=0 for deterministic)
# --------------------------------------------------

def ddim_step(x_t: torch.Tensor, eps: torch.Tensor, t: int, eta: float,
              alphas_bar: torch.Tensor, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor):
    """Single DDIM reverse step (matching equations in Song et al. 2020)."""
    if t == 0:
        return x_t  # no step

    t_prev = t - 1
    alpha_bar_t     = alphas_bar[t]
    alpha_bar_prev  = alphas_bar[t_prev]

    # deterministic DDIM
    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
    dir_xt  = torch.sqrt(1 - alpha_bar_prev) * eps
    x_prev  = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

    if eta > 0.0:
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        noise = torch.randn_like(x_t)
        x_prev = x_prev + sigma * noise
    return x_prev


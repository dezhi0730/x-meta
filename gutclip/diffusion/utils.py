import torch

def add_noise(x, noise, t, betas):
    """DDPM forward q(x_t | x_0)"""
    # ᾱ_t = \prod_{i=1}^t (1-β_i)
    alphas_cumprod = (1 - betas).cumprod(0)          # (T,)

    # √{ᾱ_t}
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()      # (T,)

    # √{1-ᾱ_t}
    sqrt_one_minus_alphas = (1 - alphas_cumprod).sqrt()  # (T,)

    # 兼容任意维度 (B, *shape)
    a = sqrt_alphas_cumprod[t]
    b = sqrt_one_minus_alphas[t]

    # 扩展到与 x 形状一致，排除 batch 维
    while a.dim() < x.dim():
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)

    return a * x + b * noise        # same shape as x
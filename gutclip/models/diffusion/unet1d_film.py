import torch, torch.nn as nn
from diffusers import UNet1DModel

class FiLMUNet1D(UNet1DModel):
    """
    输入  y_t: (B, N)            # N = y_dim (作为长度维)
    条件  cond_vec: (B, cond_dim)
    输出  eps_pred: (B, N)
    做法：在长度维做 FiLM（对每个位置生成 γ/β），然后交给 UNet1D 做时序卷积。
    """
    def __init__(self,
                 y_dim: int,           # 这里表示长度 N
                 cond_dim: int,
                 base_channels: int = 128,
                 layers_per_block: int = 2,
                 norm_num_groups: int = 8,
                 **kw):
        super().__init__(
            in_channels=1,            # 关键：单通道
            out_channels=1,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            extra_in_channels=16,     # fourier 会拼接 16 个额外通道
            time_embedding_type=kw.get("time_embedding_type", "fourier"),
            use_timestep_embedding=kw.get("use_timestep_embedding", False),
        )
        self.y_len = y_dim
        # 在长度维做 FiLM：生成 2*N 个参数
        self.film_len = nn.Linear(cond_dim, 2 * self.y_len)

    def _apply_film_len(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, N)
        gamma, beta = self.film_len(cond).chunk(2, dim=-1)   # (B, N)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, y_t: torch.Tensor, timesteps: torch.Tensor, cond_vec: torch.Tensor):
        # y_t: (B, N) -> (B, 1, N)
        x = y_t.unsqueeze(1)
        x = self._apply_film_len(x, cond_vec)

        # fourier 常用 float timestep
        t = timesteps.float()
        out = super().forward(x, t)   # 签名 (sample, timestep)

        y = out.sample if hasattr(out, "sample") else out   # (B, 1, N)
        return y.squeeze(1)          # -> (B, N)
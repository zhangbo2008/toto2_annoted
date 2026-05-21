import torch #      HF_ENDPOINT=https://hf-mirror.com   python 2.py
from toto2 import Toto2Model


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SIZE = "4m"  # 4m | 22m | 313m | 1B | 2.5B
CHECKPOINT = f"Datadog/Toto-2.0-{SIZE}"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Toto2Model.from_pretrained(CHECKPOINT, map_location=device)
model = model.to(device).eval()

print(f"Loaded {CHECKPOINT}: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Patch size: {model.config.patch_size}")





# Synthetic univariate series: trend + seasonality + noise
context_length = 512
t = torch.arange(context_length, dtype=torch.float32)
series = 100 + 0.05 * t + 10 * torch.sin(2 * torch.pi * t / 24) + torch.randn(context_length) # 长度512的序列，包含趋势、季节性和噪声

# Shape: (batch=1, n_var=1, time)
target = series.unsqueeze(0).unsqueeze(0).to(device)
target_mask = torch.ones_like(target, dtype=torch.bool)
series_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

horizon = 96  # 我们模型能看到的范围, 前96个数. # 表示输出的大小.
quantiles = model.forecast(
    {"target": target, "target_mask": target_mask, "series_ids": series_ids},
    horizon=horizon,
)

print(f"Output shape: {quantiles.shape}")  # (9, 1, 1, 96)
print(f"Quantile levels: {model.output_head.knots}")






import matplotlib.pyplot as plt

median = quantiles[4, 0, 0].cpu()  # 0.5 quantile
q10 = quantiles[0, 0, 0].cpu()     # 0.1 quantile
q90 = quantiles[8, 0, 0].cpu()     # 0.9 quantile
print(median.shape)
fig, ax = plt.subplots(figsize=(12, 4))
ctx = series[-96:].cpu()
ax.plot(range(96), ctx, label="Context", color="black")# 0到 512-96的真实数据
ax.plot(range(96, 96 + horizon), median, label="Median forecast", color="tab:blue") # 中位数预测
# ax.fill_between(range(96, 96 + horizon), q10, q90, alpha=0.2, color="tab:blue", label="80% interval")
ax.legend()
ax.set_title("Toto 2.0 Forecast")
plt.tight_layout()
plt.savefig("forecast.png")







n_var = 3
target_mv = torch.randn(1, n_var, 512, device=device)
mask_mv = torch.ones(1, n_var, 512, dtype=torch.bool, device=device)
ids_mv = torch.zeros(1, n_var, dtype=torch.long, device=device)

quantiles_mv = model.forecast(
    {"target": target_mv, "target_mask": mask_mv, "series_ids": ids_mv},
    horizon=48,
)
print(f"Multivariate output: {quantiles_mv.shape}")  # (9, 1, 3, 48)
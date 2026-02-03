import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog


# =========================
# 0) 选择CSV文件
# =========================
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="选择你的CSV文件",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not file_path:
    raise RuntimeError("你没有选择文件，程序结束。")

df = pd.read_csv(file_path)
print("✅ 文件读取成功：", file_path)
print("列名：", list(df.columns))
print(df.head())


# =========================
# 1) 自动匹配 X/Y 两列
# =========================
def find_col(candidates, cols):
    for key in candidates:
        for c in cols:
            if key in c.lower():
                return c
    return None

x_col = find_col(["data usage", "mb/day", "data"], df.columns)
y_col = find_col(["battery drain", "mah/day", "battery"], df.columns)

if x_col is None or y_col is None:
    raise ValueError(
        "没自动匹配到列名。\n"
        f"当前列名：{list(df.columns)}\n"
        "请你手动指定 x_col / y_col。"
    )

print(f"\n✅ 自动匹配列：X={x_col}  Y={y_col}")


# =========================
# 2) 清洗数据：转数值 + 去NaN
# =========================
tmp = df[[x_col, y_col]].copy()
tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
tmp = tmp.dropna()

x = tmp[x_col].to_numpy(dtype=float)
y = tmp[y_col].to_numpy(dtype=float)

print(f"\n✅ 有效样本数：{len(tmp)}")


# =========================
# 3) 基础：Pearson/Spearman + 线性拟合（原始点）
# =========================
coef = np.polyfit(x, y, 1)
r_p, p_p = pearsonr(x, y)
r_s, p_s = spearmanr(x, y)

print(f"\n📌 Pearson r = {r_p:.4f}")
print(f"📌 p-value   = {p_p:.3e}")
print(f"📌 Spearman rho = {r_s:.4f}")
print(f"📌 Spearman p   = {p_s:.3e}")
print(f"📌 线性拟合(原始点): y = {coef[0]:.4f} * x + {coef[1]:.2f}")

# 原始散点 + 线性拟合线
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.15, label="Raw data")
x_fit = np.linspace(x.min(), x.max(), 300)
plt.plot(x_fit, coef[0] * x_fit + coef[1], linewidth=2, label="Linear fit (raw)")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("Raw scatter + linear fit")
plt.grid(True)
plt.legend()
plt.show()


# =========================
# 4) 分箱中位数（500MB）
# =========================
bin_width = 500
bins = np.arange(0, x.max() + bin_width, bin_width)
bin_id = np.digitize(x, bins, right=False)  # bin index: 1..len(bins)

rows = []
for i in range(1, len(bins)):  # bin: [bins[i-1], bins[i])
    mask = bin_id == i
    if mask.sum() == 0:
        continue
    rows.append({
        "x_median": float(np.median(x[mask])),
        "y_median": float(np.median(y[mask])),
        "count": int(mask.sum()),
        "bin_left": float(bins[i-1]),
        "bin_right": float(bins[i])
    })

binned = pd.DataFrame(rows)
print("\n=== 分箱中位数统计（每段）===")
print(binned[["bin_left", "bin_right", "x_median", "y_median", "count"]].to_string(index=False))


# =========================
# 5) 去掉最后一组（最高流量 bin）
# =========================
if len(binned) < 3:
    raise RuntimeError("分箱后的组数太少（<3），无法做多模型比较。")

binned_trim = binned.iloc[:-1].copy()   # 按你要求：去掉最后一组
x_med = binned_trim["x_median"].to_numpy(dtype=float)
y_med = binned_trim["y_median"].to_numpy(dtype=float)

print("\n✅ 已去掉最后一组（最高流量bin）后，用于拟合的中位数点：")
print(pd.DataFrame({"x_median": x_med, "y_median": y_med, "count": binned_trim["count"].to_numpy()}).to_string(index=False))


# =========================
# 6) 定义模型（递增拟合候选）
# =========================
def linear(x, a, b):
    return a * x + b

def log_func(x, a, b):
    return a * np.log(x) + b

def power(x, a, b):
    return a * (x ** b)

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# 饱和模型（更贴近“高区间趋于平”）： y = L - A * exp(-k x)
def saturating_exp(x, L, A, k):
    return L - A * np.exp(-k * x)


models = {
    "Linear":        (linear, 2),
    "Log":           (log_func, 2),
    "Power":         (power, 2),
    "Quadratic":     (quadratic, 3),
    "SaturatingExp": (saturating_exp, 3),
}


# =========================
# 7) 指标：RMSE/MAE/R2/AIC/BIC
# =========================
def calc_metrics(y_true, y_pred, k_params):
    n = len(y_true)
    resid = y_true - y_pred
    rss = float(np.sum(resid**2))
    rmse = float(np.sqrt(rss / n))
    mae = float(np.mean(np.abs(resid)))

    tss = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1 - rss / tss) if tss > 0 else float("nan")

    # AIC/BIC 基于 RSS 的高斯误差形式（用于模型对比）
    if rss <= 0:
        aic = float("-inf")
        bic = float("-inf")
    else:
        aic = float(n * np.log(rss / n) + 2 * k_params)
        bic = float(n * np.log(rss / n) + k_params * np.log(n))

    return rss, rmse, mae, r2, aic, bic


# =========================
# 8) 拟合 + 输出评估表
# =========================
rows = []

for name, (fn, k) in models.items():
    # 约束：log/power 需要 x>0
    if name in ["Log", "Power"] and np.any(x_med <= 0):
        rows.append([name, None, None, None, None, None, None, None, "skip: x<=0"])
        continue

    try:
        # 初值（让拟合更稳）
        p0 = None
        if name == "Power":
            p0 = (1.0, 1.0)
        elif name == "SaturatingExp":
            # L 大概是 y 的上界，A 取 L - y_min，k 给个小正数
            L0 = float(np.max(y_med))
            A0 = float(L0 - np.min(y_med))
            k0 = 0.001
            p0 = (L0, A0, k0)

        params, _ = curve_fit(fn, x_med, y_med, p0=p0, maxfev=20000)
        y_hat = fn(x_med, *params)

        rss, rmse, mae, r2, aic, bic = calc_metrics(y_med, y_hat, k)

        rows.append([name, params, rss, rmse, mae, r2, aic, bic, ""])
    except Exception as e:
        rows.append([name, None, None, None, None, None, None, None, f"fail: {e}"])

result = pd.DataFrame(rows, columns=[
    "model", "params", "RSS", "RMSE", "MAE", "R2", "AIC", "BIC", "note"
]).sort_values(by="AIC", ascending=True, na_position="last")

print("\n=== 多模型对比分箱中位数拟合（已去掉最后一组）===")
print(result.to_string(index=False))


# =========================
# 9) 画图：中位数点 + 各模型拟合曲线
# =========================
plt.figure(figsize=(9, 7))

# 原始散点（淡）
plt.scatter(x, y, alpha=0.08, label="Raw data")

# 中位数点（去掉最后一组）
plt.scatter(x_med, y_med, s=60, label=f"Binned medians (trimmed, width={bin_width})")

x_curve = np.linspace(x_med.min(), x_med.max(), 400)

for _, row in result.iterrows():
    name = row["model"]
    params = row["params"]
    note = row["note"]
    if params is None or (isinstance(note, str) and note.startswith("fail")):
        continue
    fn, _ = models[name]
    y_curve = fn(x_curve, *params)
    plt.plot(x_curve, y_curve, linewidth=2, label=f"{name}")

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("Model fits on binned medians (last bin removed)")
plt.grid(True)
plt.legend()
plt.show()

# launcher.py
import os
import sys
import subprocess

# -----------------------------
# 1) Paths
# -----------------------------
gold_dataset = "data/gold/btc_fng_combined.csv"
app_file = "app.py"

# -----------------------------
# 2) Check dataset exists
# -----------------------------
if not os.path.exists(gold_dataset):
    print(f"[ERROR] Dataset not found: {gold_dataset}")
    sys.exit(1)

# -----------------------------
# 3) Create app.py if it doesn't exist
# -----------------------------
if not os.path.exists(app_file):
    print("[INFO] Creating app.py...")
    app_code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene

# Load dataset
gold_path = "data/gold/btc_fng_combined.csv"
merged = pd.read_csv(gold_path, parse_dates=["date"])
if "btc_return_pct" not in merged.columns:
    merged["btc_return_pct"] = merged["btc_close"].pct_change()
merged = merged.dropna(subset=["btc_return_pct"])
merged["fear"] = merged["fng_value"] < 47

fear_returns = merged.loc[merged["fear"], "btc_return_pct"]
non_fear_returns = merged.loc[~merged["fear"], "btc_return_pct"]

st.title("BTC Gold Dataset Analysis")
st.subheader("Gold Dataset Preview")
st.dataframe(merged.head(10))

# -----------------------------
# KDE Plots
# -----------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(10,12))

sns.kdeplot(fear_returns, fill=True, color="red", alpha=0.4, ax=axes[0])
axes[0].axvline(0, color='black', linestyle='--')
axes[0].set_title("Fear Days")

sns.kdeplot(non_fear_returns, fill=True, color="blue", alpha=0.4, ax=axes[1])
axes[1].axvline(0, color='black', linestyle='--')
axes[1].set_title("Non-Fear Days")

df_plot = pd.concat([pd.DataFrame({"return": fear_returns, "group":"Fear"}),
                     pd.DataFrame({"return": non_fear_returns, "group":"Non-Fear"})])
sns.kdeplot(data=df_plot, x="return", hue="group", fill=True, alpha=0.4, common_norm=False, ax=axes[2])
axes[2].axvline(0, color='black', linestyle='--')
axes[2].set_title("Fear vs Non-Fear")
plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# Levene's Test
# -----------------------------
st.subheader("Levene's Test: Fear vs Non-Fear")
stat, p = levene(fear_returns, non_fear_returns)
st.write(f"Levene statistic: {stat:.4f}")
st.write(f"p-value: {p:.4f}")
st.write("Variances are significantly different" if p<0.05 else "No significant difference")

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Analysis")
merged["is_holiday"] = merged["date"].dt.dayofweek.isin([5,6])
merged["is_weekend"] = merged["date"].dt.dayofweek >= 5
corr_columns = ["btc_return_pct","fng_value","is_holiday","is_weekend"]
corr_data = merged[corr_columns].copy()
corr_data["is_holiday"] = corr_data["is_holiday"].astype(int)
corr_data["is_weekend"] = corr_data["is_weekend"].astype(int)

corr_matrix = corr_data.corr()
st.dataframe(corr_matrix)

fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)
"""
    with open(app_file, "w") as f:
        f.write(app_code)
    print("[INFO] app.py created successfully.")

# -----------------------------
# 4) Launch Streamlit
# -----------------------------
print("[INFO] Launching Streamlit app...")
subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_file])
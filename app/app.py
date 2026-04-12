import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import levene, chi2_contingency, spearmanr, f_oneway, bartlett
from scipy import stats
# Note: Custom t-test implementation is included to avoid scipy dependency issues in Streamlit Cloud.
st.set_page_config(page_title="BTC Statistical Analysis Dashboard", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:1.5rem; }
    .section-header { font-size:1.5rem; font-weight:bold; color:#2c3e50; margin-top:1.5rem;
        margin-bottom:1rem; border-bottom:2px solid #3498db; padding-bottom:0.4rem; }
    .metric-card { background:#f8f9fa; padding:1rem; border-radius:0.5rem;
        border-left:4px solid #3498db; margin:0.5rem 0; }
    .test-result { background:#e8f4f8; padding:1rem; border-radius:0.5rem; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
#Coerce numeric values and hadle duplicate and missing data.
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df["btc_return_pct"]  = pd.to_numeric(df["btc_return_pct"],  errors="coerce")
    df["btc_volume"]      = pd.to_numeric(df["btc_volume"],       errors="coerce")
    df["days_to_holiday"] = pd.to_numeric(df["days_to_holiday"],  errors="coerce")
    df["fng_value"]       = pd.to_numeric(df["fng_value"],        errors="coerce")
    df["fear"]            = df["fear"].astype(str).str.lower().map({"true": True, "false": False})
    df["gain"]            = df["btc_return_pct"] > 0
    df = df.dropna(subset=["btc_return_pct", "fear"])
    return df


# ── T-test helpers (custom implementation) ────────────────────────────────────
#Function to compute a t-distribution PDF and CDF numerically, then use it to calculate two-sided p-values for t-tests.
def t_pdf(x, dfree):
    log_coeff = (
        math.lgamma((dfree + 1) / 2)
        - 0.5 * (math.log(dfree) + math.log(math.pi))
        - math.lgamma(dfree / 2)
    )
    log_val = log_coeff - ((dfree + 1) / 2) * math.log(1 + (x * x) / dfree)
    return math.exp(log_val)

def t_cdf_numeric(x, dfree):
    if x == 0:
        return 0.5
    sign = 1 if x > 0 else -1
    x = abs(x)
    grid_size = max(20000, int(x * 4000))
    grid = np.linspace(0, x, grid_size + 1)
    density = np.array([t_pdf(float(v), dfree) for v in grid])
    area = np.trapezoid(density, grid)
    cdf_pos = min(max(0.5 + area, 0.0), 1.0)
    return cdf_pos if sign > 0 else 1 - cdf_pos

def two_sided_p(t_stat, dfree):
    cdf = t_cdf_numeric(t_stat, dfree)
    return 2 * min(cdf, 1 - cdf)

def one_sample_t(sample, mu=0.0):
    n = len(sample)
    mean, std = sample.mean(), sample.std(ddof=1)
    se = std / math.sqrt(n)
    t_stat = (mean - mu) / se
    return {"n": n, "mean": mean, "std": std, "t_stat": t_stat, "df": n - 1,
            "p_value": two_sided_p(t_stat, n - 1)}

def welch_t(a, b):
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    std_a, std_b   = a.std(ddof=1), b.std(ddof=1)
    se2 = (std_a**2 / n_a) + (std_b**2 / n_b)
    t_stat = (mean_a - mean_b) / math.sqrt(se2)
    dfree = (se2**2) / (
        ((std_a**2 / n_a)**2 / (n_a - 1)) + ((std_b**2 / n_b)**2 / (n_b - 1))
    )
    return {"n_a": n_a, "n_b": n_b, "mean_a": mean_a, "mean_b": mean_b,
            "diff": mean_a - mean_b, "t_stat": t_stat, "df": dfree,
            "p_value": two_sided_p(t_stat, dfree)}


# ── Sidebar ────────────────────────────────────────────────────────────────────
#this side bar creates a csv file data, and lets you filter for date range, signifigance level and which analysis you want
st.sidebar.title("Settings")

uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
default_path = "data/gold/btc_fng_with_holidays_api.csv"

if uploaded:
    df_full = load_data(uploaded)
else:
    try:
        df_full = load_data(default_path)
        st.sidebar.info(f"Using default: {default_path}")
    except FileNotFoundError:
        st.error(f"CSV not found. Upload a file or place `{default_path}` in the same folder.")
        st.stop()

# Date filter
min_date = df_full["date"].min().date()
max_date = df_full["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    df = df_full[(df_full["date"].dt.date >= date_range[0]) &
                 (df_full["date"].dt.date <= date_range[1])].copy()
else:
    df = df_full.copy()

st.sidebar.caption(f"{len(df)} rows in selected range")

# Alpha level selector
alpha = st.sidebar.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05)

# Page selector
page = st.sidebar.radio("Analysis", [
    "Overview", "T-Test Analysis", "Chi-Square Analysis",
    "Correlation Analysis", "Variance Test Analysis"
])

st.markdown('<div class="main-header">BTC Statistical Analysis Dashboard</div>', unsafe_allow_html=True)


# ── OVERVIEW ──────────────────────────────────────────────────────────────────
#provides summary of the dataset and visualizations
if page == "Overview":
    st.markdown('<div class="section-header">Dataset overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Fear days", int(df["fear"].sum()))
    c3.metric("Gain days", int(df["gain"].sum()))
    c4.metric("Holiday days", int(df["is_holiday"].sum()))

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-header">Return distributions</div>', unsafe_allow_html=True)

    chart_type = st.radio("Chart type", ["Histogram", "KDE"], horizontal=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, mask, label, color in [
        (axes[0], slice(None),       "All returns",     "steelblue"),
        (axes[1], df["fear"],        "Fear days",       "tomato"),
        (axes[2], ~df["fear"],       "Non-fear days",   "mediumseagreen"),
    ]:
        data = df.loc[mask, "btc_return_pct"] if not isinstance(mask, slice) else df["btc_return_pct"]
        if chart_type == "Histogram":
            ax.hist(data, bins=40, color=color, alpha=0.75, density=True)
        else:
            sns.kdeplot(data, ax=ax, color=color, fill=True, alpha=0.4)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_title(label)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")

    plt.tight_layout()
    st.pyplot(fig)

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation heatmap</div>', unsafe_allow_html=True)
    heatmap_cols = st.multiselect(
        "Columns to include",
        options=["btc_return_pct", "fng_value", "btc_volume", "days_to_holiday",
                 "is_holiday", "is_weekend", "fear", "gain"],
        default=["btc_return_pct", "fng_value", "is_holiday", "is_weekend"]
    )
    if len(heatmap_cols) >= 2:
        hm_data = df[heatmap_cols].copy()
        for col in ["is_holiday", "is_weekend", "fear", "gain"]:
            if col in hm_data:
                hm_data[col] = hm_data[col].astype(int)
        corr_matrix = hm_data.corr()
        fig2, ax2 = plt.subplots(figsize=(max(6, len(heatmap_cols)), max(5, len(heatmap_cols) - 1)))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax2)
        ax2.set_title("Correlation heatmap")
        st.pyplot(fig2)
    else:
        st.info("Select at least 2 columns.")


# ── T-TEST ────────────────────────────────────────────────────────────────────
elif page == "T-Test Analysis":
    st.markdown('<div class="section-header">T-test: Fear vs Non-Fear returns</div>', unsafe_allow_html=True)
#lets you toggle the null mean, default is zero
    mu_input = st.number_input("Null hypothesis mean (μ₀) for one-sample tests", value=0.0, step=0.001, format="%.4f")
#splits fear and non-fear
    fear_ret     = df.loc[df["fear"],  "btc_return_pct"].dropna().values
    non_fear_ret = df.loc[~df["fear"], "btc_return_pct"].dropna().values
#guards from empty groups
    if len(fear_ret) < 2 or len(non_fear_ret) < 2:
        st.warning("Not enough data in selected range.")
        st.stop()
#runs the tests then shows results
    res_fear     = one_sample_t(fear_ret,     mu=mu_input)
    res_non_fear = one_sample_t(non_fear_ret, mu=mu_input)
    res_welch    = welch_t(fear_ret, non_fear_ret)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Group statistics")
        st.write(f"**Fear days:** n = {res_fear['n']},  mean = {res_fear['mean']:.6f},  σ = {res_fear['std']:.6f}")
        st.write(f"**Non-fear days:** n = {res_non_fear['n']},  mean = {res_non_fear['mean']:.6f},  σ = {res_non_fear['std']:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="test-result">', unsafe_allow_html=True)
        st.subheader("Test results")

        st.write(f"**One-sample t-test (fear vs μ₀={mu_input}):**")
        st.write(f"t = {res_fear['t_stat']:.4f},  df = {res_fear['df']},  p = {res_fear['p_value']:.4f}")
        st.write("✅ Reject H₀" if res_fear["p_value"] < alpha else "❌ Fail to reject H₀")

        st.write(f"**One-sample t-test (non-fear vs μ₀={mu_input}):**")
        st.write(f"t = {res_non_fear['t_stat']:.4f},  df = {res_non_fear['df']},  p = {res_non_fear['p_value']:.4f}")
        st.write("✅ Reject H₀" if res_non_fear["p_value"] < alpha else "❌ Fail to reject H₀")

        st.write("**Welch two-sample t-test (fear vs non-fear):**")
        st.write(f"t = {res_welch['t_stat']:.4f},  df = {res_welch['df']:.2f},  p = {res_welch['p_value']:.4f}")
        st.write("✅ Reject H₀ — means differ" if res_welch["p_value"] < alpha else "❌ Fail to reject H₀")
        st.markdown('</div>', unsafe_allow_html=True)
#visualizs the returns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, label, color in [
        (axes[0], fear_ret,     "Fear days",         "tomato"),
        (axes[1], non_fear_ret, "Non-fear days",     "steelblue"),
        (axes[2], None,         "Fear vs Non-Fear",  None),
    ]:
        if data is not None:
            ax.hist(data, bins=30, alpha=0.7, color=color, density=True)
        else:
            ax.hist(fear_ret,     bins=25, alpha=0.5, color="tomato",    density=True, label="Fear")
            ax.hist(non_fear_ret, bins=25, alpha=0.5, color="steelblue", density=True, label="Non-Fear")
            ax.legend()
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.axvline(mu_input, color="orange", lw=1.5, ls=":", label=f"μ₀={mu_input}")
        ax.set_title(label)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
    plt.tight_layout()
    st.pyplot(fig)


# ── CHI-SQUARE ────────────────────────────────────────────────────────────────
elif page == "Chi-Square Analysis":
    st.markdown('<div class="section-header">Chi-square: Gain/Loss independence</div>', unsafe_allow_html=True)
#user can pick group variable to test gain/loss independence against
    group_col = st.selectbox("Group variable", ["is_holiday", "is_weekend", "fear"])

    ct = pd.crosstab(df["gain"], df[group_col])
    ct.index   = ["Loss", "Gain"]
    ct.columns = [f"Not {group_col.replace('is_','').title()}", group_col.replace('is_','').title()]
#runs the chi-square test and shows results
    chi2_stat, p_chi, dof, expected = chi2_contingency(ct)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Contingency table")
        st.dataframe(ct)
        st.subheader("Expected frequencies")
        exp_df = pd.DataFrame(expected.round(2), index=ct.index, columns=ct.columns)
        st.dataframe(exp_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="test-result">', unsafe_allow_html=True)
        st.subheader("Chi-square results")
        st.write(f"**χ² statistic:** {chi2_stat:.4f}")
        st.write(f"**p-value:** {p_chi:.4f}")
        st.write(f"**Degrees of freedom:** {dof}")
        st.write("✅ Reject H₀ — relationship exists" if p_chi < alpha else "❌ Fail to reject H₀ — variables are independent")
        st.markdown('</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pct_gain = df.groupby(group_col)["gain"].mean() * 100
    axes[0].bar([ct.columns[0], ct.columns[1]], pct_gain.values, color=["steelblue", "tomato"])
    axes[0].axhline(50, color="black", lw=1, ls="--", label="50% line")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% gain days")
    axes[0].set_title(f"Gain rate by {group_col}")
    axes[0].legend()
    for i, v in enumerate(pct_gain.values):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")

    for flag, label, color in [(True, ct.columns[1], "tomato"), (False, ct.columns[0], "steelblue")]:
        axes[1].hist(df.loc[df[group_col] == flag, "btc_return_pct"],
                     bins=25, alpha=0.5, density=True, color=color, label=label)
    axes[1].axvline(0, color="black", lw=1, ls="--")
    axes[1].set_title("Return distribution by group")
    axes[1].set_xlabel("Daily return")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)


# ── CORRELATION ───────────────────────────────────────────────────────────────
elif page == "Correlation Analysis":
    st.markdown('<div class="section-header">Spearman correlation</div>', unsafe_allow_html=True)
#user picks to numerics to correlate
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    col_x = st.selectbox("X variable", numeric_cols, index=numeric_cols.index("days_to_holiday") if "days_to_holiday" in numeric_cols else 0)
    col_y = st.selectbox("Y variable", numeric_cols, index=numeric_cols.index("btc_volume") if "btc_volume" in numeric_cols else 1)
    log_y = st.checkbox("Log scale on Y axis", value=True)

    corr_df = df[[col_x, col_y]].dropna()
    #spearman correlation
    
    corr, p_corr = spearmanr(corr_df[col_x], corr_df[col_y])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Summary statistics")
        st.write(f"**Observations:** {len(corr_df)}")
        st.write(f"**{col_x}** — mean: {corr_df[col_x].mean():.2f},  std: {corr_df[col_x].std():.2f}")
        st.write(f"**{col_y}** — mean: {corr_df[col_y].mean():.2f},  std: {corr_df[col_y].std():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="test-result">', unsafe_allow_html=True)
        st.subheader("Spearman results")
        st.write(f"**Correlation (ρ):** {corr:.4f}")
        st.write(f"**p-value:** {p_corr:.4f}")
        st.write("✅ Significant relationship" if p_corr < alpha else "❌ No significant relationship")
        direction = "increases" if corr < 0 else "decreases"
        abs_corr = abs(corr)
        strength = "very weak" if abs_corr < 0.1 else "weak" if abs_corr < 0.3 else "moderate" if abs_corr < 0.5 else "strong"
        st.write(f"**Direction:** {col_y} tends to {direction} as {col_x} increases.")
        st.write(f"**Strength:** {strength} (|ρ| = {abs_corr:.4f})")
        st.markdown('</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(corr_df[col_x], corr_df[col_y], alpha=0.55, color="#5b6abf", edgecolors="#3c4a9e", lw=0.4, s=40)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(col_x)
    ax.set_ylabel(f"{col_y}{' (log scale)' if log_y else ''}")
    ax.set_title(f"{col_y} vs {col_x}\nSpearman ρ = {corr:.4f},  p = {p_corr:.4f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# ── VARIANCE TEST ─────────────────────────────────────────────────────────────
elif page == "Variance Test Analysis":
    st.markdown('<div class="section-header">Variance test: returns by day type</div>', unsafe_allow_html=True)
#three pre-defined grouping to compare to
    group_options = {
        "Weekdays / Weekends / Holidays": [
            ("Weekdays", ~df["is_weekend"] & ~df["is_holiday"]),
            ("Weekends", df["is_weekend"]  & ~df["is_holiday"]),
            ("Holidays", df["is_holiday"]),
        ],
        "Fear / Non-Fear": [
            ("Fear",     df["fear"]),
            ("Non-Fear", ~df["fear"]),
        ],
        "Holiday / Not Holiday": [
            ("Holiday",     df["is_holiday"]),
            ("Not Holiday", ~df["is_holiday"]),
        ],
    }
    grouping = st.selectbox("Group by", list(group_options.keys()))
    group_defs = group_options[grouping]

    colors = ["#3266ad", "#d85a30", "#1d9e75", "#9b59b6"]
    groups, labels = [], []
    for label, mask in group_defs:
        g = df.loc[mask, "btc_return_pct"].dropna().values
        groups.append(g)
        labels.append(label)

    if any(len(g) < 2 for g in groups):
        st.warning("One or more groups have fewer than 2 observations. Adjust the date range.")
        st.stop()

    f_stat, p_anova = f_oneway(*groups)
    lev_stat, lev_p = levene(*groups)
    bart_stat, bart_p = bartlett(*groups)

    pair_results = []
    from itertools import combinations
    for (i, j) in combinations(range(len(groups)), 2):
        a, b = groups[i], groups[j]
        f_ratio = max(a.var(ddof=1), b.var(ddof=1)) / min(a.var(ddof=1), b.var(ddof=1))
        df1, df2 = len(a) - 1, len(b) - 1
        p_val = 2 * min(stats.f.cdf(f_ratio, df1, df2), stats.f.sf(f_ratio, df1, df2))
        pair_results.append({"label": f"{labels[i]} vs {labels[j]}", "F": f_ratio, "p": p_val})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Group statistics")
        for g, lbl in zip(groups, labels):
            st.write(f"**{lbl}:** n = {len(g)},  mean = {g.mean():.4f},  σ = {g.std():.4f},  var = {g.var(ddof=1):.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
#
    with c2:
        st.markdown('<div class="test-result">', unsafe_allow_html=True)
        st.subheader("Test results")
        for name, stat_val, p_val in [
            #three tests for variance/mean differences
            ("ANOVA F-test",    f_stat,    p_anova),
            ("Levene's test",   lev_stat,  lev_p),
            ("Bartlett's test", bart_stat, bart_p),
        ]:
            st.write(f"**{name}:** stat = {stat_val:.4f},  p = {p_val:.4f}")
            st.write("✅ Reject H₀ — variances/means differ" if p_val < alpha else "❌ Fail to reject H₀")

        st.subheader("Pairwise F-tests")
        for pr in pair_results:
            st.write(f"**{pr['label']}:** F = {pr['F']:.4f},  p = {pr['p']:.4f}  "
                     + ("✅" if pr["p"] < alpha else "❌"))
        st.markdown('</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for g, lbl, col in zip(groups, labels, colors):
        sns.kdeplot(g, ax=axes[0], color=col, label=f"{lbl} (σ={g.std():.3f})", fill=True, alpha=0.25)
    axes[0].axvline(0, color="black", lw=1, ls="--")
    axes[0].set_title("Return distributions")
    axes[0].set_xlabel("Daily return")
    axes[0].legend(fontsize=8)

    bp = axes[1].boxplot(groups, patch_artist=True, labels=labels)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55")
        patch.set_edgecolor(col)
    axes[1].set_title("Box plots")
    axes[1].set_ylabel("Daily return")

    variances = [g.var(ddof=1) for g in groups]
    bars = axes[2].bar(labels, variances, color=[c + "bb" for c in colors], edgecolor=colors, linewidth=1.5)
    for bar, var in zip(bars, variances):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(variances) * 0.01,
                     f"{var:.4f}", ha="center", va="bottom", fontsize=9)
    axes[2].set_title("Variance comparison")
    axes[2].set_ylabel("Variance")

    plt.tight_layout()
    st.pyplot(fig)
"""
Rotterdam Port Operations - Sections 4.1 (Understanding the Data)
and 4.3 (Understanding Cargo Volumes).

Run from the project root, so that the path Data/rotterdam_dataset_37.xlsx works.
Needs: pandas, numpy, scipy, matplotlib, openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FILE = "./data/rotterdam_dataset_37.xlsx"
os.makedirs("figures", exist_ok=True)

# ======================================================================
# Load data
# ======================================================================
df = pd.read_excel(FILE)

# Arrival_Date is stored as an Excel serial number (e.g. 44562.12).
# Convert it to a real date and time (Excel day 0 = 1899-12-30).
if np.issubdtype(df["Arrival_Date"].dtype, np.number):
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], unit="D",
                                        origin="1899-12-30")

types = ["Container", "Tanker", "Bulk"]

# ======================================================================
# 4.1 UNDERSTANDING THE DATA
# ======================================================================
print("=" * 70)
print("4.1 DATA QUALITY")
print("=" * 70)

# --- Structure ---
print("\nShape (rows, columns):", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nFirst rows:")
print(df.head())

# --- Missing values ---
print("\nMissing values per column:")
print(df.isna().sum())

# --- Duplicates ---
print("\nFully duplicated rows:", df.duplicated().sum())
print("Duplicated Vessel_IDs:  ", df["Vessel_ID"].duplicated().sum())

# --- Consistency ---
print("\nVessel types found:")
print(df["Vessel_Type"].value_counts())
unknown = ~df["Vessel_Type"].isin(types)
print("Rows with unknown vessel type:", unknown.sum())

print("\nNon-positive cargo values:   ", (df["Cargo_tons"] <= 0).sum())
print("Non-positive berth times:    ", (df["Berth_Time_hr"] <= 0).sum())

print("\nFirst arrival:", df["Arrival_Date"].min())
print("Last arrival: ", df["Arrival_Date"].max())
print("Period covered (days):",
      (df["Arrival_Date"].max() - df["Arrival_Date"].min()).days)
print("\nVessels per year:")
print(df["Arrival_Date"].dt.year.value_counts().sort_index())

# --- Unusual data points (IQR rule, per vessel type) ---
print("\nOutliers according to the 1.5*IQR rule (per vessel type):")
for t in types:
    sub = df[df["Vessel_Type"] == t]
    for col in ["Cargo_tons", "Berth_Time_hr"]:
        q1, q3 = sub[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((sub[col] < low) | (sub[col] > high)).sum()
        print(f"  {t:10s} {col:14s}: {n_out:4d} of {len(sub)} "
              f"(limits {low:,.1f} to {high:,.1f})")

# --- Descriptive statistics ---
print("\nDescriptive statistics (all vessels):")
print(df[["Cargo_tons", "Berth_Time_hr"]].describe())

print("\nDescriptive statistics per vessel type:")
print(df.groupby("Vessel_Type")[["Cargo_tons", "Berth_Time_hr"]]
      .describe().T)

print("\nCorrelation between cargo and berth time:")
print("  overall:", round(df["Cargo_tons"].corr(df["Berth_Time_hr"]), 3))
for t in types:
    sub = df[df["Vessel_Type"] == t]
    print(f"  {t:10s}:", round(sub["Cargo_tons"].corr(sub["Berth_Time_hr"]), 3))

# --- Plots 4.1 ---
# Vessel counts per type
plt.figure(figsize=(5, 4))
df["Vessel_Type"].value_counts().reindex(types).plot(kind="bar")
plt.title("Number of vessel calls per type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/4_1_counts_per_type.png", dpi=150)

# Boxplots per type
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
df.boxplot(column="Cargo_tons", by="Vessel_Type", ax=axes[0])
axes[0].set_title("Cargo (tons) by vessel type")
df.boxplot(column="Berth_Time_hr", by="Vessel_Type", ax=axes[1])
axes[1].set_title("Berth time (hr) by vessel type")
fig.suptitle("")
plt.tight_layout()
plt.savefig("figures/4_1_boxplots.png", dpi=150)

# Histograms of cargo and berth time
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["Cargo_tons"], bins=50, edgecolor="black")
axes[0].set_title("Cargo (tons), all vessels")
axes[0].set_xlabel("tons")
axes[1].hist(df["Berth_Time_hr"], bins=50, edgecolor="black")
axes[1].set_title("Berth time (hr), all vessels")
axes[1].set_xlabel("hours")
plt.tight_layout()
plt.savefig("figures/4_1_histograms.png", dpi=150)

# Cargo versus berth time
plt.figure(figsize=(6, 5))
for t in types:
    sub = df[df["Vessel_Type"] == t]
    plt.scatter(sub["Cargo_tons"], sub["Berth_Time_hr"], s=8, alpha=0.5,
                label=t)
plt.xlabel("Cargo (tons)")
plt.ylabel("Berth time (hr)")
plt.title("Berth time versus cargo")
plt.legend()
plt.tight_layout()
plt.savefig("figures/4_1_cargo_vs_berth.png", dpi=150)

# Vessel calls per day (quick look at data over time)
daily = df.groupby(df["Arrival_Date"].dt.floor("D")).size()
plt.figure(figsize=(11, 3.5))
plt.plot(daily.index, daily.values, linewidth=0.6)
plt.title("Vessel arrivals per day")
plt.ylabel("Vessels")
plt.tight_layout()
plt.savefig("figures/4_1_arrivals_per_day.png", dpi=150)

# ======================================================================
# 4.3 UNDERSTANDING CARGO VOLUMES
# ======================================================================
print("\n" + "=" * 70)
print("4.3 CARGO VOLUMES")
print("=" * 70)

# --- Distributional shape, skewness, kurtosis per vessel type ---
rows = []
for t in types + ["All"]:
    x = df["Cargo_tons"] if t == "All" else \
        df.loc[df["Vessel_Type"] == t, "Cargo_tons"]
    rows.append({
        "Type": t,
        "n": len(x),
        "Mean": x.mean(),
        "Std": x.std(),
        "CV": x.std() / x.mean(),
        "Median": x.median(),
        "Skewness": stats.skew(x),
        "Excess kurtosis": stats.kurtosis(x),
        "Max": x.max(),
    })
shape_table = pd.DataFrame(rows).set_index("Type")
print("\nShape of the cargo distribution:")
print(shape_table.round(3))

# --- Extreme observations ---
print("\nExtreme observations:")
for t in types:
    x = df.loc[df["Vessel_Type"] == t, "Cargo_tons"].sort_values()
    q99 = x.quantile(0.99)
    top1 = x[x >= q99]
    share = top1.sum() / x.sum()
    print(f"  {t:10s}: 99% quantile = {q99:,.0f}, max = {x.max():,.0f}, "
          f"top 1% of vessels carry {share:.1%} of the tons")

# --- Fit candidate distributions per vessel type ---
# Candidates: lognormal, gamma, Weibull (all with location fixed at 0),
# fitted by maximum likelihood. The lognormal is also fitted by the
# method of moments, as needed for the basic scenario.
candidates = {
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
}

fit_params = {}      # fit_params[type][name] = parameters
results = []

for t in types:
    x = df.loc[df["Vessel_Type"] == t, "Cargo_tons"].values
    fit_params[t] = {}
    for name, dist in candidates.items():
        params = dist.fit(x, floc=0)
        fit_params[t][name] = params
        loglik = np.sum(dist.logpdf(x, *params))
        k = 2                                   # free parameters (loc fixed)
        aic = 2 * k - 2 * loglik
        bic = k * np.log(len(x)) - 2 * loglik
        ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)
        results.append({"Type": t, "Distribution": name,
                        "LogLik": loglik, "AIC": aic, "BIC": bic,
                        "KS stat": ks_stat, "KS p-value": ks_p})

fit_table = pd.DataFrame(results)
print("\nGoodness of fit (lower AIC/BIC and KS stat = better):")
print(fit_table.round(4).to_string(index=False))
print("\nNote: KS p-values are optimistic because the parameters were "
      "estimated from the same data.")

best = fit_table.loc[fit_table.groupby("Type")["AIC"].idxmin(),
                     ["Type", "Distribution", "AIC"]]
print("\nBest distribution per vessel type according to AIC:")
print(best.to_string(index=False))

# --- Method of moments lognormal (used in the basic scenario) ---
print("\nLognormal parameters, method of moments:")
mom = {}
for t in types:
    x = df.loc[df["Vessel_Type"] == t, "Cargo_tons"]
    m, v = x.mean(), x.var()
    sigma2 = np.log(1 + v / m**2)
    mu = np.log(m) - sigma2 / 2
    mom[t] = (mu, np.sqrt(sigma2))
    print(f"  {t:10s}: mu = {mu:.4f}, sigma = {np.sqrt(sigma2):.4f}")

print("\nLognormal parameters, maximum likelihood (for comparison):")
for t in types:
    s, loc, scale = fit_params[t]["Lognormal"]
    print(f"  {t:10s}: mu = {np.log(scale):.4f}, sigma = {s:.4f}")

# --- Plots 4.3 ---
# Histograms with fitted densities
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, t in zip(axes, types):
    x = df.loc[df["Vessel_Type"] == t, "Cargo_tons"].values
    grid = np.linspace(x.min(), x.max(), 400)
    ax.hist(x, bins=40, density=True, alpha=0.5, edgecolor="black")
    for name, dist in candidates.items():
        ax.plot(grid, dist.pdf(grid, *fit_params[t][name]), label=name)
    ax.set_title(f"{t}: cargo (tons)")
    ax.set_xlabel("tons")
    ax.legend()
plt.tight_layout()
plt.savefig("figures/4_3_fitted_densities.png", dpi=150)

# Histograms of log(cargo) - bell shaped means lognormal is plausible
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, t in zip(axes, types):
    x = df.loc[df["Vessel_Type"] == t, "Cargo_tons"].values
    ax.hist(np.log(x), bins=40, edgecolor="black")
    ax.set_title(f"{t}: log(cargo)")
    ax.set_xlabel("log(tons)")
plt.tight_layout()
plt.savefig("figures/4_3_log_histograms.png", dpi=150)

# QQ plots: rows = vessel type, columns = candidate distribution
fig, axes = plt.subplots(3, 3, figsize=(12, 11))
for i, t in enumerate(types):
    x = np.sort(df.loc[df["Vessel_Type"] == t, "Cargo_tons"].values)
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    for j, (name, dist) in enumerate(candidates.items()):
        theo = dist.ppf(probs, *fit_params[t][name])
        ax = axes[i, j]
        ax.scatter(theo, x, s=6)
        lims = [min(theo.min(), x.min()), max(theo.max(), x.max())]
        ax.plot(lims, lims, "r--")
        ax.set_title(f"{t} - {name}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
plt.tight_layout()
plt.savefig("figures/4_3_qqplots.png", dpi=150)

# Log-scale tail plot (survival function) to judge the tail fit
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, t in zip(axes, types):
    x = np.sort(df.loc[df["Vessel_Type"] == t, "Cargo_tons"].values)
    n = len(x)
    surv = 1 - (np.arange(1, n + 1) - 0.5) / n
    ax.loglog(x, surv, ".", label="Data")
    for name, dist in candidates.items():
        ax.loglog(x, dist.sf(x, *fit_params[t][name]), label=name)
    ax.set_title(f"{t}: tail P(X > x)")
    ax.set_xlabel("tons")
    ax.legend()
plt.tight_layout()
plt.savefig("figures/4_3_tail_plots.png", dpi=150)

plt.show()
print("\nDone. Figures saved in the 'figures' folder.")
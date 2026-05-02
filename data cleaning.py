import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import warnings
import sys
import io
warnings.filterwarnings('ignore')
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── 1. LOAD RAW DATA ───────────────────────────────────────────────────────
df_raw = pd.read_csv(r'C:\Users\PC\Downloads\messy_dataset_Mukesh.csv')
print("=" * 60)
print("RAW DATA")
print("=" * 60)
print(df_raw.to_string())
print(f"\nShape: {df_raw.shape}")
print("\nData types:\n", df_raw.dtypes)
print("\nMissing values:\n", df_raw.isnull().sum())

# ─── 2. DOCUMENT ALL ISSUES ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ISSUES IDENTIFIED")
print("=" * 60)
issues = [
    "Duplicate ID=2 for 'Bob' across two rows (complementary missing values)",
    "Missing ID for 'Eve' (row 7)",
    "Age 'thirty-eight' (text) for David — should be 38",
    "Salary 'sixty five thousand' (text) for ID=7 — should be 65000",
    "Invalid date '2019-13-01' for Eve (month=13 doesn't exist)",
    "Missing Join Date for Charlie",
    "Missing Age for Bob (one row) and Heidi",
    "Missing Salary for Bob (one row) and Heidi",
    "Missing Country for Grace",
    "Inconsistent country code: 'AU' should be 'AUS'",
]
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")

# ─── 3. CLEANING ────────────────────────────────────────────────────────────
df = df_raw.copy()

# 3a. Fix text-encoded numbers
df['Age'] = df['Age'].replace('thirty-eight', 38)
df['Salary'] = df['Salary'].replace('sixty five thousand', 65000)

# 3b. Convert columns to numeric (coerce turns remaining non-numeric to NaN)
df['ID']     = pd.to_numeric(df['ID'],     errors='coerce')
df['Age']    = pd.to_numeric(df['Age'],    errors='coerce')
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# 3c. Merge duplicate Bob rows (ID=2): fill NaN from the sibling row
bob_mask = df['ID'] == 2
bob_merged = df[bob_mask].ffill().bfill().iloc[[0]]
df = pd.concat([df[~bob_mask], bob_merged], ignore_index=True)

# 3d. Assign missing ID for Eve (next logical value = 6)
df.loc[df['Name'] == 'Eve', 'ID'] = 6

# 3e. Standardise country codes
df['Country'] = df['Country'].replace('AU', 'AUS')

# 3f. Parse dates — handle multiple formats; invalid dates → NaT
def parse_date(val):
    if pd.isnull(val):
        return pd.NaT
    for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            pass
    return pd.NaT  # e.g. '2019-13-01' (month 13) → NaT

df['Join Date'] = df['Join Date'].apply(parse_date)

# 3g. Sort by ID and reset index
df = df.sort_values('ID').reset_index(drop=True)
df['ID'] = df['ID'].astype(int)

# 3h. Impute remaining numeric NaNs with column median
df['Age']    = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# 3i. Impute missing Country with mode
df['Country'] = df['Country'].fillna(df['Country'].mode()[0])

# 3j. Impute missing Join Date with median date
valid_dates = df['Join Date'].dropna()
median_date = valid_dates.sort_values().iloc[len(valid_dates) // 2]
df['Join Date'] = df['Join Date'].fillna(median_date)

print("\n" + "=" * 60)
print("CLEANED DATA")
print("=" * 60)
print(df.to_string())
print(f"\nRemaining NaNs:\n{df.isnull().sum()}")

# ─── 4. OUTLIER DETECTION (IQR method) ──────────────────────────────────────
print("\n" + "=" * 60)
print("OUTLIER DETECTION (IQR)")
print("=" * 60)
numeric_cols = ['Age', 'Salary']
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lo) | (df[col] > hi)]
    outlier_summary[col] = outliers
    print(f"  {col}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}, "
          f"bounds=[{lo:.1f}, {hi:.1f}]  →  {len(outliers)} outlier(s)")
    if not outliers.empty:
        print(outliers[['ID', 'Name', col]].to_string(index=False))

# ─── 5. PEARSON CORRELATION ─────────────────────────────────────────────────
df['Join_Year'] = df['Join Date'].dt.year
corr_df = df[['Age', 'Salary', 'Join_Year']].copy()
pearson_corr = corr_df.corr(method='pearson')
print("\n" + "=" * 60)
print("PEARSON CORRELATION MATRIX")
print("=" * 60)
print(pearson_corr.round(4).to_string())

# ─── 6. VISUALISATIONS ──────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')

# Two-section figure: top 6 charts, then full-width heatmap below
fig = plt.figure(figsize=(20, 18))
fig.suptitle("Data Cleaning & Visualization — messy_dataset_Mukesh.csv",
             fontsize=17, fontweight='bold', y=0.98)

# Outer grid: row 0 = charts (2/3 height), row 1 = heatmap (1/3 height)
outer = gridspec.GridSpec(2, 1, figure=fig,
                          height_ratios=[2, 1],
                          hspace=0.45)

# Inner grid for 6 charts (2 rows × 3 cols)
inner = gridspec.GridSpecFromSubplotSpec(2, 3,
                                         subplot_spec=outer[0],
                                         hspace=0.45, wspace=0.38)

# — 6a. Salary distribution ───────────────────────────────────────
ax1 = fig.add_subplot(inner[0, 0])
ax1.hist(df['Salary'], bins=6, color='steelblue', edgecolor='white')
ax1.set_title('Salary Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Salary ($)')
ax1.set_ylabel('Count')
ax1.xaxis.set_tick_params(labelsize=9)

# — 6b. Age distribution ──────────────────────────────────────────
ax2 = fig.add_subplot(inner[0, 1])
ax2.hist(df['Age'], bins=6, color='coral', edgecolor='white')
ax2.set_title('Age Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')

# — 6c. Country count ─────────────────────────────────────────────
ax3 = fig.add_subplot(inner[0, 2])
country_counts = df['Country'].value_counts()
bar_colors = ['#2196F3', '#FF9800', '#4CAF50']
bars = ax3.bar(country_counts.index, country_counts.values,
               color=bar_colors[:len(country_counts)], width=0.5)
ax3.set_title('Employees by Country', fontsize=12, fontweight='bold')
ax3.set_xlabel('Country')
ax3.set_ylabel('Count')
ax3.set_ylim(0, country_counts.max() + 1.2)
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.1,
             str(int(bar.get_height())), ha='center', va='bottom', fontsize=11)

# — 6d. Age Boxplot ───────────────────────────────────────────────
ax4 = fig.add_subplot(inner[1, 0])
bp1 = ax4.boxplot(df['Age'], patch_artist=True, widths=0.4,
                  boxprops=dict(facecolor='#FFD700', color='#555'),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='#555'),
                  capprops=dict(color='#555'),
                  flierprops=dict(marker='o', color='red', markersize=7))
ax4.set_title('Age — Boxplot (Outlier Check)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Age')
ax4.set_xticks([1])
ax4.set_xticklabels(['Age'])
# annotate IQR bounds
Q1_a = df['Age'].quantile(0.25); Q3_a = df['Age'].quantile(0.75)
ax4.axhline(Q1_a - 1.5*(Q3_a-Q1_a), color='green', linestyle=':', linewidth=1, label='IQR bounds')
ax4.axhline(Q3_a + 1.5*(Q3_a-Q1_a), color='green', linestyle=':', linewidth=1)
ax4.legend(fontsize=8, loc='upper right')

# — 6e. Salary Boxplot ────────────────────────────────────────────
ax5 = fig.add_subplot(inner[1, 1])
ax5.boxplot(df['Salary'], patch_artist=True, widths=0.4,
            boxprops=dict(facecolor='#90EE90', color='#555'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='#555'),
            capprops=dict(color='#555'),
            flierprops=dict(marker='o', color='red', markersize=7))
ax5.set_title('Salary — Boxplot (Outlier Check)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Salary ($)')
ax5.set_xticks([1])
ax5.set_xticklabels(['Salary'])
Q1_s = df['Salary'].quantile(0.25); Q3_s = df['Salary'].quantile(0.75)
ax5.axhline(Q1_s - 1.5*(Q3_s-Q1_s), color='green', linestyle=':', linewidth=1, label='IQR bounds')
ax5.axhline(Q3_s + 1.5*(Q3_s-Q1_s), color='green', linestyle=':', linewidth=1)
ax5.legend(fontsize=8, loc='upper right')

# — 6f. Scatter: Age vs Salary ────────────────────────────────────
ax6 = fig.add_subplot(inner[1, 2])
dot_colors = {'NZ': '#2196F3', 'AUS': '#FF9800'}
for country, grp in df.groupby('Country'):
    ax6.scatter(grp['Age'], grp['Salary'],
                label=country, color=dot_colors.get(country, 'grey'),
                s=90, edgecolors='black', linewidth=0.6, zorder=3)

# Smart label placement to avoid overlaps
from adjustText import adjust_text
texts = []
for _, row in df.dropna(subset=['Name']).iterrows():
    texts.append(ax6.text(row['Age'], row['Salary'], row['Name'], fontsize=7.5))
try:
    adjust_text(texts, ax=ax6,
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))
except Exception:
    for t in texts:
        t.set_fontsize(7)

m, b = np.polyfit(df['Age'], df['Salary'], 1)
x_line = np.linspace(df['Age'].min() - 1, df['Age'].max() + 1, 200)
ax6.plot(x_line, m * x_line + b, 'r--', linewidth=1.2, label='Trend line')
ax6.set_title(f'Age vs Salary  (Pearson r = {pearson_corr.loc["Age","Salary"]:.3f})',
              fontsize=11, fontweight='bold')
ax6.set_xlabel('Age')
ax6.set_ylabel('Salary ($)')
ax6.legend(fontsize=8)

# — 6g. Pearson Correlation Heatmap (full-width bottom section) ───
# Use a dedicated sub-figure so we control exact cell size
heatmap_spec = outer[1].subgridspec(1, 3, width_ratios=[1, 2, 1])
ax7 = fig.add_subplot(heatmap_spec[0, 1])   # centre column only

corr_display = pearson_corr.copy()
corr_display.index   = ['Age', 'Salary', 'Join Year']
corr_display.columns = ['Age', 'Salary', 'Join Year']

# Two-line annotation: value on top, interpretation label below
def make_label(val):
    sign = '+' if val >= 0 else ''
    bar = '|||||' if abs(val) > 0.6 else '|||' if abs(val) > 0.3 else '|'
    return f'{sign}{val:.4f}'

annot_text = corr_display.applymap(make_label)

hm = sns.heatmap(
    corr_display,
    annot=annot_text, fmt='',
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    linewidths=3, linecolor='white',
    annot_kws={'size': 14, 'weight': 'bold', 'family': 'monospace'},
    ax=ax7, square=False,
    cbar_kws={'shrink': 0.8, 'label': 'Pearson r', 'pad': 0.04}
)

ax7.set_title('Pearson Correlation Heatmap\n(Pearson Algorithm — Linear Correlation)',
              fontsize=13, fontweight='bold', pad=10)
ax7.tick_params(axis='both', labelsize=12)
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=0, fontsize=12)
ax7.set_yticklabels(ax7.get_yticklabels(), rotation=0, fontsize=12)

output_path = r'C:\Users\PC\Documents\CW\Data Clean\data_cleaning_Mukesh.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved -> {output_path}")

# ─── 7. SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL CLEANED DATASET")
print("=" * 60)
print(df[['ID','Name','Age','Country','Salary','Join Date']].to_string(index=False))

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
r_age_sal = pearson_corr.loc['Age', 'Salary']
r_age_yr  = pearson_corr.loc['Age', 'Join_Year']
r_sal_yr  = pearson_corr.loc['Salary', 'Join_Year']

print(f"  • Age ↔ Salary      : r = {r_age_sal:.4f}  → "
      + ("moderate positive linear relationship" if r_age_sal > 0.4
         else "weak/no linear relationship"))
print(f"  • Age ↔ Join Year   : r = {r_age_yr:.4f}  → "
      + ("moderate positive" if r_age_yr > 0.4
         else "weak/negative — older employees joined earlier"))
print(f"  • Salary ↔ Join Year: r = {r_sal_yr:.4f}  → "
      + ("moderate" if abs(r_sal_yr) > 0.4 else "weak relationship"))
print(f"\n  • Average Salary: ${df['Salary'].mean():,.0f}")
print(f"  • Average Age:    {df['Age'].mean():.1f} years")
print(f"  • Countries:      {', '.join(df['Country'].unique())}")

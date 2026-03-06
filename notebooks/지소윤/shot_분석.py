import pandas as pd
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

t1 = pd.read_csv('../../data/df_final_Type1.csv')
t2 = pd.read_csv('../../data/df_final_Type2.csv')

# ============================================================
# 1. 기본 통계
# ============================================================
print("=== Type1 Shot 기본 통계 ===")
print(t1['Shot'].describe())
print(f"unique 수: {t1['Shot'].nunique()}")

print("\n=== Type2 Shot 기본 통계 ===")
print(t2['Shot'].describe())
print(f"unique 수: {t2['Shot'].nunique()}")

# ============================================================
# 2. 양품/불량별 Shot 통계
# ============================================================
print("\n=== Type1 Shot - 양품/불량별 ===")
print(t1.groupby('Defect_Status')['Shot'].describe().round(1))

print("\n=== Type2 Shot - 양품/불량별 ===")
print(t2.groupby('Defect_Status')['Shot'].describe().round(1))

# ============================================================
# 3. Shot 구간별 불량률
# ============================================================
t1['Shot_구간'] = pd.cut(t1['Shot'],
    bins=[0, 200, 400, 600, 800, 1000, 1300],
    labels=['0-200', '201-400', '401-600', '601-800', '801-1000', '1001+'])

t2['Shot_구간'] = pd.cut(t2['Shot'],
    bins=[0, 100, 200, 300, 400, 500, 600, 733],
    labels=['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601+'])

print("\n=== Type1 Shot 구간별 불량률 ===")
t1_grp = t1.groupby('Shot_구간')['Defect_Status'].agg(
    전체='count', 불량='sum', 불량률='mean').reset_index()
t1_grp['불량률'] = (t1_grp['불량률'] * 100).round(1)
print(t1_grp.to_string(index=False))

print("\n=== Type2 Shot 구간별 불량률 ===")
t2_grp = t2.groupby('Shot_구간')['Defect_Status'].agg(
    전체='count', 불량='sum', 불량률='mean').reset_index()
t2_grp['불량률'] = (t2_grp['불량률'] * 100).round(1)
print(t2_grp.to_string(index=False))

# ============================================================
# 4. 시각화
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#F8FAFB')

# Type1 Shot 분포 (양품/불량)
for label, color in [(0, '#2ECC71'), (1, '#E74C3C')]:
    axes[0, 0].hist(t1[t1['Defect_Status'] == label]['Shot'],
                    bins=40, alpha=0.6, color=color,
                    label='양품' if label == 0 else '불량')
axes[0, 0].set_title('Type1 - Shot 분포 (양품 vs 불량)', fontweight='bold')
axes[0, 0].set_xlabel('Shot'); axes[0, 0].set_ylabel('빈도')
axes[0, 0].legend(); axes[0, 0].grid(linestyle='--', alpha=0.4)

# Type2 Shot 분포 (양품/불량)
for label, color in [(0, '#2ECC71'), (1, '#E74C3C')]:
    axes[0, 1].hist(t2[t2['Defect_Status'] == label]['Shot'],
                    bins=40, alpha=0.6, color=color,
                    label='양품' if label == 0 else '불량')
axes[0, 1].set_title('Type2 - Shot 분포 (양품 vs 불량)', fontweight='bold')
axes[0, 1].set_xlabel('Shot'); axes[0, 1].set_ylabel('빈도')
axes[0, 1].legend(); axes[0, 1].grid(linestyle='--', alpha=0.4)

# Type1 구간별 불량률
axes[1, 0].bar(t1_grp['Shot_구간'].astype(str), t1_grp['불량률'], color='#E74C3C', alpha=0.8)
axes[1, 0].set_title('Type1 - Shot 구간별 불량률 (%)', fontweight='bold')
axes[1, 0].set_xlabel('Shot 구간'); axes[1, 0].set_ylabel('불량률 (%)')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.4)
for i, v in enumerate(t1_grp['불량률']):
    axes[1, 0].text(i, v + 0.3, f'{v}%', ha='center', fontsize=9)

# Type2 구간별 불량률
axes[1, 1].bar(t2_grp['Shot_구간'].astype(str), t2_grp['불량률'], color='#3498DB', alpha=0.8)
axes[1, 1].set_title('Type2 - Shot 구간별 불량률 (%)', fontweight='bold')
axes[1, 1].set_xlabel('Shot 구간'); axes[1, 1].set_ylabel('불량률 (%)')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.4)
for i, v in enumerate(t2_grp['불량률']):
    axes[1, 1].text(i, v + 0.3, f'{v}%', ha='center', fontsize=9)

plt.suptitle('Shot 구간별 불량 분석', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 5. 카이제곱 검정 — 구간별 불량률 차이 유의성
# ============================================================
from scipy.stats import chi2_contingency
from itertools import combinations
import numpy as np

def chi2_test(df, label):
    grp = df.groupby('Shot_구간')['Defect_Status'].agg(불량='sum', 전체='count')
    grp['양품'] = grp['전체'] - grp['불량']
    contingency = grp[['불량', '양품']].values

    chi2, p, dof, _ = chi2_contingency(contingency)
    print(f"\n[{label}] 전체 카이제곱 검정")
    print(f"  chi2={chi2:.4f}, p-value={p:.4f}, dof={dof}")
    if p < 0.05:
        print("  → 구간 간 불량률 차이 통계적으로 유의함 (p < 0.05)")
    else:
        print("  → 구간 간 불량률 차이 유의하지 않음 (p >= 0.05)")

    # 사후 검정: 구간 쌍별 카이제곱 + Bonferroni 보정
    labels = grp.index.tolist()
    pairs = list(combinations(range(len(labels)), 2))
    n_pairs = len(pairs)
    alpha_corrected = 0.05 / n_pairs  # Bonferroni 보정

    print(f"\n  [사후 검정 - Bonferroni 보정, 유의수준 {alpha_corrected:.4f}]")
    print(f"  {'구간1':<12} {'구간2':<12} {'p-value':>10}  판정")
    print("  " + "-" * 50)

    sig_pairs = []
    for i, j in pairs:
        sub = contingency[[i, j], :]
        _, p_ij, _, _ = chi2_contingency(sub)
        sig = "* 유의" if p_ij < alpha_corrected else ""
        print(f"  {labels[i]:<12} {labels[j]:<12} {p_ij:>10.4f}  {sig}")
        if p_ij < alpha_corrected:
            sig_pairs.append((labels[i], labels[j], p_ij))

    if sig_pairs:
        print(f"\n  유의한 쌍 ({len(sig_pairs)}개):")
        for a, b, p_val in sig_pairs:
            print(f"    {a} vs {b}  (p={p_val:.4f})")
    else:
        print("\n  Bonferroni 보정 후 유의한 쌍 없음")

chi2_test(t1, 'Type1')
chi2_test(t2, 'Type2')

# ============================================================
# 6. 생산코드(id 앞 1자리)별 불량률 차이 검정
# ============================================================
for df in [t1, t2]:
    df['id_str']  = df['id'].astype(str).str.zfill(7)
    df['생산코드'] = df['id_str'].str[:1]

def chi2_by_code(df, label):
    grp = df.groupby('생산코드')['Defect_Status'].agg(불량='sum', 전체='count')
    grp['양품']   = grp['전체'] - grp['불량']
    grp['불량률'] = (grp['불량'] / grp['전체'] * 100).round(1)

    print(f"\n[{label}] 생산코드별 불량률")
    print(grp[['전체', '불량', '불량률']].to_string())

    ct = grp[['불량', '양품']].values
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"\n전체 카이제곱: chi2={chi2:.2f}, p={p:.4f}")
    if p < 0.05:
        print("→ 생산코드 간 불량률 차이 유의함 (p < 0.05)")
    else:
        print("→ 생산코드 간 불량률 차이 유의하지 않음")

    codes = grp.index.tolist()
    pairs = list(combinations(range(len(codes)), 2))
    alpha_b = 0.05 / len(pairs)
    print(f"\n사후 검정 (Bonferroni 보정 유의수준: {alpha_b:.4f})")
    print(f"  {'코드A':<8} {'코드B':<8} {'불량률A':>8} {'불량률B':>8} {'p-value':>10}  판정")
    print("  " + "-" * 55)
    for i, j in pairs:
        sub = ct[[i, j], :]
        _, p_ij, _, _ = chi2_contingency(sub)
        sig = "* 유의" if p_ij < alpha_b else ""
        ra = grp['불량률'].iloc[i]
        rb = grp['불량률'].iloc[j]
        print(f"  {codes[i]:<8} {codes[j]:<8} {ra:>7.1f}% {rb:>7.1f}% {p_ij:>10.4f}  {sig}")

chi2_by_code(t1, 'Type1')
chi2_by_code(t2, 'Type2')

# ============================================================
# 7. 생산코드별 불량률 시각화
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#F8FAFB')

colors_t1 = ['#E74C3C', '#E67E22', '#F39C12', '#2ECC71', '#1ABC9C']
colors_t2 = ['#1ABC9C', '#3498DB', '#9B59B6', '#E91E63']

for ax, df, label, colors in [
    (axes[0], t1, 'Type1', colors_t1),
    (axes[1], t2, 'Type2', colors_t2),
]:
    grp = df.groupby('생산코드')['Defect_Status'].mean() * 100
    bars = ax.bar(grp.index, grp.values, color=colors[:len(grp)], alpha=0.85, width=0.6)
    ax.set_title(f'{label} - 생산코드별 불량률', fontweight='bold')
    ax.set_xlabel('생산코드 (id 앞 1자리)')
    ax.set_ylabel('불량률 (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('생산코드별 불량률 비교', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

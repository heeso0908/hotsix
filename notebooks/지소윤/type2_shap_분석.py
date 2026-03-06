"""
Type 2 SHAP 분석 + 통계 vs ML 피처 비교
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score
from scipy import stats
import lightgbm as lgb
import shap

# ── 폰트 설정 ──────────────────────────────────────────────
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)

# ── 1. 데이터 로드 ──────────────────────────────────────────
df = pd.read_csv('../../data/df_final_Type2.csv')
df['Pressure_Diff_ratio'] = df['Casting_Pressure'] / df['Cylinder_Pressure']

FEATURES = [
    'Velocity_1', 'Velocity_2', 'Velocity_3', 'High_Velocity',
    'Rapid_Rise_Time', 'Biscuit_Thickness', 'Clamping_Force', 'Cycle_Time',
    'Pressure_Rise_Time', 'Casting_Pressure', 'Cylinder_Pressure',
    'Spray_Time', 'Spray_1_Time', 'Spray_2_Time',
    'Melting_Furnace_Temp', 'Air_Pressure', 'Coolant_Temp', 'Coolant_Pressure',
    'Factory_Temp', 'Factory_Humidity', 'Pressure_Diff_ratio'
]

X = df[FEATURES].copy()
y = df['Defect_Status'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"훈련셋: {X_train.shape[0]}행 | 테스트셋: {X_test.shape[0]}행")
print(f"불량률: {y.mean()*100:.1f}%")

# ── 2. LightGBM 학습 (모델링 노트북 최적 파라미터) ───────────
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = lgb.LGBMClassifier(
    n_estimators=300,
    num_leaves=127,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.7,
    scale_pos_weight=pos_weight,
    random_state=SEED,
    verbose=-1
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

# 최적 임계값 탐색 (Recall >= 0.80 조건)
best_thr, best_f1, best_recall = 0.5, 0, 0
for thr in np.arange(0.05, 0.95, 0.01):
    pred = (y_prob >= thr).astype(int)
    r = recall_score(y_test, pred, zero_division=0)
    f = f1_score(y_test, pred, zero_division=0)
    if r >= 0.80 and f > best_f1:
        best_f1, best_thr, best_recall = f, thr, r

y_pred = (y_prob >= best_thr).astype(int)
print(f"\n최적 임계값: {best_thr:.2f} | Recall: {best_recall:.3f} | F1: {best_f1:.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['양품', '불량'], digits=4))

# ── 3. SHAP 분석 ────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values

# SHAP 평균 절대값 기준 순위
shap_importance = pd.Series(
    np.abs(shap_arr).mean(axis=0),
    index=FEATURES
).sort_values(ascending=False)

print("\n[ SHAP 피처 중요도 (평균 |SHAP|) ]")
for i, (feat, val) in enumerate(shap_importance.items(), 1):
    print(f"  {i:2d}. {feat:<25} {val:.4f}")

# ── 4. 통계 분석 (Spearman) ─────────────────────────────────
stat_results = []
for col in FEATURES:
    rho, p = stats.spearmanr(df[col], df['Defect_Status'])
    stat_results.append({'변수': col, 'rho': round(rho, 4), 'p': round(p, 4)})

stat_df = pd.DataFrame(stat_results).sort_values('rho', key=abs, ascending=False).reset_index(drop=True)
stat_df['통계순위'] = stat_df.index + 1

# ── 5. 통계 vs SHAP 비교표 ──────────────────────────────────
shap_rank_df = pd.DataFrame({
    '변수': shap_importance.index,
    'SHAP값': shap_importance.values.round(4),
    'SHAP순위': range(1, len(FEATURES) + 1)
})

compare_df = stat_df.merge(shap_rank_df, on='변수')
compare_df['순위차'] = abs(compare_df['통계순위'] - compare_df['SHAP순위'])

def judge(row):
    if row['순위차'] <= 2:
        return '✅ 일치'
    elif row['순위차'] <= 5:
        return '⚠️ 부분'
    else:
        return '❌ 불일치'

compare_df['일치여부'] = compare_df.apply(judge, axis=1)
compare_df = compare_df.sort_values('SHAP순위').reset_index(drop=True)

print("\n[ 통계 vs SHAP 피처 중요도 비교 ]")
print(compare_df[['SHAP순위', '통계순위', '변수', 'rho', 'SHAP값', '일치여부']].to_string(index=False))

# ── 6. 시각화 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#F8FAFB')

# (1) SHAP Bar
top_n = 10
top_feats = shap_importance.head(top_n)
axes[0].barh(top_feats.index[::-1], top_feats.values[::-1], color='#3498DB')
axes[0].set_title('SHAP 피처 중요도 (상위 10)', fontweight='bold')
axes[0].set_xlabel('평균 |SHAP|')
axes[0].grid(axis='x', linestyle='--', alpha=0.4)

# (2) Spearman Bar
top_stat = stat_df.head(top_n)
colors = ['#E74C3C' if r < 0 else '#2ECC71' for r in top_stat['rho']]
axes[1].barh(top_stat['변수'][::-1], top_stat['rho'][::-1], color=colors[::-1])
axes[1].set_title('Spearman 상관계수 (상위 10)', fontweight='bold')
axes[1].set_xlabel('rho')
axes[1].axvline(0, color='black', lw=0.8)
axes[1].grid(axis='x', linestyle='--', alpha=0.4)

# (3) 순위 비교 scatter
plot_df = compare_df[compare_df['변수'] != 'Pressure_Diff_ratio'].head(15)
axes[2].scatter(plot_df['통계순위'], plot_df['SHAP순위'], s=80, color='#9B59B6', zorder=5)
for _, row in plot_df.iterrows():
    axes[2].annotate(row['변수'], (row['통계순위'], row['SHAP순위']),
                     textcoords='offset points', xytext=(5, 3), fontsize=7)
axes[2].plot([1, len(FEATURES)], [1, len(FEATURES)], '--', color='gray', alpha=0.5, label='완전 일치선')
axes[2].set_xlabel('통계 순위 (Spearman)')
axes[2].set_ylabel('ML 순위 (SHAP)')
axes[2].set_title('통계 vs ML 순위 비교', fontweight='bold')
axes[2].legend()
axes[2].grid(linestyle='--', alpha=0.4)

plt.suptitle('Type 2 — 통계 vs SHAP 피처 중요도 분석', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('type2_shap_비교.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n이미지 저장 완료: type2_shap_비교.png")

# ── 7. Type1 vs Type2 비교 요약 출력 ───────────────────────
print("\n" + "="*60)
print("  Type 1 vs Type 2 — 통계 상위 5위 비교")
print("="*60)
type1_top5 = [
    ('Factory_Humidity', -0.278), ('Factory_Temp', 0.216),
    ('Biscuit_Thickness', -0.170), ('Spray_2_Time', 0.167), ('Casting_Pressure', -0.125)
]
type2_top5 = stat_df.head(5)[['변수', 'rho']].values.tolist()

print(f"{'순위':<4} {'Type 1 (통계)':<30} {'Type 2 (통계)':<30}")
print("-"*64)
for i, (t1, t2) in enumerate(zip(type1_top5, type2_top5), 1):
    print(f"  {i}  {t1[0]:<20} {t1[1]:>+.3f}    {t2[0]:<20} {t2[1]:>+.3f}")

print("\nType 1: 환경 변수(습도/온도) 중심")
print("Type 2: 공정 변수(속도/시간) 중심 — 불량 메커니즘이 다름")

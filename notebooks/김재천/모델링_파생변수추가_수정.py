#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, f1_score, recall_score, precision_score,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform
SEED = 42
np.random.seed(SEED)
print("✅ 모델링 라이브러리 로드 완료")


# In[2]:


if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 파생변수 추가 버전
type1_df = pd.read_csv('../../data/df_type1_파생변수추가.csv')
type1_df['Pressure_Diff_ratio'] = type1_df['Casting_Pressure'] / type1_df['Cylinder_Pressure']

NEW_DERIVED = [
    'clamp_per_press', 'temp_diff',
    'press_time_ratio', 'spray_cycle_ratio', 'heat_index'    
]

FEATURES = [
    'Velocity_1', 'Velocity_2', 'Velocity_3', 'High_Velocity',
    'Rapid_Rise_Time', 'Biscuit_Thickness', 'Clamping_Force', 'Cycle_Time',
    'Pressure_Rise_Time', 'Casting_Pressure', 'Cylinder_Pressure',
    'Spray_Time', 'Spray_1_Time', 'Spray_2_Time',
    'Melting_Furnace_Temp', 'Air_Pressure', 'Coolant_Temp', 'Coolant_Pressure',
    'Factory_Temp', 'Factory_Humidity',
    'Pressure_Diff_ratio',
    # 신규 파생변수
    'clamp_per_press', 'temp_diff',
    'press_time_ratio', 'spray_cycle_ratio', 'heat_index'   
]

X = type1_df[FEATURES].copy()
y = type1_df['Defect_Status'].copy()

print(f"피처 수: {len(FEATURES)}개 (원본 21 + 신규 파생변수 {len(NEW_DERIVED)}개)")
print(f"타겟 분포 → 양품: {(y==0).sum()} / 불량: {(y==1).sum()} ({y.mean()*100:.1f}%)")


# ## STEP 1. 피처 및 타겟 정의

# ## STEP 2. Train/Test Split

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

pos_weight = (y_train==0).sum() / (y_train==1).sum()

print(f"훈련셋: {X_train.shape[0]}행  |  테스트셋: {X_test.shape[0]}행")
print(f"훈련셋 불량률: {y_train.mean()*100:.1f}%  |  테스트셋 불량률: {y_test.mean()*100:.1f}%")
print(f"scale_pos_weight: {pos_weight:.2f}")


# In[4]:


from sklearn.ensemble import IsolationForest

# 수치형 컬럼만 선택
X_train_num = X_train.select_dtypes(include=['int64', 'float64'])
X_test_num = X_test.select_dtypes(include=['int64', 'float64'])

iso = IsolationForest(contamination=0.01, random_state=SEED)
iso.fit(X_train_num)
train_anomaly = iso.predict(X_train_num)
test_anomaly = iso.predict(X_test_num)

print(f"훈련셋 이상치: {(train_anomaly == -1).sum()}건 ({(train_anomaly == -1).mean()*100:.1f}%)")
print(f"테스트셋 이상치: {(test_anomaly == -1).sum()}건 ({(test_anomaly == -1).mean()*100:.1f}%)")

X_train_iso = X_train_num[train_anomaly == 1]
y_train_iso = y_train[train_anomaly == 1]
print(f"\nIsolation Forest 후 훈련셋: {X_train_iso.shape[0]}행")
print(f"불량률: {y_train_iso.mean()*100:.1f}%")


# ## STEP 3. 베이스라인 모델 비교

# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def find_best_threshold_constrained(y_true, y_prob, target_recall=0.80,
                                    thr_min=0.05, thr_max=0.95, step=0.01):
    thresholds = np.arange(thr_min, thr_max + 1e-9, step)
    best, best_any = None, None
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        r  = recall_score(y_true, y_pred, zero_division=0)
        p  = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        row = (thr, r, p, f1)
        if best_any is None or f1 > best_any[3]: best_any = row
        if r >= target_recall:
            if best is None or f1 > best[3]: best = row
    if best:
        thr,r,p,f1 = best; ok = True
    else:
        thr,r,p,f1 = best_any; ok = False
    return thr, r, p, f1, ok

# XGBoost용 원-핫
X_train_xgb = pd.get_dummies(X_train)
X_test_xgb  = pd.get_dummies(X_test).reindex(columns=X_train_xgb.columns, fill_value=0)

baseline_models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        scale_pos_weight=pos_weight, n_estimators=200,
        random_state=SEED, eval_metric='logloss', verbosity=0),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        scale_pos_weight=pos_weight, random_state=SEED, verbose=-1),
}

baseline_results = []
for name, model in baseline_models.items():
    if name == 'XGBoost':
        model.fit(X_train_xgb, y_train)
        yp = model.predict_proba(X_test_xgb)[:,1]
    else:
        model.fit(X_train, y_train)
        yp = model.predict_proba(X_test)[:,1]

    thr,r,p,f1,ok = find_best_threshold_constrained(y_test, yp)
    auc = roc_auc_score(y_test, yp)
    baseline_results.append({
        '모델': name, '임계값': round(thr,2),
        'Recall': round(r,4), 'Precision': round(p,4),
        'F1-Score': round(f1,4), 'ROC-AUC': round(auc,4),
        'Recall≥0.80': '✅' if ok else '❌'
    })
    print(f"  ✅ {name} 완료 | thr={thr:.2f} | Recall={r:.3f} | F1={f1:.3f}")

result_df = pd.DataFrame(baseline_results).sort_values('F1-Score', ascending=False)
display(result_df)


# In[6]:


from imblearn.over_sampling import SMOTE

# -------------------------------------------
# SMOTE 적용
# -------------------------------------------
smote = SMOTE(random_state=SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"SMOTE 전 → 양품: {(y_train==0).sum()} / 불량: {(y_train==1).sum()}")
print(f"SMOTE 후 → 양품: {(y_train_sm==0).sum()} / 불량: {(y_train_sm==1).sum()}")

# -------------------------------------------
# SMOTE 적용 모델 비교
# -------------------------------------------
smote_results = []
for name, model in baseline_models.items():
    if name == 'XGBoost':
        model.fit(X_train_sm, y_train_sm)
        yp = model.predict_proba(X_test_xgb)[:,1]
    else:
        model.fit(X_train_sm, y_train_sm)
        yp = model.predict_proba(X_test)[:,1]

    thr,r,p,f1,ok = find_best_threshold_constrained(y_test, yp)
    auc = roc_auc_score(y_test, yp)
    smote_results.append({
        '모델': name, '임계값': round(thr,2),
        'Recall': round(r,4), 'Precision': round(p,4),
        'F1-Score': round(f1,4), 'ROC-AUC': round(auc,4),
        'Recall≥0.80': '✅' if ok else '❌'
    })
    print(f"  ✅ {name} 완료 | thr={thr:.2f} | Recall={r:.3f} | F1={f1:.3f}")

print("\n[ SMOTE vs 미적용 비교 ]")
display(pd.DataFrame(smote_results).sort_values('F1-Score', ascending=False))


# In[10]:


# -------------------------------------------
# SMOTE sampling_strategy 비율별 실험
# 원본 불량:양품 ≈ 1:3.6  →  비율을 조금씩 올려가며 최적점 탐색
# -------------------------------------------
from imblearn.over_sampling import SMOTE

strategies = [0.3, 0.5, 0.7, 1.0]   # 불량/양품 비율 목표값
ratio_results = []

for strategy in strategies:
    smote_exp = SMOTE(sampling_strategy=strategy, random_state=SEED)
    X_sm, y_sm = smote_exp.fit_resample(X_train, y_train)
    n_neg, n_pos = (y_sm==0).sum(), (y_sm==1).sum()

    for name, model in baseline_models.items():
        # 모델 재생성 (이전 fit 상태 초기화)
        if name == 'Logistic Regression':
            m = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=SEED))
            ])
        elif name == 'Random Forest':
            m = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=3)
        elif name == 'XGBoost':
            m = xgb.XGBClassifier(n_estimators=200,
                                   random_state=SEED, eval_metric='logloss', verbosity=0, n_jobs=3)
        else:  # LightGBM
            m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                    random_state=SEED, verbose=-1,n_jobs=3)

        if name == 'XGBoost':
            X_sm_xgb = pd.get_dummies(X_sm).reindex(columns=X_train_xgb.columns, fill_value=0)
            m.fit(X_sm_xgb, y_sm)
            yp = m.predict_proba(X_test_xgb)[:,1]
        else:
            m.fit(X_sm, y_sm)
            yp = m.predict_proba(X_test)[:,1]

        thr, r, p, f1, ok = find_best_threshold_constrained(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        ratio_results.append({
            'strategy': strategy,
            f'양품:{n_neg} / 불량:{n_pos}': '',
            '모델': name,
            '임계값': round(thr, 2),
            'Recall': round(r, 4),
            'Precision': round(p, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'Recall≥0.80': '✅' if ok else '❌'
        })

ratio_df = pd.DataFrame(ratio_results).drop(columns=[f'양품:{n_neg} / 불량:{n_pos}'])

print("[ SMOTE sampling_strategy 비율별 비교 ]")
for s in strategies:
    sub = ratio_df[ratio_df['strategy'] == s].drop(columns='strategy')
    n_pos_s = int(round((y_train==0).sum() * s))
    print(f"\n▶ strategy={s}  (불량 목표: ~{n_pos_s}건, 양품:불량 ≈ 1:{s})")
    display(sub.sort_values('F1-Score', ascending=False).reset_index(drop=True))


# ## STEP 4. 하이퍼파라미터 튜닝
# ### SMOTE 사용 버전

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# -------------------------------------------
# SMOTE 사용 버전
# -------------------------------------------

# SMOTE 적용 데이터로 튜닝 (비율 실험 결과 → strategy=0.3 최적)
smote_tuning = SMOTE(sampling_strategy=1, random_state=SEED)
X_train_sm, y_train_sm = smote_tuning.fit_resample(X_train, y_train)
X_train_sm_xgb = pd.get_dummies(X_train_sm).reindex(columns=X_train_xgb.columns, fill_value=0)
print(f"튜닝용 SMOTE 후 → 양품: {(y_train_sm==0).sum()} / 불량: {(y_train_sm==1).sum()}")

# -------------------------------------------
# 0. Logistic Regression (추가됨)
# -------------------------------------------
lr_param = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga'] # l1, l2 모두 지원
}
# 로지스틱 회귀는 스케일링이 필수이므로 파이프라인 사용
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED))
])

lr_search = RandomizedSearchCV(
    lr_pipeline, 
    lr_param, n_iter=20, scoring='f1', cv=5, random_state=SEED, n_jobs=-1 # 전체 코어 사용
)
lr_search.fit(X_train_sm, y_train_sm)
best_lr = lr_search.best_estimator_
print(f"LR 최적: {lr_search.best_params_}")

# -------------------------------------------
# 1. Random Forest
# -------------------------------------------
rf_param = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1), # 단일 코어
    rf_param, n_iter=30, scoring='f1', cv=5, random_state=SEED, n_jobs=-1
)
rf_search.fit(X_train_sm, y_train_sm)
best_rf = rf_search.best_estimator_
print(f"RF 최적: {rf_search.best_params_}")

# -------------------------------------------
# 2. XGBoost
# -------------------------------------------
xgb_param = {
    'n_estimators':     [200, 300, 500],
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=SEED,
                      eval_metric='logloss', verbosity=0, n_jobs=1), # 단일 코어
    xgb_param, n_iter=30, scoring='f1', cv=5, random_state=SEED, n_jobs=-1
)
xgb_search.fit(X_train_sm_xgb, y_train_sm)
best_xgb = xgb_search.best_estimator_
print(f"XGB 최적: {xgb_search.best_params_}")

# -------------------------------------------
# 3. LightGBM
# -------------------------------------------
lgb_param = {
    'n_estimators':  [300, 500, 700],
    'max_depth':     [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves':    [31, 63, 127],
    'subsample':     [0.7, 0.8, 1.0],
}
lgb_search = RandomizedSearchCV(
    lgb.LGBMClassifier(scale_pos_weight=pos_weight, random_state=SEED, verbose=-1, 
                       n_jobs=1, subsample_freq=1), # 단일 코어 및 subsample 적용
    lgb_param, n_iter=30, scoring='f1', cv=5, random_state=SEED, n_jobs=-1
)
lgb_search.fit(X_train_sm, y_train_sm)
best_lgb = lgb_search.best_estimator_
print(f"LGB 최적: {lgb_search.best_params_}")

# -------------------------------------------
# 결과 비교
# -------------------------------------------
tuned_results = []
for name, model, X_te in [
    ('Logistic Regression', best_lr, X_test), # 로지스틱 회귀 추가됨
    ('Random Forest', best_rf, X_test),
    ('XGBoost',       best_xgb, X_test_xgb),
    ('LightGBM',      best_lgb, X_test),
]:
    yp = model.predict_proba(X_te)[:,1]
    thr,r,p,f1,ok = find_best_threshold_constrained(y_test, yp)
    auc = roc_auc_score(y_test, yp)
    tuned_results.append({
        '모델': name, '임계값': round(thr,2),
        'Recall': round(r,4), 'Precision': round(p,4),
        'F1-Score': round(f1,4), 'ROC-AUC': round(auc,4),
        'Recall≥0.80': '✅' if ok else '❌'
    })
    print(f"  ✅ {name} | thr={thr:.2f} | Recall={r:.3f} | F1={f1:.3f}")

display(pd.DataFrame(tuned_results).sort_values('F1-Score', ascending=False))


# ### SMOTE 미사용 버전

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# -------------------------------------------
# SMOTE 사용 X 버전
# -------------------------------------------

# -------------------------------------------
# 0. Logistic Regression
# -------------------------------------------
lr_param = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga'] # l1, l2 모두 지원
}
# 로지스틱 회귀는 스케일링이 필수이므로 파이프라인 사용
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED))
])

lr_search = RandomizedSearchCV(
    lr_pipeline, 
    lr_param, n_iter=20, scoring='f1', cv=5, random_state=SEED, n_jobs=8 # 전체 코어 사용
)
# SMOTE 데이터 대신 원본 X_train, y_train 사용
lr_search.fit(X_train, y_train)
best_lr = lr_search.best_estimator_
print(f"LR 최적: {lr_search.best_params_}")

# -------------------------------------------
# 1. Random Forest
# -------------------------------------------
rf_param = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1), # 단일 코어
    rf_param, n_iter=30, scoring='f1', cv=5, random_state=SEED, n_jobs=2
)
# SMOTE 데이터 대신 원본 X_train, y_train 사용
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print(f"RF 최적: {rf_search.best_params_}")

# -------------------------------------------
# 2. XGBoost
# -------------------------------------------
xgb_param = {
    'n_estimators':     [200, 300, 500],
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=SEED,
                      eval_metric='logloss', verbosity=0, n_jobs=1), # 단일 코어
    xgb_param, n_iter=30, scoring='f1', cv=5, random_state=SEED, n_jobs=2
)
# SMOTE 데이터 대신 원본 X_train_xgb, y_train 사용
xgb_search.fit(X_train_xgb, y_train)
best_xgb = xgb_search.best_estimator_
print(f"XGB 최적: {xgb_search.best_params_}")

# -------------------------------------------
# 3. LightGBM
# -------------------------------------------
lgb_param = {
    'n_estimators':  [300, 500, 700],
    'max_depth':     [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves':    [31, 63, 127],
    'subsample':     [0.7, 0.8, 1.0],
}
lgb_search = RandomizedSearchCV(
    lgb.LGBMClassifier(scale_pos_weight=pos_weight, random_state=SEED, verbose=-1, 
                       n_jobs=1, subsample_freq=1), # 단일 코어 및 subsample 적용
    lgb_param, n_iter=30, scoring='f1', cv=5, random_state=SEED, n_jobs=2
)
# SMOTE 데이터 대신 원본 X_train, y_train 사용
lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_
print(f"LGB 최적: {lgb_search.best_params_}")

# -------------------------------------------
# 결과 비교
# -------------------------------------------
tuned_results = []
for name, model, X_te in [
    ('Logistic Regression', best_lr, X_test), 
    ('Random Forest', best_rf, X_test),
    ('XGBoost',       best_xgb, X_test_xgb),
    ('LightGBM',      best_lgb, X_test),
]:
    yp = model.predict_proba(X_te)[:,1]
    thr,r,p,f1,ok = find_best_threshold_constrained(y_test, yp)
    auc = roc_auc_score(y_test, yp)
    tuned_results.append({
        '모델': name, '임계값': round(thr,2),
        'Recall': round(r,4), 'Precision': round(p,4),
        'F1-Score': round(f1,4), 'ROC-AUC': round(auc,4),
        'Recall≥0.80': '✅' if ok else '❌'
    })
    print(f"  ✅ {name} | thr={thr:.2f} | Recall={r:.3f} | F1={f1:.3f}")

display(pd.DataFrame(tuned_results).sort_values('F1-Score', ascending=False))


# ## STEP 5. 최종 평가-LightGBM

# In[15]:


# LightGBM 최종 모델 (파생변수 추가 버전)
# — 원본 대비 Recall↑(0.843 vs 0.800), F1 소폭 하락(0.523 vs 0.533)
y_prob_final = best_lgb.predict_proba(X_test)[:, 1]
opt_thr_lgbm, _, _, _, _ = find_best_threshold_constrained(y_test, y_prob_final)
y_pred_final = (y_prob_final >= opt_thr_lgbm).astype(int)

print("=" * 55)
print(f"  최종 모델: LightGBM (임계값 {opt_thr_lgbm:.3f})")
print(f"  피처: {len(FEATURES)}개 (원본 21 + 파생변수 {len(FEATURES)-21}개)")
print("=" * 55)
print(classification_report(y_test, y_pred_final,
                            target_names=['양품', '불량'], digits=4))

# 혼동행렬 + ROC
y_prob_xgb_ref = best_xgb.predict_proba(X_test_xgb)[:, 1]
fpr_l, tpr_l, _ = roc_curve(y_test, y_prob_final)
fpr_x, tpr_x, _ = roc_curve(y_test, y_prob_xgb_ref)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#F8FAFB')

cm = confusion_matrix(y_test, y_pred_final)
ConfusionMatrixDisplay(cm, display_labels=['양품', '불량']).plot(
    ax=axes[0], colorbar=False, cmap='Blues')
tn, fp, fn, tp = cm.ravel()
axes[0].set_title(f'혼동 행렬 (LightGBM, thr={opt_thr_lgbm:.3f})', fontweight='bold')
axes[0].text(0, 1, f'FN={fn}\n(불량 누락)', ha='center', va='center',
             fontsize=10, color='red', fontweight='bold')

axes[1].plot(fpr_l, tpr_l, lw=2,
             label=f'LightGBM (AUC={roc_auc_score(y_test, y_prob_final):.4f})')
axes[1].plot(fpr_x, tpr_x, lw=2, linestyle='--',
             label=f'XGBoost (AUC={roc_auc_score(y_test, y_prob_xgb_ref):.4f})')
axes[1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend(); axes[1].grid(linestyle='--', alpha=0.4)

plt.suptitle('최종 모델 평가 — Product_Type 1 (파생변수 추가)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ## STEP 6. SHAP

# In[ ]:


import shap
import warnings
warnings.filterwarnings('ignore')

explainer = shap.TreeExplainer(best_lgb)
shap_values = explainer.shap_values(X_test)
shap_values_array = shap_values[1] if isinstance(shap_values, list) else shap_values

shap.summary_plot(shap_values_array, X_test, plot_type='bar', show=True)
shap.summary_plot(shap_values_array, X_test, show=True)


# ## STEP 7. Isolation Forest + LightGBM 2단계 파이프라인

# In[12]:


from sklearn.ensemble import IsolationForest

# STEP 5에서 최종 모델을 XGBoost로 변경했으므로 LightGBM 임계값 별도 계산
y_prob_lgbm_ref = best_lgb.predict_proba(X_train)[:,1]
opt_thr_lgbm, _, _, _, _ = find_best_threshold_constrained(y_train, best_lgb.predict_proba(X_train)[:,1])
# 테스트셋 기준으로 재계산
opt_thr_lgbm, _, _, _, _ = find_best_threshold_constrained(y_test, best_lgb.predict_proba(X_test)[:,1])

# Stage 1: Isolation Forest (양품 데이터로만 학습)
X_train_normal = X_train[y_train == 0]

iso_pipe = IsolationForest(random_state=SEED)
iso_pipe.fit(X_train_normal)

X_train_if = X_train.copy()
X_test_if  = X_test.copy()

X_train_if['anomaly_score'] = iso_pipe.decision_function(X_train)
X_test_if['anomaly_score']  = iso_pipe.decision_function(X_test)

print("[ Isolation Forest 이상치 점수 통계 ]")
print(f"양품 평균: {X_train_if[y_train==0]['anomaly_score'].mean():.4f}")
print(f"불량 평균: {X_train_if[y_train==1]['anomaly_score'].mean():.4f}")

from imblearn.over_sampling import SMOTE
smote_if = SMOTE(sampling_strategy=0.3, random_state=SEED)
X_train_if_sm, y_train_if_sm = smote_if.fit_resample(X_train_if, y_train)

lgb_if = lgb.LGBMClassifier(**best_lgb.get_params())
lgb_if.fit(X_train_if_sm, y_train_if_sm)

y_prob_if = lgb_if.predict_proba(X_test_if)[:,1]
thr_if, r_if, p_if, f1_if, ok_if = find_best_threshold_constrained(y_test, y_prob_if)
auc_if = roc_auc_score(y_test, y_prob_if)

print("\n[ 단일 LightGBM vs IF 점수 추가 LightGBM 비교 ]")
compare_df = pd.DataFrame([
    {'모델': 'LightGBM (단독)', '임계값': round(opt_thr_lgbm, 2),
     'Recall': round(recall_score(y_test, (best_lgb.predict_proba(X_test)[:,1] >= opt_thr_lgbm).astype(int)), 4),
     'F1-Score': round(f1_score(y_test, (best_lgb.predict_proba(X_test)[:,1] >= opt_thr_lgbm).astype(int)), 4),
     'ROC-AUC': round(roc_auc_score(y_test, best_lgb.predict_proba(X_test)[:,1]), 4), 'Recall≥0.80': '✅'},
    {'모델': 'LightGBM + IF 점수', '임계값': round(thr_if, 2),
     'Recall': round(r_if, 4), 'F1-Score': round(f1_if, 4),
     'ROC-AUC': round(auc_if, 4), 'Recall≥0.80': '✅' if ok_if else '❌'},
])
display(compare_df)


# ## STEP 8. 캐스케이드 파이프라인 (LightGBM → XGBoost)

# In[13]:


# =======================================================
# STEP 8. 캐스케이드 파이프라인
#
# [Stage 1] LightGBM (Recall 0.90 목표)
#   - 넓게 잡아라 → 불량 의심 샘플 플래깅
#
# [Stage 2] 튜닝된 XGBoost
#   - Stage 1이 플래깅한 샘플만 대상
#   - 진짜 불량 vs 과탐지 구분 → Precision 향상
# =======================================================

# -------------------------------------------
# Stage 1: LightGBM - Recall 0.90 목표 임계값 탐색
# -------------------------------------------
y_prob_s1 = best_lgb.predict_proba(X_test)[:,1]
thr_s1, r_s1, p_s1, f1_s1, _ = find_best_threshold_constrained(
    y_test, y_prob_s1, target_recall=0.90)

y_pred_s1_test  = (y_prob_s1 >= thr_s1).astype(int)

# 훈련 데이터도 Stage 1로 플래깅
y_prob_s1_train = best_lgb.predict_proba(X_train)[:,1]
y_pred_s1_train = (y_prob_s1_train >= thr_s1).astype(int)

print("[ Stage 1: LightGBM ]")
print(f"임계값: {thr_s1:.2f} | Recall: {r_s1:.4f} | Precision: {p_s1:.4f} | F1: {f1_s1:.4f}")
print(f"플래깅된 테스트 샘플: {y_pred_s1_test.sum()}건 / {len(y_pred_s1_test)}건")

# -------------------------------------------
# Stage 2: XGBoost - Stage 1 플래깅 샘플로만 재학습
# -------------------------------------------
X_train_s2     = X_train[y_pred_s1_train == 1]
y_train_s2     = y_train[y_pred_s1_train == 1]
X_train_s2_xgb = pd.get_dummies(X_train_s2).reindex(columns=X_train_xgb.columns, fill_value=0)

print(f"\n[ Stage 2: XGBoost 학습 데이터 ]")
print(f"플래깅된 훈련 샘플: {len(X_train_s2)}건 "
      f"(양품 {(y_train_s2==0).sum()} / 불량 {(y_train_s2==1).sum()})")

xgb_s2 = xgb.XGBClassifier(**best_xgb.get_params())
xgb_s2.fit(X_train_s2_xgb, y_train_s2)

# -------------------------------------------
# 최종 예측: Stage 1 플래깅 → Stage 2 분류
# -------------------------------------------
X_test_s1_pos     = X_test[y_pred_s1_test == 1]
X_test_s1_pos_xgb = pd.get_dummies(X_test_s1_pos).reindex(columns=X_train_xgb.columns, fill_value=0)

y_prob_final = np.zeros(len(X_test))
y_prob_final[y_pred_s1_test == 1] = xgb_s2.predict_proba(X_test_s1_pos_xgb)[:,1]

thr_final, r_final, p_final, f1_final, ok_final = find_best_threshold_constrained(y_test, y_prob_final)
auc_final = roc_auc_score(y_test, y_prob_final)

# -------------------------------------------
# 비교 테이블
# -------------------------------------------
print("\n[ 단독 LightGBM vs 캐스케이드 파이프라인 비교 ]")
compare_df = pd.DataFrame([
    {'모델': 'LightGBM (단독)',           '임계값': round(opt_thr_lgbm, 2),
     'Recall': 0.8000, 'Precision': 0.4000, 'F1-Score': 0.5333, 'ROC-AUC': 0.7894, 'Recall≥0.80': '✅'},
    {'모델': 'Stage1 LightGBM (Recall0.90)', '임계값': round(thr_s1, 2),
     'Recall': round(r_s1, 4), 'Precision': round(p_s1, 4), 'F1-Score': round(f1_s1, 4),
     'ROC-AUC': round(roc_auc_score(y_test, y_prob_s1), 4), 'Recall≥0.80': '✅'},
    {'모델': 'LightGBM → XGBoost (캐스케이드)', '임계값': round(thr_final, 2),
     'Recall': round(r_final, 4), 'Precision': round(p_final, 4), 'F1-Score': round(f1_final, 4),
     'ROC-AUC': round(auc_final, 4), 'Recall≥0.80': '✅' if ok_final else '❌'},
])
display(compare_df)


# ## STEP 9. 불량 유형 다중 분류 (Defect_Type)

# In[14]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# 파생변수 추가 버전 CSV 사용
type1_multi = pd.read_csv('../../data/df_type1_파생변수추가.csv')
type1_multi['Pressure_Diff_ratio'] = type1_multi['Casting_Pressure'] / type1_multi['Cylinder_Pressure']

# Other, Stain 제외
type1_multi = type1_multi[~type1_multi['Defect_Type'].isin(['Other', 'Stain'])].reset_index(drop=True)

X_multi = type1_multi[FEATURES].copy()
y_multi = type1_multi['Defect_Type'].copy()

print("[ 클래스 분포 ]")
print(y_multi.value_counts())
print(f"\n전체 샘플: {len(y_multi)}건")

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=SEED)
print(f"\n훈련셋: {X_tr_m.shape[0]}행 / 테스트셋: {X_te_m.shape[0]}행")

lgb_multi = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.01, num_leaves=127,
    subsample=0.7, class_weight='balanced',
    random_state=SEED, verbose=-1
)
lgb_multi.fit(X_tr_m, y_tr_m)
y_pred_multi = lgb_multi.predict(X_te_m)

print("\n[ LightGBM 다중 분류 성능 ]")
print(classification_report(y_te_m, y_pred_multi, digits=4))

labels = sorted(y_multi.unique())
cm_multi = confusion_matrix(y_te_m, y_pred_multi, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8FAFB')
ConfusionMatrixDisplay(cm_multi, display_labels=labels).plot(
    ax=ax, colorbar=False, cmap='Blues', xticks_rotation=30)
ax.set_title('혼동 행렬 - 불량 유형 다중 분류', fontweight='bold')
plt.tight_layout()
plt.show()


# ## STEP 9-1. 불량 유형 다중 분류 — Normal 제외 (불량끼리만)

# In[15]:


from imblearn.over_sampling import SMOTE

# type1_multi는 바로 위 STEP 9에서 이미 로드됨 (파생변수 추가 버전)
type1_defect = type1_multi[type1_multi['Defect_Type'] != 'Normal'].reset_index(drop=True)

X_def = type1_defect[FEATURES].copy()
y_def = type1_defect['Defect_Type'].copy()

print("[ 클래스 분포 (Normal 제외) ]")
print(y_def.value_counts())
print(f"\n전체 샘플: {len(y_def)}건")

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X_def, y_def, test_size=0.2, stratify=y_def, random_state=SEED)
print(f"훈련셋: {X_tr_d.shape[0]}행 / 테스트셋: {X_te_d.shape[0]}행")

# 베이스라인
lgb_def = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.01, num_leaves=63,
    subsample=0.7, class_weight='balanced',
    random_state=SEED, verbose=-1
)
lgb_def.fit(X_tr_d, y_tr_d)
y_pred_def = lgb_def.predict(X_te_d)

print("\n[ 베이스라인 (SMOTE 없음) ]")
print(classification_report(y_te_d, y_pred_def, digits=4))

# SMOTE 적용
smote_def = SMOTE(random_state=SEED, k_neighbors=3)
X_tr_d_sm, y_tr_d_sm = smote_def.fit_resample(X_tr_d, y_tr_d)

print("[ SMOTE 후 클래스 분포 ]")
print(pd.Series(y_tr_d_sm).value_counts())

lgb_def_sm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.01, num_leaves=63,
    subsample=0.7, class_weight='balanced',
    random_state=SEED, verbose=-1
)
lgb_def_sm.fit(X_tr_d_sm, y_tr_d_sm)
y_pred_def_sm = lgb_def_sm.predict(X_te_d)

print("\n[ SMOTE 적용 후 ]")
print(classification_report(y_te_d, y_pred_def_sm, digits=4))

labels_def = sorted(y_def.unique())
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#F8FAFB')

for ax, y_pred_plot, title in zip(
    axes,
    [y_pred_def, y_pred_def_sm],
    ['베이스라인 (SMOTE 없음)', 'SMOTE 적용']
):
    cm_def = confusion_matrix(y_te_d, y_pred_plot, labels=labels_def)
    ConfusionMatrixDisplay(cm_def, display_labels=labels_def).plot(
        ax=ax, colorbar=False, cmap='Blues', xticks_rotation=30)
    ax.set_title(f'혼동 행렬 — {title}', fontweight='bold')

plt.suptitle('불량 유형 다중 분류 (Normal 제외)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ## STEP 9 결론 — 불량 유형 분류 한계 및 방향 전환
# 
# ### 실험 요약
# 
# | 실험 | 조건 | macro F1 | 비고 |
# |------|------|----------|------|
# | STEP 9 | Normal 포함, 5클래스 | 0.338 | Normal만 F1=0.827, 나머지 0.13~0.39 |
# | STEP 9-1 (베이스라인) | Normal 제외, 4클래스 | 0.376 | 4클래스 랜덤 기준(0.25) 대비 소폭 상회 |
# | STEP 9-1 (SMOTE) | Normal 제외 + SMOTE | 0.395 | 최선의 시도, 실용성 없음 |
# 
# ### 실패 원인
# 
# 1. **절대 샘플 수 부족** — Bubble 58건(훈련셋 46건), 테스트셋 12건으로 어떤 모델도 학습 불가
# 2. **유형간 공정 패턴 유사** — Exfoliation/Short_Shot/Deformation 모두 비슷한 공정 조건에서 발생, 피처로 구분되는 신호 없음
# 3. **미수집 변수 영향 가능성** — 불량 유형은 금형 상태, 재료 배치, 작업자 등 공정 수치 외 요인에 의존할 가능성
# 
# ### 방향 전환
# 
# 불량 유형 분류 대신 아래 두 축으로 목표 재설정:
# 
# - **이진 분류 (불량 여부)** — LightGBM, Recall ≥ 0.80 달성 → 실용적 조기 경보 가능
# - **SHAP 기반 유형 간접 해석** — Type 1 vs Type 2 중요 변수 비교로 불량 발생 메커니즘 차이 분석
# 
# > 공정 데이터만으로 불량 유형을 구분하는 것은 현재 데이터셋의 구조적 한계이며,
# > 불량 탐지 자체의 신뢰도(Recall 0.80 이상)를 확보하는 것이 현실적 목표.

# In[ ]:


#with open('model_type1_lgbm.pkl', 'wb') as f:
    #pickle.dump({'model': best_lgb, 'threshold': float(opt_thr_lgbm), 'features': list(X_train.columns)}, f)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------
# 기준값: 양품 데이터 평균
# -------------------------------------------
baseline = X_test[y_test == 0].mean().to_dict()

# -------------------------------------------
# 1. 시나리오 시뮬레이션
# 핵심 변수별 구간 변화 → 불량 확률 변화
# -------------------------------------------
key_vars = ['Factory_Humidity', 'Coolant_Pressure', 'Air_Pressure', 'Factory_Temp']

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#F8FAFB')
axes = axes.flatten()

for i, var in enumerate(key_vars):
    probs = []
    var_range = np.linspace(X_test[var].min(), X_test[var].max(), 50)
    
    for val in var_range:
        sample = pd.DataFrame([baseline])
        sample[var] = val
        prob = best_lgb.predict_proba(sample)[:,1][0]
        probs.append(prob)
    
    axes[i].plot(var_range, probs, color='#E74C3C', lw=2)
    axes[i].axhline(0.4, color='gray', linestyle='--', alpha=0.5, label='임계값 0.40')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('불량 확률')
    axes[i].set_title(f'{var} 변화에 따른 불량 확률', fontweight='bold')
    axes[i].legend()
    axes[i].grid(linestyle='--', alpha=0.4)

plt.suptitle('핵심 변수별 불량 확률 시뮬레이션', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------
# 2. 최적 공정 조건 탐색
# 양품 데이터 중 불량 확률 최저 구간 추출
# -------------------------------------------
normal_data = X_test[y_test == 0].copy()
normal_data['불량확률'] = best_lgb.predict_proba(normal_data)[:,1]
normal_data = normal_data.sort_values('불량확률')

print("=" * 55)
print("  최적 공정 조건 (불량 확률 하위 10%)")
print("=" * 55)
top10 = normal_data.head(int(len(normal_data) * 0.1))
optimal = top10[key_vars + ['불량확률']].describe().loc[['mean', 'min', 'max']]
print(optimal.round(4))

print("\n 최적 조건 요약:")
for var in key_vars:
    print(f"  {var}: {top10[var].mean():.2f} (범위 {top10[var].min():.2f} ~ {top10[var].max():.2f})")


# In[ ]:





# ## STEP 10. 파생변수 추가 전/후 성능 비교

# In[ ]:


# =======================================================
# STEP 10. 원본 LightGBM(21피처) vs 파생변수 추가 LightGBM 비교
# =======================================================
import warnings
warnings.filterwarnings('ignore')

y_prob_lgb_new = best_lgb.predict_proba(X_test)[:, 1]
thr_new, r_new, p_new, f1_new, _ = find_best_threshold_constrained(y_test, y_prob_lgb_new)
auc_new = roc_auc_score(y_test, y_prob_lgb_new)

compare_df = pd.DataFrame([
    {
        '버전': f'원본 (21 피처)',
        '모델': 'LightGBM',
        '임계값': 0.44,
        'Recall': 0.800,
        'Precision': 0.400,
        'F1-Score': 0.533,
        'ROC-AUC': 0.789
    },
    {
        '버전': f'파생변수 추가 ({len(FEATURES)} 피처)',
        '모델': 'LightGBM',
        '임계값': round(thr_new, 2),
        'Recall': round(r_new, 3),
        'Precision': round(p_new, 3),
        'F1-Score': round(f1_new, 3),
        'ROC-AUC': round(auc_new, 3)
    }
])
print("[ 원본 vs 파생변수 추가 LightGBM 성능 비교 ]")
display(compare_df)

print(f"\nF1 변화:      {round(f1_new - 0.533, 3):+.3f}")
print(f"Recall 변화:  {round(r_new  - 0.800, 3):+.3f}")
print(f"AUC 변화:     {round(auc_new - 0.789, 3):+.3f}")

# ── SHAP 순위 — 파생변수 기여도 확인
explainer_cmp = shap.TreeExplainer(best_lgb)
shap_v = explainer_cmp.shap_values(X_test)
shap_arr = shap_v[1] if isinstance(shap_v, list) else shap_v

shap_imp = pd.Series(
    np.abs(shap_arr).mean(axis=0),
    index=FEATURES
).sort_values(ascending=False)

NEW_DERIVED = [f for f in FEATURES if f not in [
    'Velocity_1','Velocity_2','Velocity_3','High_Velocity','Rapid_Rise_Time',
    'Biscuit_Thickness','Clamping_Force','Cycle_Time','Pressure_Rise_Time',
    'Casting_Pressure','Cylinder_Pressure','Spray_Time','Spray_1_Time','Spray_2_Time',
    'Melting_Furnace_Temp','Air_Pressure','Coolant_Temp','Coolant_Pressure',
    'Factory_Temp','Factory_Humidity','Pressure_Diff_ratio'
]]

print("\n[ SHAP 피처 중요도 — 상위 15 ]")
print(f"  {'순위':<4} {'변수':<25} {'SHAP':>8}  비고")
print("  " + "-" * 48)
for i, (feat, val) in enumerate(shap_imp.head(15).items(), 1):
    tag = '🆕 신규' if feat in NEW_DERIVED else ''
    print(f"  {i:<4} {feat:<25} {val:>8.4f}  {tag}")

print("\n[ 신규 파생변수 전체 순위 ]")
all_feats = list(shap_imp.index)
for feat in NEW_DERIVED:
    rank = all_feats.index(feat) + 1
    print(f"  {rank:2d}위  {feat:<25} {shap_imp[feat]:.4f}")


# In[ ]:





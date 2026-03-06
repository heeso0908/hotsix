"""
Isolation Forest 실험 코드 (Product Type 1)
============================================================
전제 조건: 모델링.ipynb의 STEP 1~5 실행 후 아래 변수가 메모리에 있어야 함
  - X_train, X_test, y_train, y_test
  - best_lgb (LightGBM 최적 모델, SMOTE strategy=0.3 튜닝)
  - best_xgb (XGBoost 최적 모델)
  - X_train_xgb, X_test_xgb (XGBoost용 원-핫)
  - opt_thr_lgbm (LightGBM 최적 임계값)
  - find_best_threshold_constrained (함수)
============================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

SEED = 42


# ============================================================
# 실험 배경
# ============================================================
# LightGBM 단독 모델 한계 보완 목적
# - 불량 탐지 Recall 0.80 → 더 높이기 위해 IF 결합 시도
# - 두 가지 방법 실험: IF 점수 피처 추가 / 캐스케이드 파이프라인


# ============================================================
# 방법 1. Isolation Forest 이상치 점수를 피처로 추가
# ============================================================
# 구조:
#   공정 데이터 (21개) + anomaly_score (1개) = 22개 피처 → LightGBM

# IF 학습: 양품 데이터로만
X_train_normal = X_train[y_train == 0]

iso_pipe = IsolationForest(random_state=SEED)
iso_pipe.fit(X_train_normal)

# decision_function: 낮을수록 이상, 높을수록 정상
X_train_if = X_train.copy()
X_test_if  = X_test.copy()

X_train_if['anomaly_score'] = iso_pipe.decision_function(X_train)
X_test_if['anomaly_score']  = iso_pipe.decision_function(X_test)

print("[ IF 이상치 점수 통계 ]")
print(f"훈련셋 전체 mean={X_train_if['anomaly_score'].mean():.4f}")
print(f"양품 평균: {X_train_if[y_train==0]['anomaly_score'].mean():.4f}")
print(f"불량 평균: {X_train_if[y_train==1]['anomaly_score'].mean():.4f}")

# SMOTE 적용 (strategy=0.3)
smote_if = SMOTE(sampling_strategy=0.3, random_state=SEED)
X_train_if_sm, y_train_if_sm = smote_if.fit_resample(X_train_if, y_train)

# LightGBM 재학습 (STEP 4 최적 파라미터 사용)
lgb_if = lgb.LGBMClassifier(**best_lgb.get_params())
lgb_if.fit(X_train_if_sm, y_train_if_sm)

y_prob_if = lgb_if.predict_proba(X_test_if)[:,1]
thr_if, r_if, p_if, f1_if, ok_if = find_best_threshold_constrained(y_test, y_prob_if)
auc_if = roc_auc_score(y_test, y_prob_if)

print("\n[ 방법 1 결과: 단독 LightGBM vs LightGBM + IF 점수 ]")
compare_m1 = pd.DataFrame([
    {'모델': 'LightGBM (단독)',     '임계값': round(opt_thr_lgbm, 2),
     'Recall': 0.8000, 'Precision': 0.4000, 'F1': 0.5333, 'ROC-AUC': 0.7894},
    {'모델': 'LightGBM + IF 점수', '임계값': round(thr_if, 2),
     'Recall': round(r_if, 4), 'Precision': round(p_if, 4),
     'F1': round(f1_if, 4), 'ROC-AUC': round(auc_if, 4)},
])
print(compare_m1.to_string(index=False))


# ============================================================
# 방법 2. 캐스케이드 파이프라인 (LightGBM → XGBoost)
# ============================================================
# 구조:
#   전체 데이터
#       ↓
#   [Stage 1: LightGBM - Recall 0.90 목표]
#   불량 의심 샘플 플래깅
#       ↓
#   [Stage 2: 튜닝된 XGBoost]
#   진짜 불량 vs 과탐지 구분 → Precision 향상

# Stage 1: Recall 0.90 목표 임계값
y_prob_s1 = best_lgb.predict_proba(X_test)[:,1]
thr_s1, r_s1, p_s1, f1_s1, _ = find_best_threshold_constrained(
    y_test, y_prob_s1, target_recall=0.90)

y_pred_s1_test  = (y_prob_s1 >= thr_s1).astype(int)
y_prob_s1_train = best_lgb.predict_proba(X_train)[:,1]
y_pred_s1_train = (y_prob_s1_train >= thr_s1).astype(int)

print("\n[ Stage 1: LightGBM ]")
print(f"임계값: {thr_s1:.2f} | Recall: {r_s1:.4f} | Precision: {p_s1:.4f} | F1: {f1_s1:.4f}")
print(f"플래깅된 테스트 샘플: {y_pred_s1_test.sum()}건 / {len(y_pred_s1_test)}건")

# Stage 2: 플래깅 샘플만으로 XGBoost 재학습
X_train_s2     = X_train[y_pred_s1_train == 1]
y_train_s2     = y_train[y_pred_s1_train == 1]
X_train_s2_xgb = pd.get_dummies(X_train_s2).reindex(columns=X_train_xgb.columns, fill_value=0)

print(f"\n[ Stage 2: XGBoost 학습 데이터 ]")
print(f"플래깅된 훈련 샘플: {len(X_train_s2)}건 "
      f"(양품 {(y_train_s2==0).sum()} / 불량 {(y_train_s2==1).sum()})")

xgb_s2 = xgb.XGBClassifier(**best_xgb.get_params())
xgb_s2.fit(X_train_s2_xgb, y_train_s2)

# 최종 예측
X_test_s1_pos     = X_test[y_pred_s1_test == 1]
X_test_s1_pos_xgb = pd.get_dummies(X_test_s1_pos).reindex(columns=X_train_xgb.columns, fill_value=0)

y_prob_final = np.zeros(len(X_test))
y_prob_final[y_pred_s1_test == 1] = xgb_s2.predict_proba(X_test_s1_pos_xgb)[:,1]

thr_final, r_final, p_final, f1_final, ok_final = find_best_threshold_constrained(y_test, y_prob_final)
auc_final = roc_auc_score(y_test, y_prob_final)

print("\n[ 방법 2 결과: 단독 LightGBM vs 캐스케이드 파이프라인 ]")
compare_m2 = pd.DataFrame([
    {'모델': 'LightGBM (단독)',               '임계값': round(opt_thr_lgbm, 2),
     'Recall': 0.8000, 'Precision': 0.4000, 'F1': 0.5333, 'ROC-AUC': 0.7894},
    {'모델': 'Stage1 LightGBM (Recall0.90)', '임계값': round(thr_s1, 2),
     'Recall': round(r_s1, 4), 'Precision': round(p_s1, 4),
     'F1': round(f1_s1, 4), 'ROC-AUC': round(roc_auc_score(y_test, y_prob_s1), 4)},
    {'모델': 'LightGBM → XGBoost (캐스케이드)',  '임계값': round(thr_final, 2),
     'Recall': round(r_final, 4), 'Precision': round(p_final, 4),
     'F1': round(f1_final, 4), 'ROC-AUC': round(auc_final, 4)},
])
print(compare_m2.to_string(index=False))

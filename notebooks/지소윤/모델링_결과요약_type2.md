# Product Type 2 불량 예측 모델링 결과 요약

## 1. 데이터 개요

- 전체 샘플: 1,964건 (양품 1,467 / 불량 497, 불량률 25.3%)
- 피처 수: 21개 (수치형)
- 타겟: Defect_Status (0=양품, 1=불량)
- 제거 피처: Defect_Status(타겟 누수), Defect_Type(타겟 직결 누수)

---

## 2. 모델 선택 과정

### 베이스라인 (4개 모델 비교)

| 모델 | Recall | F1 | Recall>=0.80 |
|------|--------|----|--------------|
| Random Forest | 0.869 | 0.497 | O |
| Logistic Regression | 0.838 | 0.481 | O |
| XGBoost | 0.556 | 0.470 | X |
| LightGBM | 0.606 | 0.458 | X |

### SMOTE sampling_strategy 비율 실험

| strategy | 불량 수 | LR F1 | RF F1 | XGB F1 | LGB F1 |
|----------|---------|-------|-------|--------|--------|
| 0.5 | 586건 | 0.494 (O) | 0.514 (O) | 0.460 (X) | 0.448 (X) |
| 0.7 | 821건 | 0.503 (O) | 0.488 (O) | 0.476 (X) | 0.456 (X) |
| 1.0 | 1173건 | 0.502 (O) | 0.494 (O) | 0.464 (X) | 0.444 (X) |

- XGBoost, LightGBM은 scale_pos_weight 내부 보정과 SMOTE 이중 적용으로 성능 저하 (Type 1과 동일 패턴)

### 하이퍼파라미터 튜닝 결과

| 모델 | 임계값 | Recall | F1 | Recall>=0.80 |
|------|--------|--------|----|--------------|
| LightGBM | 0.35 | 0.818 | **0.526** | O |
| Random Forest | 0.33 | 0.848 | 0.519 | O |
| XGBoost | 0.41 | 0.879 | 0.509 | O |

**최종 모델: LightGBM** (F1 최고, Recall 0.80 충족)

#### LightGBM 최적 파라미터
- n_estimators: 300
- num_leaves: 127
- max_depth: 7
- learning_rate: 0.01
- subsample: 0.7
- scale_pos_weight: 2.95

---

## 3. 최종 모델 성능 (LightGBM, 임계값 0.35)

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|----|
| 양품 | 0.902 | 0.565 | 0.695 |
| 불량 | 0.388 | 0.818 | 0.526 |
| 전체 정확도 | | | 0.629 |

- 실제 불량 99건 중 81건 탐지, 18건 누락
- ROC-AUC: 0.7335

---

## 4. SHAP 피처 중요도

| 순위 | 변수 | SHAP값 | 유형 |
|------|------|--------|------|
| 1 | **High_Velocity** | 0.820 | 공정 (압도적 1위) |
| 2 | Factory_Humidity | 0.232 | 환경 |
| 3 | Coolant_Pressure | 0.163 | 공정 |
| 4 | Melting_Furnace_Temp | 0.144 | 공정 |
| 5 | Air_Pressure | 0.133 | 환경 |
| 6 | Biscuit_Thickness | 0.125 | 공정 |
| 7 | Pressure_Rise_Time | 0.091 | 공정 |
| 8 | Factory_Temp | 0.091 | 환경 |

- **공정 변수(High_Velocity)가 압도적 1위** — Type 1(환경 주도)과 반대 양상
- Spray_2_Time, Spray_1_Time은 통계 상위지만 SHAP 0 (다른 변수와 혼재)

---

## 5. 최적 공정 조건 (불량 확률 하위 10% 기준)

| 변수 | 권장 평균 | 권장 범위 | 현재 전체 평균 |
|------|---------|---------|------------|
| High_Velocity | **2.705** | 2.678 ~ 2.730 | 2.576 |
| Factory_Humidity | **51.1%** | 46.0 ~ 57.0% | 61.3% |
| Coolant_Pressure | **2.699** | 2.680 ~ 2.720 | 2.691 |
| Melting_Furnace_Temp | **653.7°C** | 635.3 ~ 672.1°C | 655.7°C |

이 조건 유지 시 불량 확률 평균 **3.84%** (현재 25.3% 대비 1/7 수준)

### 조건별 해석
- **High_Velocity**: 현재 평균(2.576)보다 높게(2.705) 유지해야 불량 감소 — 속도가 너무 낮으면 불량 증가
- **Factory_Humidity**: 현재 평균(61.3%)보다 낮게(51.1%) 유지가 핵심 — 습도가 높을수록 불량 증가
- **Coolant_Pressure**: 허용 오차 ±0.02로 매우 좁은 범위 유지 필요
- **Melting_Furnace_Temp**: 전체 평균과 유사한 수준 유지 (655°C 전후)

---

## 6. 통계 분석 vs ML(SHAP) 피처 중요도 비교

### 순위 비교표

| SHAP순위 | 통계순위 | 변수 | rho | SHAP값 | 일치여부 |
|---------|---------|------|-----|--------|---------|
| 1 | 1 | High_Velocity | -0.233 | 0.820 | ✅ 완전 일치 |
| 2 | 9 | Factory_Humidity | +0.123 | 0.232 | ❌ 불일치 |
| 3 | 6 | Coolant_Pressure | +0.158 | 0.163 | ⚠️ 부분 일치 |
| 4 | 14 | Melting_Furnace_Temp | -0.083 | 0.144 | ❌ 불일치 |
| 5 | 19 | Air_Pressure | +0.017 | 0.133 | ❌ 불일치 |
| 7 | 7 | Pressure_Rise_Time | -0.145 | 0.091 | ✅ 일치 |
| 8 | 2 | Factory_Temp | -0.200 | 0.091 | ❌ 불일치 |
| 19 | 5 | Spray_Time | +0.165 | 0.016 | ❌ 불일치 |
| 20 | 3 | Spray_2_Time | -0.188 | 0.000 | ❌ 불일치 |

### 주요 인사이트

**High_Velocity — 두 분석 모두 압도적 1위**
- 통계: rho=-0.233 (절대값 기준 1위)
- ML SHAP: 0.82로 2위(0.23) 대비 3.5배
- 가장 신뢰도 높은 핵심 변수

**Factory_Humidity — ML은 2위, 통계는 9위**
- 단순 상관은 약하지만 비선형 상호작용 존재
- Type 1에서도 동일한 패턴 (Coolant_Pressure) → ML이 추가 발견한 변수

**Factory_Temp — 통계는 2위, SHAP은 8위**
- 단독 상관은 높지만 High_Velocity 등과 혼재 가능성
- 통계가 과대평가한 변수

**Spray_2_Time, Spray_Time — 통계 상위, SHAP 최하위(0)**
- 다른 핵심 변수들과 공분산 관계
- Type 1에서도 동일하게 통계만 중요하고 ML에서 걸러짐 (두 타입 공통 패턴)

### 종합 결론

| 구분 | 변수 |
|------|------|
| 두 분석 모두 중요 (최우선 관리) | High_Velocity, Pressure_Rise_Time |
| ML만 발견 (비선형 패턴) | Factory_Humidity, Coolant_Pressure, Melting_Furnace_Temp, Air_Pressure |
| 통계만 중요 (ML에서 걸러짐) | Factory_Temp, Spray_2_Time, Spray_Time, Clamping_Force |

---

## 7. Type 1 vs Type 2 비교

### 통계 상위 5위

| 순위 | Type 1 | rho | Type 2 | rho |
|------|--------|-----|--------|-----|
| 1 | Factory_Humidity | -0.278 | **High_Velocity** | -0.233 |
| 2 | Factory_Temp | +0.216 | Factory_Temp | -0.200 |
| 3 | Biscuit_Thickness | -0.170 | Spray_2_Time | -0.188 |
| 4 | Spray_2_Time | +0.167 | Clamping_Force | +0.166 |
| 5 | Casting_Pressure | -0.125 | Spray_Time | +0.165 |

### SHAP 상위 3위

| 순위 | Type 1 | SHAP값 | Type 2 | SHAP값 |
|------|--------|--------|--------|--------|
| 1 | Factory_Humidity | ~0.87 | High_Velocity | 0.82 |
| 2 | Coolant_Pressure | ~0.29 | Factory_Humidity | 0.23 |
| 3 | Factory_Temp | ~0.20 | Coolant_Pressure | 0.16 |

### 최적 공정 조건 비교

| 변수 | Type 1 권장값 | Type 2 권장값 |
|------|------------|------------|
| Factory_Humidity | 57.7 ~ 70.6% | **46.0 ~ 57.0%** (더 낮게) |
| Coolant_Pressure | 2.59 ~ 2.65 | 2.680 ~ 2.720 |
| Factory_Temp | 31.6 ~ 35.1°C | — (SHAP 하위) |
| High_Velocity | — (SHAP 하위) | 2.678 ~ 2.730 |
| 목표 불량 확률 | **2.56%** | **3.84%** |

### 핵심 차이

| 항목 | Type 1 | Type 2 |
|------|--------|--------|
| 불량 원인 주도 | 공장 환경 (습도/온도) | 공정 파라미터 (사출 속도) |
| Factory_Humidity 역할 | SHAP 1위, 필수 관리 | SHAP 2위, 보조적 |
| High_Velocity 역할 | 하위권 | SHAP 1위, 필수 관리 |
| 관리 핵심 포인트 | 항온항습 설비 | 사출 속도 안정화 |

> **결론**: 불량 유형(Type 1 vs Type 2)에 따라 불량 발생 메커니즘이 다르다.
> 타입별로 관리해야 할 공정 변수가 다르므로, 타입 분류 후 타입별 알람 기준을 차별화해야 한다.

---

## 8. 비즈니스 시사점

1. **사출 속도(High_Velocity) 안정 제어** — 현재 평균(2.576)보다 높게(2.7 이상) 유지 시 불량 급감
2. **습도 관리는 Type 1보다 낮은 기준 적용** — Type 2는 46~57% 범위가 최적 (Type 1은 58~71%)
3. **타입별 알람 기준 차별화** 권장
   - Type 1: 공장 습도 이상 → 즉시 경보
   - Type 2: 사출 속도 이상 → 즉시 경보
4. Coolant_Pressure는 두 타입 모두 SHAP 상위권 → 공통 정밀 관리 변수
5. 최적 조건 달성 시 불량률 25.3% → 3.84% (약 1/7 수준) 감소 기대

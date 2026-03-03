# 다이캐스팅 공정 품질 예측 — 작업 로그
> **날짜**: 2026-03-03  
> **분석 대상**: DieCasting_Quality_Raw_Data.csv (7,535행 × 57열)  
> **담당**: 지소윤 (EDA / 전처리 / 모델링 파이프라인)

---

## 1. 데이터 기본 현황

| 항목 | 내용 |
|---|---|
| 전체 행 수 | 7,535행 |
| 변수 수 | 57개 (Process 17 / Sensor 14 / Defects 26) |
| Product_Type | 1 (55.8%) / 2 (44.2%) |
| 전체 불량률 | 22.4% |
| Type 1 불량률 | 17.6% |
| Type 2 불량률 | 28.6% |

---

## 2. 주요 전처리 결정 사항

### 2-1. 중복 데이터 처리
- **발견**: id 제외 기준 완전 중복 행 2,918건 (전체의 38.7%)
- **판단**: Cavity 정보가 이미 `_1`, `_2` 컬럼으로 하나의 행에 통합되어 있어 완전 중복 행은 데이터 수집 오기로 판단
- **처리**: `drop_duplicates(keep='first')` → 첫 번째 행 유지
- **결과**: 7,535행 → 4,617행 (전체 기준) / Type 1: 4,207 → 2,653행

### 2-2. 이상치 처리 합의 (260303 정규 회의)

| 항목 | 처리 방향 | 합의 수준 |
|---|---|---|
| Velocity=0 (4건) | 삭제 (물리적으로 사출 불가) | ✅ 확정 |
| 극단값 절사 기준 | IQR / 1% / 0.5% 실험 후 비교 | ✅ 확정 |
| Cycle_Time 극단값 | 공정 지연 신호로 유지 (Capping 제외) | ✅ 확정 |
| IQR vs 시그마 | 전처리=IQR, 관리선=6시그마 | ✅ 확정 |
| Velocity_2 이상치 | type1으로 분리 시 튀는 값이 되진 않음  | type2 데이터 확인 |
| 극단값 절사 비교 지표 | SHAP 등 모델 기반 > 상관계수 | 💡 참고 의견 |

### 2-3. IQR=0 변수 사전 제거 (버그 수정)
- **문제**: `Rapid_Rise_Time`, `Spray_2_Time`이 IQR=0 임에도 Capping 대상에 포함되어 Capping 후 상수로 변환됨 → 상관계수 NaN 발생
- **원인**: 분산=0 확인 로직이 Capping 이후에 있어 감지 실패
- **수정**: Capping 전에 `IQR == 0` 조건으로 사전 감지 → cap_cols에서 제외
- **결과**: NaN 상관계수 없음, FEATURES에서 자동 제외

### 2-4. 다중공선성 처리
- `Casting_Pressure` / `Cylinder_Pressure` / `Pressure_Diff` 상관계수 0.99
- **처리**: `Pressure_Diff`(파생변수)만 유지, 원본 두 변수 제거
- **근거**: 두 압력의 차이가 실질적 정보를 담고 있음

### 2-5. SMOTE 미적용 결정
| 구분 | Recall | F1 | AUC |
|---|---|---|---|
| SMOTE 적용 | 0.855 | 0.554 | 0.812 |
| SMOTE 미적용 | 0.843 | **0.657** | **0.883** |

- 불량률 22.4%로 심각한 불균형 아님
- SMOTE 적용 시 합성 샘플 노이즈로 Precision 과도하게 낮아짐
- **결론**: `class_weight='balanced'` / `scale_pos_weight`으로 대응

---

## 3. 상관분석 결과

### 3-1. 전체 데이터 (중복 제거 후)
- 최고 상관계수: **0.082** (Factory_Humidity)
- 중복 제거 전(0.146)보다 하락 → 중복 데이터가 상관계수를 인위적으로 부풀렸던 것

### 3-2. Product_Type 1만 분리 후
- 최고 상관계수: **0.303** (Factory_Humidity) → 전체 대비 약 4배
- Type 2 혼합으로 인한 패턴 희석이 확인됨

| 순위 | 변수 | 상관계수 | 방향 |
|---|---|---|---|
| 1 | Factory_Humidity | -0.303 | 습도 높을수록 불량 감소 |
| 2 | Factory_Temp | +0.248 | 온도 높을수록 불량 증가 |
| 3 | Spray_Time | -0.179 | 분사 길수록 불량 감소 |
| 4 | Biscuit_Thickness | -0.164 | 두꺼울수록 불량 감소 |
| 5 | Pressure_Diff | -0.164 | 압력 차이 클수록 불량 감소 |

---

## 4. 모델링 결과

### 4-1. 전체 데이터 (중복 제거 후)

| 모델 | Recall | F1 | AUC | Recall≥0.80 |
|---|---|---|---|---|
| **Random Forest** | 0.820 | **0.739** | **0.922** | ✅ |
| LightGBM 베이스라인 | 0.805 | 0.730 | 0.916 | ✅ |
| LightGBM 튜닝 | 0.843 | 0.657 | 0.883 | ✅ |
| XGBoost 베이스라인 | 0.802 | 0.709 | 0.907 | ✅ |

- **Random Forest 베이스라인이 전체 최고 균형 성능**
- 튜닝 모델이 베이스라인보다 F1 낮음 → Recall 최우선 튜닝으로 Precision 희생

### 4-2. Product_Type 1 분리 모델

| 모델 | Recall | F1 | AUC | Recall≥0.80 |
|---|---|---|---|---|
| LightGBM 튜닝 | 0.826 | **0.518** | **0.784** | ✅ |
| Logistic Regression | 0.826 | 0.519 | 0.775 | ✅ |
| XGBoost 튜닝 | 0.843 | 0.512 | 0.782 | ✅ |
| Random Forest | 0.809 | 0.488 | 0.752 | ✅ |

- 전체 대비 F1 크게 하락 (0.739 → 0.518) — 데이터 2,653행으로 감소 영향
- Logistic Regression이 복잡한 트리 모델과 거의 동일 성능 → **Type 1 불량 패턴이 비교적 선형적**임을 시사

### 4-3. 최종 피처 (Type 1 기준, 19개)
```
Velocity_1, Velocity_2, Velocity_3, High_Velocity,
Biscuit_Thickness, Clamping_Force, Cycle_Time, Pressure_Rise_Time,
Spray_Time, Spray_1_Time,
Melting_Furnace_Temp, Air_Pressure, Coolant_Temp, Coolant_Pressure,
Factory_Temp, Factory_Humidity,
Velocity_Avg, Pressure_Diff, Coolant_Temp_Range
```
- 제거: `Rapid_Rise_Time`, `Spray_2_Time` (IQR=0, 상수값)
- 제거: `Casting_Pressure`, `Cylinder_Pressure` (다중공선성 0.99)

---

## 5. SHAP 분석 결과 (Product_Type 1)

| 순위 | 변수 | SHAP | 상관분석 순위 |
|---|---|---|---|
| 1 | **Factory_Humidity** | 0.7239 | 1위 ✅ 일치 |
| 2 | **Coolant_Pressure** | 0.3254 | 15위 → 급상승 |
| 3 | **Spray_Time** | 0.1354 | 3위 ✅ 일치 |
| 4 | **Factory_Temp** | 0.1306 | 2위 ✅ 일치 |
| 5 | **High_Velocity** | 0.1296 | 13위 → 급상승 |

- **Factory_Humidity SHAP 0.724로 압도적 1위** — Type 1 불량의 핵심 변수
- `Coolant_Pressure`, `High_Velocity`는 단독 선형 관계는 약하지만 다른 변수와 조합 시 강한 예측력

---

## 6. Velocity_2 이상치 심층 분석

| 구분 | 데이터 | Velocity_1 IQR 이상치 | Velocity_2 IQR 이상치 |
|---|---|---|---|
| 전체 (혼합) | 4,617행 | 94건 (2.0%) | 94건 (2.0%) |
| Type 1만 | 2,653행 | **139건 (5.24%)** ↑ | **103건 (3.88%)** ↓ |

- **전체 94건이 거의 전부 Type 1에서 발생** → Type 2는 Velocity_2 안정적
- Type 1 분리 후 Velocity_1(139건) vs Velocity_2(103건) — 비슷한 수준으로 특별히 튀는 변수 아님
- **전체에서 Velocity_2가 유독 많아 보였던 이유**: Type 2의 Velocity_2 이상치가 섞여있었기 때문
- **결론**: Velocity_2 이상치는 **Type 2에서 더 집중적으로 조사 필요**, Type 1에서는 정상 수준

---

## 7. 통계 vs 모델 설명력 차이

- Pearson 상관계수가 낮아도 모델 AUC가 높을 수 있음
- 이유: 상관계수는 선형 관계만 측정, 모델은 비선형/상호작용 패턴까지 학습
- 이번 프로젝트: 전체 최고 Pearson 0.082 → LightGBM AUC 0.883
- **SHAP이 상관분석보다 신뢰할 수 있는 변수 중요도 지표**

---

## 8. 산출물 목록

| 파일 | 설명 |
|---|---|
| `eda_pipeline_v4.ipynb` | 전체 데이터 EDA + 모델링 파이프라인 |
| `eda_pipeline_type1_v3.ipynb` | Product_Type 1 전용 분석 파이프라인 (최종) |
| `DieCasting_Preprocessed.csv` | 전체 전처리 완료 데이터 |
| `DieCasting_Preprocessed_Type1.csv` | Type 1 전처리 완료 데이터 |
| `model_final_lgbm.pkl` | 전체 데이터 LightGBM 최종 모델 |
| `model_type1_lgbm.pkl` | Type 1 LightGBM 최종 모델 |
| `README.md` | 팀 레포지토리 문서 |

---

## 9. 미결 / 후속 과제

| 항목 | 내용 |
|---|---|
| ⏳ Velocity_2 추가 조사 | Type 1 고유 현상 여부 확인, 불량 구간 vs 정상 구간 비교 |
| ⏳ Product_Type 2 분석 | Type 1과 비교 |
| 💡 Mutual Information 추가 | Pearson 보완, 비선형 변수 관계 탐색 |
| 💡 Product_Type 2 SHAP 비교 | Type 1과 핵심 변수 차이 확인 |

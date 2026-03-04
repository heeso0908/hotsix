# 다이캐스팅 공정 품질 예측 — 작업 로그
> **분석 대상**: DieCasting_Quality_Raw_Data.csv  
> **담당**: 지소윤 (EDA / 전처리 / 통계 분석 / 모델링 파이프라인)

---

## 2026-03-03

> **분석 대상**: DieCasting_Quality_Raw_Data.csv (7,535행 × 57열)

### 1. 데이터 기본 현황

| 항목 | 내용 |
|---|---|
| 전체 행 수 | 7,535행 |
| 변수 수 | 57개 (Process 17 / Sensor 14 / Defects 26) |
| Product_Type | 1 (55.8%) / 2 (44.2%) |
| 전체 불량률 | 22.4% |
| Type 1 불량률 | 17.6% |
| Type 2 불량률 | 28.6% |

### 2. 주요 전처리 결정 사항

#### 2-1. 중복 데이터 처리
- **발견**: id 제외 기준 완전 중복 행 2,918건 (전체의 38.7%)
- **판단**: Cavity 정보가 이미 `_1`, `_2` 컬럼으로 하나의 행에 통합되어 있어 완전 중복 행은 데이터 수집 오기로 판단
- **처리**: `drop_duplicates(keep='first')` → 첫 번째 행 유지
- **결과**: 7,535행 → 4,617행 (전체 기준) / Type 1: 4,207 → 2,653행

#### 2-2. 이상치 처리 합의 (260303 정규 회의)

| 항목 | 처리 방향 | 합의 수준 |
|---|---|---|
| Velocity=0 (4건) | 삭제 (물리적으로 사출 불가) | ✅ 확정 |
| 극단값 절사 기준 | IQR / 1% / 0.5% 실험 후 비교 | ✅ 확정 |
| Cycle_Time 극단값 | 공정 지연 신호로 유지 (Capping 제외) | ✅ 확정 |
| IQR vs 시그마 | 전처리=IQR, 관리선=6시그마 | ✅ 확정 |
| Velocity_2 이상치 | type1으로 분리 시 튀는 값이 되진 않음 | type2 데이터 확인 |
| 극단값 절사 비교 지표 | SHAP 등 모델 기반 > 상관계수 | 💡 참고 의견 |

#### 2-3. IQR=0 변수 사전 제거 (버그 수정)
- **문제**: `Rapid_Rise_Time`, `Spray_2_Time`이 IQR=0 임에도 Capping 대상에 포함되어 Capping 후 상수로 변환됨 → 상관계수 NaN 발생
- **수정**: Capping 전에 `IQR == 0` 조건으로 사전 감지 → cap_cols에서 제외
- **결과**: NaN 상관계수 없음, FEATURES에서 자동 제외

#### 2-4. 다중공선성 처리
- `Casting_Pressure` / `Cylinder_Pressure` / `Pressure_Diff` 상관계수 0.99
- **처리**: `Pressure_Diff`(파생변수)만 유지, 원본 두 변수 제거
- **근거**: 두 압력의 차이가 실질적 정보를 담고 있음

#### 2-5. SMOTE 미적용 결정
| 구분 | Recall | F1 | AUC |
|---|---|---|---|
| SMOTE 적용 | 0.855 | 0.554 | 0.812 |
| SMOTE 미적용 | 0.843 | **0.657** | **0.883** |

- **결론**: `class_weight='balanced'` / `scale_pos_weight`으로 대응

### 3. 상관분석 결과 (피어슨)

| 순위 | 변수 | 상관계수 | 방향 |
|---|---|---|---|
| 1 | Factory_Humidity | -0.303 | 습도 높을수록 불량 감소 |
| 2 | Factory_Temp | +0.248 | 온도 높을수록 불량 증가 |
| 3 | Spray_Time | -0.179 | 분사 길수록 불량 감소 |
| 4 | Biscuit_Thickness | -0.164 | 두꺼울수록 불량 감소 |
| 5 | Pressure_Diff | -0.164 | 압력 차이 클수록 불량 감소 |

### 4. 모델링 결과

#### 4-1. 전체 데이터

| 모델 | Recall | F1 | AUC | Recall≥0.80 |
|---|---|---|---|---|
| **Random Forest** | 0.820 | **0.739** | **0.922** | ✅ |
| LightGBM 베이스라인 | 0.805 | 0.730 | 0.916 | ✅ |
| LightGBM 튜닝 | 0.843 | 0.657 | 0.883 | ✅ |
| XGBoost 베이스라인 | 0.802 | 0.709 | 0.907 | ✅ |

#### 4-2. Product_Type 1 분리 모델

| 모델 | Recall | F1 | AUC | Recall≥0.80 |
|---|---|---|---|---|
| LightGBM 튜닝 | 0.826 | **0.518** | **0.784** | ✅ |
| Logistic Regression | 0.826 | 0.519 | 0.775 | ✅ |
| XGBoost 튜닝 | 0.843 | 0.512 | 0.782 | ✅ |
| Random Forest | 0.809 | 0.488 | 0.752 | ✅ |

- Logistic Regression ≈ 트리 모델 → **Type 1 불량 패턴이 비교적 선형적**임을 시사

### 5. SHAP 분석 결과 (Product_Type 1)

| 순위 | 변수 | SHAP | 상관분석 순위 |
|---|---|---|---|
| 1 | **Factory_Humidity** | 0.7239 | 1위 ✅ 일치 |
| 2 | **Coolant_Pressure** | 0.3254 | 15위 → 급상승 |
| 3 | **Spray_Time** | 0.1354 | 3위 ✅ 일치 |
| 4 | **Factory_Temp** | 0.1306 | 2위 ✅ 일치 |
| 5 | **High_Velocity** | 0.1296 | 13위 → 급상승 |

### 6. Velocity_2 이상치 분석
- 전체 94건이 거의 전부 Type 1에서 발생 → Type 2는 Velocity_2 안정적
- **결론**: Velocity_2 이상치는 **Type 2에서 더 집중적으로 조사 필요**

---

## 2026-03-04

> **분석 대상**: df_clean_Type1.csv (2,651행 × 31열)

### 1. 오늘 작업 요약

| 작업 | 상태 |
|---|---|
| 피어슨 → 스피어만 상관분석 전환 | ✅ 완료 |
| 신규 변수 발굴 및 추가 | ✅ 완료 |
| 파생변수 생성 (Velocity, Pressure) | ✅ 완료 |
| 맨휘트니 재수행 (전체 변수) | ✅ 완료 |
| 크루스칼-왈리스 재수행 | ✅ 완료 |
| 효과크기 계산 (r, ε²) | ✅ 완료 |
| 던 테스트 (사후검정) 추가 | ✅ 완료 |
| 최종 변수 20개 확정 | ✅ 완료 |
| 통계분석_결과_v2.md 작성 | ✅ 완료 |
| Git PR #7 머지 완료 | ✅ 완료 |

### 2. 분석 방법 전환: 피어슨 → 스피어만

#### 전환 이유
- 정규성 검정(Shapiro-Wilk) 결과 전 변수 비정규분포 확인
- 피어슨은 정규분포 + 선형관계 가정 → 비모수 환경에서 부적절
- 스피어만은 순위 기반으로 비선형 관계도 포착 가능

#### 결과 비교
| 변수 | 피어슨 | 스피어만 |
|---|---|---|
| Factory_Humidity | -0.303 | -0.278 |
| Factory_Temp | +0.248 | +0.216 |
| Spray_Time | -0.179 | -0.096 |

- 피어슨이 상관계수를 과대 추정하고 있었음

### 3. 신규 변수 발굴

| 변수 | 스피어만 상관계수 |
|---|---|
| Spray_2_Time | 0.1669 |
| Casting_Pressure | -0.1253 |
| Cylinder_Pressure | -0.1210 |
| Cycle_Time | -0.1204 |
| Melting_Furnace_Temp | -0.0770 |
| Spray_1_Time | 0.0485 |
| Coolant_Temp | -0.0416 |
| Velocity_2 | -0.0408 |

> 기존 피어슨 기반 상위 변수만 선택하다 보니 비선형 관계 변수 누락됐던 것

### 4. 파생변수 생성

```python
Velocity_diff_1_2    = Velocity_2 - Velocity_1
Velocity_diff_2_3    = Velocity_3 - Velocity_2
Velocity_diff_3_high = High_Velocity - Velocity_3
Velocity_minmax      = max(Velocity cols) - min(Velocity cols)
Pressure_Diff_ratio  = Pressure_Diff / Cylinder_Pressure
```

| 변수 | 스피어만 | 맨휘트니 | 크루스칼 | 최종 |
|---|---|---|---|---|
| Velocity_diff_3_high | ✅ | ✅ | ✅ | ✅ 포함 |
| Velocity_minmax | ✅ | ✅ | ✅ | ✅ 포함 |
| Velocity_diff_1_2 | ✅ | ✅ | ❌ | ❌ 제외 |
| Velocity_diff_2_3 | ❌ | ❌ | ❌ | ❌ 제외 |
| Pressure_Diff_ratio | ✅ | ✅ | ✅ | ✅ 포함 |

### 5. 통계 검정 결과

#### 맨휘트니 효과크기 (r)
| 해석 | 변수 | r |
|---|---|---|
| 중 | Factory_Humidity | 0.3895 |
| 중 | Factory_Temp | 0.3013 |
| 소 | Biscuit_Thickness | 0.2221 |
| 소 | 나머지 17개 | < 0.18 |

#### 크루스칼-왈리스 효과크기 (ε²)
| 해석 | 변수 | ε² |
|---|---|---|
| 중 | Factory_Humidity | 0.1032 |
| 중 | Factory_Temp | 0.0834 |
| 소 | 나머지 18개 | < 0.03 |

### 6. 던 테스트 (사후검정) 주요 결과

- **Factory_Humidity**: Normal vs 모든 불량 유형 전부 유의 / Exfoliation vs Short_Shot 유의하지 않음
- **Factory_Temp**: Exfoliation vs Normal 유의하지 않음 → **Exfoliation은 온도보다 습도에 민감**
- **Biscuit_Thickness**: Normal vs Bubble 유의하지 않음 → Bubble은 두께와 무관

### 7. 최종 변수 확정 (20개)

```python
final_vars = [
    'Factory_Humidity',       # 핵심 ★
    'Factory_Temp',           # 핵심 ★
    'Biscuit_Thickness',
    'Spray_2_Time',
    'Coolant_Temp',
    'Pressure_Diff',
    'Pressure_Diff_ratio',
    'Cycle_Time',
    'Casting_Pressure',
    'Cylinder_Pressure',
    'Spray_1_Time',
    'Spray_Time',
    'Pressure_Rise_Time',
    'Melting_Furnace_Temp',
    'High_Velocity',
    'Velocity_diff_3_high',
    'Velocity_minmax',
    'Clamping_Force',
    'Air_Pressure',
    'Coolant_Pressure',
]
```

> 어제 대비 변경: `Velocity_1~3`, `Velocity_Avg`, `Coolant_Temp_Range` 제거 / `Spray_2_Time`, `Coolant_Temp`, `Casting_Pressure`, `Cylinder_Pressure`, `Melting_Furnace_Temp`, `Spray_1_Time`, `Velocity_diff_3_high`, `Velocity_minmax`, `Pressure_Diff_ratio` 추가

### 8. Git 작업 내역

| PR | 내용 |
|---|---|
| PR #5 | 통계 분석 초안 |
| PR #6 | 파생변수 추가 및 스피어만 전환 |
| PR #7 | 던 테스트 추가, 최종 변수 확정 |

### 9. 후속 과제

| 항목 | 내용 |
|---|---|
| ⏳ 모델 재학습 | final_vars 20개로 피처 교체 후 재학습 |
| ⏳ SHAP 재확인 | 통계 결과와 SHAP 중요도 비교 |
| ⏳ eda_pipeline_type1 업데이트 | STEP 9 피어슨 → 스피어만 교체 |
| 💡 Exfoliation 심층 분석 | 습도 구간별 박리 불량률 분석 |
| 💡 Type 2 팀과 비교 | Pressure_Diff_ratio 효과 비교 |

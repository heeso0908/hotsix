# 다이캐스팅 공정 품질 예측 — 작업 로그
> **분석 대상**: DieCasting_Quality_Raw_Data.csv, df_clean.csv, Product_Type1.csv  
> **담당**: 김재천 (EDA / 다중공선성 확인 / 통계 분석 파이프라인 구축)

---

## 작업 요약 (EDA 및 통계 분석)

> **분석 대상**: 기본 공정 데이터 및 Product_Type 1 분리 데이터

### 1. 기본 현황 및 EDA (탐색적 데이터 분석)

| 작업 | 내용 | 상태 |
|---|---|---|
| 중복 데이터 확인 | id 제외 중복 2,918건 확인 (1 Shot = 2번 기록).<br/>중복 데이터 내역 분석 (불량 일치 1,232건, 양품 일치 4,604건 식별) | ✅ 완료 |
| 결측치 점검 | Sensor 하위 6개 컬럼에 각각 90개의 결측치 발견 (Type 1은 결측 없음 확인) | ✅ 완료 |
| 분포 파악 및 시각화 | 각 수치형 변수 기준 Histogram, Boxplot, Pairplot 그리기 | ✅ 완료 |
| 기술 통계 | 전체 수치형 컬럼의 왜도(Skewness)와 첨도(Kurtosis) 산출 후, 비대칭성에 따른 분포(양/음의 귀무가설, 이상치 꼬리 두께) 정성 해석 추가 | ✅ 완료 |
| 제품 유형 분리 | Product_Type 1과 2의 값 분포가 확연히 달라 개별 데이터프레임으로 분리 저장 (`Product_Type1.csv`, `Product_Type2.csv`) | ✅ 확정 |

### 2. 다중공선성(VIF) 검토 및 변수 정리 결정 사항

*원 변수와 파생 변수 간 상관계수 히트맵 및 통계적 VIF(Variance Inflation Factor) 수치를 기반으로 변수 선택 방향 결정.*

| 검토 그룹 | 판단 / 분석 결과 | 합의 수준 |
|---|---|---|
| 단위 시간 속도 <br/>(`Velocity_1~3`, `High`) | 공정상 연속 진행 속도들이라 불가피한 다중공선성 존재.<br/>파생 변수(구간 속도 차 등) 포함 시 극단적 다중공선성 유발. | 💡 확인 |
| 성형 및 사출 압력 <br/>(`Casting`, `Cylinder`) | `Pressure_Difference`, `Pressure_Difference_Ratio` 와 함께 분석한 결과 높은 다중공선성 확인. | 💡 확인 |
| **최종 채택 변수** | 정보 손실 방지 및 설명력 확보 측면에서 **`Velocity 1, 2, 3`, `High_Velocity`** 및 **`Pressure_Difference_Ratio`** 유지 채택 | ✅ 확정 |
| **최종 제외 변수** | 다중공선성 이슈로 인해 `Velocity 구간 파생변수`, `Casting_Pressure`, `Cylinder_Pressure`, `Pressure_Difference` 는 분석에서 제거 결정 | ✅ 확정 |

> **결론**: 제외 확정 컬럼 및 `Product_Type`, `Defect_Status`, `Defect_Type` 등을 제외하고 통계 검정에 사용할 **독립 변수(X)** 리스트 선별.

### 3. 비모수 통계 분석 파이프라인 도출

정규성 검정 결과에 따라 전체 파이프라인을 비모수(Non-Parametric) 검정으로 방향 전환하여 유의성 도출 및 시각화 진행.

#### 3-1. 정규성 검정 (Shapiro-Wilk)
- **분석**: `scipy.stats.shapiro` 적용 및 QQ-Plot 시각화.
- **결과**: p-value < 0.05로 대상 전체 변수가 꼬리의 비정규성을 가져 정규분포 가설 기각(❌).
- **결단**: **비모수 검정 파이프라인** 수행으로 방향 전환 완료.

#### 3-2. Mann-Whitney U 검정 & 효과크기 (양품 vs 전체 불량)
- **분석**: `Defect_Status` (1=양품, 0=불량) 기준의 두 집단 간 분포 차이 검정.
- **효과크기**: Rank-Biserial Correlation (r) 계산 도입 및 절대값 크기 기준별(r ≥ 0.5: 대, r ≥ 0.3: 중, 그 외 소) 해석 기준 정립.

#### 3-3. Kruskal-Wallis H 검정 & 효과크기 (결함 유형 별 차이)
- **분석**: `Defect_Type` (`Exfoliation`, `Short_Shot`, `Deformation`, `Bubble` 등) 간 독립변수들의 분포 차이가 유의미한지 종합 평가.
- **효과크기**: Epsilon-squared ($\epsilon^2$) 계산 적용, 집단 간 차이의 크기(분산 비율)를 정량화해 순위화 (`대/중/소` 표기).

#### 3-4. Dunn's Test (사후검정) 및 히트맵 시각화
- **분석**: Kruskal-Wallis에서 유의미하게 나온 변수를 대상으로 하여 구체적으로 어떤 불량-불량(또는 양품-불량) 쌍에서 차이가 나타났는지 사후 검정 (Bonferroni 보정 포함).
- **시각화 결과**: Dunn's Test의 p-value 행렬을 **히트맵(Heatmap)** 으로 구성. 0.05 통계적 유의수준을 붉은색 팔레트로 강조하여, 유의한 집단간 영향력을 팀뷰어 직관적으로 확인할 수 있도록 구현 완료.

### 4. 진행 과제

| 항목 | 내용 |
|---|---|
| ⏳ **피처 동기화** | VIF 판단 제외 리스트와 비모수효과 상위 변수 정보 교류 |
| ⏳ **Type 2 EDA 적용** | 본 파이프라인(`statistics_all_iv.ipynb`) 로직을 `df_clean_Type2.csv`에 동일 적용하여 패턴 대조 |

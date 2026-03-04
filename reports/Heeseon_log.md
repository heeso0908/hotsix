# 🏭 다이캐스팅 불량 예측 모델 — 작업 로그

> 👤 **작성자:** 김희선  

---

## 2026-03-03

## 1. 📊 데이터 기본 현황

| 항목 | 값 |
|---|---|
| 전체 데이터 건수 | **7,535건** |
| 양품 | 5,842건 (77.6%) |
| 불량 | 1,689건 (22.4%) |
| 제품 타입 | Product_Type 1 / 2 (별도 분석 예정) |

---

## 2. 🔍 중복 데이터 처리

### 중복 현황

| 중복 기준 | 건수 |
|---|---|
| 전체 행 기준 중복 | 0건 |
| ID 기준 중복 | 0건 |
| **ID 제외 행 중복** | **2,918건 (x 2 = 5,836건)** |
| 단일 기록 | 1,699건 |

### 원인 파악

Shot 1번당 Cavity 1, 2의 불량 데이터가 모두 기록되는 구조에서,
**데이터 수집 과정에 일부 행이 중복 저장**되어 동일 Shot이 2개씩 존재하는 행이 다수 발생.

### 처리 결정

중복 제거 전후 불량률 차이가 크지 않아 **(0.2243 → 0.2325)** 중복 행 제거 후 분석 진행.

```python
df_dedup = df.drop_duplicates(subset=df.columns.drop('Id'))
# 원본: 7,531건 → 중복 제거 후: 4,615건
```

---

## 3. ⚠️ 이상치 처리 — `High_Velocity == 0`

### 발견 경위

`df.describe()`에서 `High_Velocity`의 **Min 값이 0** → 이상치 의심

| 항목 | 값 |
|---|---|
| 0 값 개수 | **4건** |
| 전체 대비 비율 | 0.05% |

### 추가 확인

`High_Velocity == 0`인 4개 행의 **Time 관련 컬럼도 모두 0**이었으며,
해당 행의 **불량 컬럼 26개 전체가 0** (불량 없음).

```
Velocity_1, Velocity_2, Velocity_3, Rapid_Rise_Time, Pressure_Rise_Time 모두 0
불량도 없음 (Short_Shot ~ Inclusions 전 항목 = 0)
```

### 처리 결정

공정 중간에 **중단된 샷(Aborted Shot)** 으로 판단 → **4개 행 삭제**

---

## 4. 🛠️ 파생 변수 생성

### Any_Defect — 불량 여부 통합 컬럼

불량 컬럼 26개 중 **하나라도 불량이 있으면 1**, 없으면 0.

```python
defect_cols = [c for c in df.columns
               if any(x in c for x in [
                   'Short_Shot', 'Bubble', 'Exfoliation', 'Blow_Hole',
                   'Stain', 'Dent', 'Deformation', 'Contamination',
                   'Impurity', 'Crack', 'Scratch', 'Buring_Mark', 'Inclusions'
               ])]

df['Any_Defect'] = (df[defect_cols].sum(axis=1) > 0).astype(int)
```

### 속도 변화량 컬럼 — 속도의 급격한 변화 포착

```python
df['Vel_1_2_Diff']    = df['Velocity_2'] - df['Velocity_1']
df['Vel_2_3_Diff']    = df['Velocity_3'] - df['Velocity_2']
df['Vel_Total_Range'] = df['High_Velocity'] - df[['Velocity_1','Velocity_2','Velocity_3']].min(axis=1)
```

> 💡 속도가 급격히 변하는 구간에서 불량 발생 가능성이 높을 것으로 가설 설정

---

## 5. ⚖️ 클래스 불균형 확인

| 클래스 | 건수 | 비율 |
|---|---|---|
| 양품 (0) | 5,842건 | **77.6%** |
| 불량 (1) | 1,689건 | **22.4%** |

> ⚠️ 약 **4:1 불균형** — 모델링 시 오버샘플링(SMOTE 등) 또는 class_weight 조정 고려 필요

---

## 6. 📉 중복 제거 전후 불량률 비교

| 구분 | 행 수 | 불량률 |
|---|---|---|
| 원본 | 7,531건 | 22.43% |
| 중복 제거 후 | 4,615건 | 23.25% |

불량률 변화 미미 **(+0.82%p)** → 중복 제거가 전체 분포에 큰 영향 없음을 확인

---

## 7. 🔜 다음 단계

- [ ] `Product_Type 1` / `Product_Type 2` 분리하여 각각 EDA 추가 진행
- [ ] 피처별 불량률 분포 시각화 (Cylinder_Pressure, Casting_Pressure 등)
- [ ] 속도 변화량 컬럼과 불량의 상관관계 확인
- [ ] 전처리 파이프라인 정의 후 모델링 시작

---

> 📝 **오늘의 핵심 인사이트**
> 중복 데이터는 수집 구조상 발생한 것으로, 제거 후에도 불량률 변화 미미.
> `High_Velocity == 0` 행은 중단된 샷으로 판단하여 제거.
> `Product_Type`별로 데이터 특성이 다를 수 있어 분리 분석 예정.

---

## 2026-03-04

## 1️⃣ 분석 목적

- Product_Type별 공정 데이터 특성 비교
- 이상치 존재 여부 확인 및 처리 여부 판단
- 공정 변수 간 다중공선성 확인
- 머신러닝 모델 학습을 위한 Feature 정리


---

# 2️⃣ Product_Type 2 데이터 EDA 진행

기존에 수행했던  

`eda_pipeline_type1_v3.ipynb`

EDA 과정을 **Product_Type = 2 데이터에도 동일하게 적용하여 분석 진행**


---

# 3️⃣ Rapid_Rise_Time 이상치 처리 여부 검토

## 🔎 문제 제기

Product_Type = 2 데이터 기준  

`Rapid_Rise_Time` 컬럼에서 **IQR 기준 이상치 145건 확인**

### IQR 기준 (Sec)

| 항목 | 값 |
|----|----|
| Lower Bound | 0.0095 |
| Upper Bound | 0.0135 |

### 실제 데이터 범위 (Sec)

| 항목 | 값 |
|----|----|
| Min | 0.0090 |
| Max | 0.0140 |

### 관찰 결과

- 전체 데이터 범위 자체가 매우 좁음
- IQR 기준 이상치로 분류된 값도 실제 데이터 범위와 큰 차이가 없음
- 공정 특성상 **센서 노이즈 또는 환경 영향으로 발생 가능한 수준의 변동**으로 판단

### ✔ 판단

- 실제 공정 데이터로 판단 가능
- 이상치 제거 시 **공정 데이터 왜곡 가능성 존재**

### 🛠 조치

**Rapid_Rise_Time 이상치 제거하지 않고 유지**


---

# 4️⃣ Pressure 관련 컬럼 다중공선성 확인

## 대상 컬럼

- `Cylinder_Pressure`
- `Casting_Pressure`
- `Pressure_Difference`
- `Pressure_Difference_Ratio`

## 문제

VIF 분석 결과  

**압력 변수 간 다중공선성 매우 높음**

→ 일부 변수는 **선형 종속 관계 가능성 존재**

## 원인

압력 컬럼들이 서로 파생 관계

- Pressure_Difference = Cylinder_Pressure - Casting_Pressure
- Pressure_Difference_Ratio = Casting_Pressure / Cylinder_Pressure


즉 동일한 정보를 여러 컬럼이 포함하고 있음

### ✔ 판단

공정 의미를 유지하면서  
다중공선성을 최소화하기 위해

**Pressure_Difference_Ratio 컬럼만 사용**

### 🛠 조치

| 유지 컬럼 | 제거 컬럼 |
|----|----|
| Pressure_Difference_Ratio | Cylinder_Pressure |
| | Casting_Pressure |
| | Pressure_Difference |


---

# 5️⃣ Velocity 관련 컬럼 정리

## 초기 분석 변수

### 기본 컬럼 (레시피 설정 값)

- `Velocity_1`
- `Velocity_2`
- `Velocity_3`
- `High_Velocity`

### 파생 컬럼 (속도 변화량)

EDA 과정에서 속도 변화 패턴 확인을 위해 생성

- Velocity_2_1 = Velocity_2 - Velocity_1
- Velocity_3_2 = Velocity_3 - Velocity_2
- Velocity_High_3 = High_Velocity - Velocity_3
- Velocity_Max_Min = max(Velocity_1, Velocity_2, Velocity_3, High_Velocity) - min(...)


## 문제

속도 관련 변수 간 **다중공선성 가능성 존재**

- 파생 컬럼이 기존 Velocity 값으로부터 계산된 값
- 동일 정보가 여러 변수에 중복 포함될 가능성 존재

## 실무 관점 고려

다이캐스팅 공정에서는 실제로  

**레시피 설정 값(속도 값)을 직접 조정하여 공정을 튜닝**

실제 현장에서 조정 가능한 값

- `Velocity_1`
- `Velocity_2`
- `Velocity_3`
- `High_Velocity`

파생 변수는 **공정 설정값이 아닌 분석용 변수**

### ✔ 판단

모델 결과를 **실제 공정 최적화에 활용하기 위해**

파생 컬럼 제외

**레시피 설정 값 기반으로 모델 학습 진행**

### 🛠 최종 Velocity 변수

| 사용 컬럼 | 제외 컬럼 |
|----|----|
| Velocity_1 | Velocity_2_1 |
| Velocity_2 | Velocity_3_2 |
| Velocity_3 | Velocity_High_3 |
| High_Velocity | Velocity_Max_Min |


---

# 6️⃣ 최종 Feature 구성 방향

## Velocity 관련 변수

- Velocity_1
- Velocity_2
- Velocity_3
- High_Velocity

## Pressure 관련 변수

- Pressure_Difference_Ratio  
  *(= Casting_Pressure / Cylinder_Pressure)*


---

# 📌 오늘 분석 핵심 정리

- Product_Type 2 데이터 EDA 수행
- Rapid_Rise_Time IQR 기준 이상치 145건 확인
- 실제 공정 범위와 차이가 작아 **이상치 제거하지 않기로 결정**

- Pressure 관련 변수 **다중공선성 확인**
- 압력 변수 중 **Pressure_Difference_Ratio만 사용**

- Velocity 파생 변수 제거
- 실제 공정 **레시피 설정값 중심으로 Feature 구성**


---

# 📎 다음 분석 예정

- Product_Type별 공정 변수 분포 비교
- 속도 / 압력 변수와 불량 발생 관계 분석
- Feature 중요도 분석
- 머신러닝 모델 구축 및 성능 비교
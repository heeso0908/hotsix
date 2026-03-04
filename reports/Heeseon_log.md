# 🏭 다이캐스팅 불량 예측 모델 — 작업 로그

> 📅 **작성일:** 2026-03-03  
> 👤 **작성자:** 김희선  
> 🔖 **오늘 작업:** EDA · 변수/피처 정의

---

## 📋 목차
1. [데이터 기본 현황]
2. [중복 데이터 처리]
3. [이상치 처리 — High_Velocity == 0]
4. [파생 변수 생성]
5. [클래스 불균형 확인]
6. [중복 제거 전후 불량률 비교]
7. [다음 단계]

---

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
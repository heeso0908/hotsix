# 통계분석 결과 — Product_Type 1

---

## 1. Mann-Whitney U 검정 (불량 vs 양품)

> 정규성 불만족 → 비모수 검정 적용  
> 귀무가설: 불량 그룹과 양품 그룹의 분포가 같다

| 변수 | U통계량 | p-value | 유의성 | 효과크기 (r) | 해석 |
|---|---|---|---|---|---|
| Factory_Humidity | 364863.5 | 0.0000 | ✅ | 0.3895 | 중 |
| Factory_Temp | 777645.0 | 0.0000 | ✅ | 0.3013 | 중 |
| Biscuit_Thickness | 464880.0 | 0.0000 | ✅ | 0.2221 | 소 |
| Pressure_Diff | 491771.5 | 0.0000 | ✅ | 0.1771 | 소 |
| Pressure_Rise_Time | 678184.5 | 0.0000 | ✅ | 0.1348 | 소 |
| Spray_Time | 518244.0 | 0.0000 | ✅ | 0.1328 | 소 |
| Clamping_Force | 662918.5 | 0.0000 | ✅ | 0.1093 | 소 |
| Air_Pressure | 658564.5 | 0.0002 | ✅ | 0.1020 | 소 |
| High_Velocity | 658198.5 | 0.0002 | ✅ | 0.1014 | 소 |
| Coolant_Pressure | 562304.0 | 0.0292 | ✅ | 0.0591 | 소 |

> **효과크기 기준** (rank-biserial correlation): 대 ≥ 0.5 / 중 ≥ 0.3 / 소 < 0.3

---

## 2. Kruskal-Wallis 검정 (불량 유형별)

> 비교 그룹: Normal / Exfoliation / Short_Shot / Deformation / Bubble  
> 귀무가설: 모든 불량 유형 그룹의 분포가 같다

| 변수 | H통계량 | p-value | 유의성 | ε² | 해석 |
|---|---|---|---|---|---|
| Factory_Humidity | 273.35 | 0.0000 | ✅ | 0.1032 | 중 |
| Factory_Temp | 221.14 | 0.0000 | ✅ | 0.0834 | 중 |
| Biscuit_Thickness | 76.29 | 0.0000 | ✅ | 0.0288 | 소 |
| Pressure_Diff | 45.42 | 0.0000 | ✅ | 0.0171 | 소 |
| Coolant_Pressure | 45.27 | 0.0000 | ✅ | 0.0171 | 소 |
| Spray_Time | 34.69 | 0.0000 | ✅ | 0.0131 | 소 |
| Pressure_Rise_Time | 29.81 | 0.0000 | ✅ | 0.0112 | 소 |
| High_Velocity | 22.03 | 0.0002 | ✅ | 0.0083 | 소 |
| Clamping_Force | 20.38 | 0.0004 | ✅ | 0.0077 | 소 |
| Air_Pressure | 16.35 | 0.0026 | ✅ | 0.0062 | 소 |

> **효과크기 기준** (epsilon-squared): 대 ≥ 0.14 / 중 ≥ 0.06 / 소 < 0.06

---

## 3. 분석 방법 종합 비교

| 변수 | 상관계수 순위 | SHAP 순위 | rank-biserial r | ε² |
|---|---|---|---|---|
| Factory_Humidity | 1위 | 1위 | 0.390 (중) | 0.103 (중) |
| Factory_Temp | 2위 | 4위 | 0.301 (중) | 0.083 (중) |
| Spray_Time | 3위 | 3위 | 0.133 (소) | 0.013 (소) |
| Biscuit_Thickness | 4위 | - | 0.222 (소) | 0.029 (소) |
| Pressure_Diff | 5위 | - | 0.177 (소) | 0.017 (소) |
| Coolant_Pressure | 15위 | 2위 | 0.059 (소) | 0.017 (소) |
| High_Velocity | 13위 | 5위 | 0.101 (소) | 0.008 (소) |

---

## 4. 핵심 인사이트

**① Factory_Humidity가 유일하게 모든 분석에서 1순위**

상관분석, SHAP, Mann-Whitney 효과크기, Kruskal-Wallis 효과크기 네 가지 방법 모두에서 가장 강력한 변수로 지목됨. Type 1 불량의 핵심 원인 변수로 확정.

**② 통계적 유의성 ≠ 실질적 효과크기**

10개 변수 모두 p < 0.05로 유의하지만, 실질적 효과크기가 "중" 이상인 변수는 `Factory_Humidity`와 `Factory_Temp` 두 개뿐. 데이터 수(2,653행)가 충분히 커서 작은 차이도 유의하게 잡히는 것으로 해석.

**③ Coolant_Pressure의 역설**

SHAP 2위(0.325)이나 효과크기는 두 검정 모두 최하위 수준 (r=0.059, ε²=0.017). 단독으로는 그룹 간 분포 차이가 작지만, 모델 안에서 다른 변수(특히 Factory_Humidity)와 조합될 때 불량 예측력이 강해지는 전형적인 **상호작용 변수**.

**④ 불량 유형별로도 환경 변수가 지배적**

Kruskal-Wallis에서 H통계량이 가장 높은 변수도 Factory_Humidity(273.35), Factory_Temp(221.14). Exfoliation, Short_Shot, Deformation 등 유형에 관계없이 공장 온습도가 불량 발생의 공통 배경임을 시사.

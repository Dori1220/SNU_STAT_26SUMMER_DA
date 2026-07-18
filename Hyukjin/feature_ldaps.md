# LDAPS 시간 단위 집계 (KPX Group 1)

LDAPS 격자별 long 데이터(420,864행 × 42컬럼)를 **시각당 1행**(26,304행 × 30컬럼)으로 축약하는 파이프라인.

- 입력: `ldaps_train_group1` (격자별 파생변수 생성 완료 상태), `nearest_info_ldaps_group1`
- 출력: `ldaps_group1_hourly` — `train_labels`와 `forecast_kst_dtm` 기준으로 join 가능한 wide 테이블

---

## 1. 파이프라인 구조

```
격자별 long (시각 × 16격자)
   │
   │  ① 격자별 파생변수 계산  ← 비선형 파생은 반드시 이 단계에서
   ▼
   │  ② 터빈별 IDW 보간 (터빈 6개 × 최근접 격자 4개, 거리 역수 가중)
   ▼
터빈별 (시각 × 6터빈)
   │  ③ 그룹 집계 — power_proxy는 sum, 나머지는 mean, ws_hub는 std 추가
   ▼
시각당 1행 (26,304 × 30)
```

**핵심 원칙: 비선형 파생은 격자별로 먼저 계산하고 나중에 집계한다.**
`mean(v³) ≠ mean(v)³` 이므로 집계 후 파생을 만들면 물리적으로 틀린 값이 나온다.

**터빈 단계를 거치는 이유**: 터빈 6개를 동일 가중으로 평균하므로 수학적으로는 "격자별 결합가중치로 가중평균"과 같다. 그럼에도 터빈 단계를 두는 것은 `ws_hub_std`(터빈 간 편차) 때문이며, 이 값은 터빈 단위 값이 있어야만 계산된다.

---

## 2. 전체 코드

```python
# ── 1) 격자별 난류강도 파생 (50m max-min 폭) ──────────────
ldaps_train_group1['gust50_ldaps'] = np.hypot(
    ldaps_train_group1['heightAboveGround_50_50MUmax'] - ldaps_train_group1['heightAboveGround_50_50MUmin'],
    ldaps_train_group1['heightAboveGround_50_50MVmax'] - ldaps_train_group1['heightAboveGround_50_50MVmin'],
)

# ── 2) 집계 대상 변수 (삭제 대상은 여기 미포함) ───────────
IDW_COLS = [
    'power_proxy_ldaps',                                     # sum 으로 집계
    'ws_hub_ldaps', 'ws10_ldaps', 'ws50_ldaps', 'alpha', 'gust50_ldaps',
    'heightAboveGround_10_10u', 'heightAboveGround_10_10v',  # 풍향 계산용
    'rho_hub_ldaps', 't_hub_ldaps',
    'heightAboveGround_2_r', 'heightAboveGround_2_q',
    'surface_0_sp', 'etc_0_blh',
    'surface_0_NDNSW', 'surface_0_NDNLW',
    'etc_0_hcc', 'etc_0_mcc', 'etc_0_lcc',
    'surface_0_lssrate', 'surface_0_SNOM',
    'heightAboveGround_5_XBLWS', 'heightAboveGround_5_YBLWS',
]

# ── 3) 터빈별 IDW 가중치 (터빈별 합 = 1로 정규화) ─────────
weight_ldaps_group1 = nearest_info_ldaps_group1[['호기', 'grid_id', 'dist_km']].copy()
weight_ldaps_group1['weight'] = 1 / weight_ldaps_group1['dist_km']
weight_ldaps_group1['weight'] = (
    weight_ldaps_group1['weight'] /
    weight_ldaps_group1.groupby('호기')['weight'].transform('sum')
)

# ── 4) 터빈 단위로 축약 (시각 × 터빈) ─────────────────────
merged_ldaps_group1 = ldaps_train_group1[['forecast_kst_dtm', 'grid_id'] + IDW_COLS].merge(
    weight_ldaps_group1[['호기', 'grid_id', 'weight']],
    on = 'grid_id',
    how = 'inner'
)
for col in IDW_COLS:
    merged_ldaps_group1[col] = merged_ldaps_group1[col] * merged_ldaps_group1['weight']

turbine_level_ldaps_group1 = (
    merged_ldaps_group1
    .groupby(['forecast_kst_dtm', '호기'])[IDW_COLS]
    .sum()
)

# ── 5) 그룹 단위로 축약 (시각 1행) ────────────────────────
MEAN_COLS = [c for c in IDW_COLS if c != 'power_proxy_ldaps']
agg_spec = {c: 'mean' for c in MEAN_COLS}
agg_spec['power_proxy_ldaps'] = 'sum'          # label = 터빈 발전량의 합

grouped = turbine_level_ldaps_group1.groupby('forecast_kst_dtm')
ldaps_group1_hourly = grouped.agg(agg_spec)
ldaps_group1_hourly['ws_hub_std_ldaps'] = grouped['ws_hub_ldaps'].std()   # 터빈 간 이질성
ldaps_group1_hourly = ldaps_group1_hourly.rename(
    columns = {'power_proxy_ldaps': 'group1_power_proxy_ldaps'}
)

# ── 6) 풍향: 성분 평균 후 각도 계산 -> sin/cos 인코딩 ─────
# 각도를 산술평균하면 0°/360° 경계에서 깨지므로 u, v 평균 후 atan2
wd_rad = np.arctan2(-ldaps_group1_hourly['heightAboveGround_10_10u'],
                    -ldaps_group1_hourly['heightAboveGround_10_10v'])
ldaps_group1_hourly['wd10_sin_ldaps'] = np.sin(wd_rad)
ldaps_group1_hourly['wd10_cos_ldaps'] = np.cos(wd_rad)

# ── 7) 누수 검증용 시각 결합 ──────────────────────────────
avail_ldaps_group1 = ldaps_train_group1.groupby('forecast_kst_dtm')['data_available_kst_dtm'].first()
ldaps_group1_hourly = ldaps_group1_hourly.join(avail_ldaps_group1).reset_index()

# ── 8) 시간 파생변수 ─────────────────────────────────────
dtm = pd.to_datetime(ldaps_group1_hourly['forecast_kst_dtm'])
ldaps_group1_hourly['hour']  = dtm.dt.hour     # 안정도 편향 + 예보 리드타임을 함께 담음
ldaps_group1_hourly['month'] = dtm.dt.month    # 계절별 예보 편향
```

**검증 결과**: 26,304행 × 30컬럼, 결측 0, `power_proxy` 합 일치, `sin²+cos²=1`.

---

## 3. 집계 방식별 근거

### `sum` — `power_proxy_ldaps` 하나뿐

label이 그룹 총발전량이므로 터빈별 발전량의 합과 대응된다. 나머지 기상변수는 "그룹이 놓인 조건"이므로 평균이 맞다.

### `mean` — 대부분의 변수

터빈 6개가 1.06 km 안에 모여 있어(LDAPS 격자 간격 1.5 km) 사실상 같은 조건을 공유한다. 평균이 그룹 대표값으로 충분하다.

### `std` — `ws_hub_ldaps` 만

터빈 간 풍속 편차(상대 3.11%)를 1컬럼으로 압축한다. 터빈별 6개 컬럼을 그대로 노출하는 대신 이것만 남긴다 (근거는 5절 참조).

### 벡터평균 — 풍향

u, v를 먼저 평균한 뒤 `atan2` 적용. 격자별로 각도를 구해 산술평균하면 359°와 1°가 180°로 뭉개진다. `atan2(-u, -v)`는 기상학 관례인 "바람이 불어오는 방향"을 준다.

---

## 4. 삭제 변수

### 수학적으로 확실한 삭제

| 변수 | 근거 |
|---|---|
| `grid_id`, `latitude`, `longitude` | 집계 키·식별자. 집계 후 의미 소멸 |
| `surface_0_lsm` | 전 행이 `1.0` (모두 육지). 분산 0 |
| `surface_0_h` | 격자별 시간변동 0. 집계하면 상수 (869~1001m는 격자 간 차이일 뿐) |
| `heightAboveGround_2_t` | `t_hub_ldaps`와 **r = 1.0000000000** (상수 이동) |
| `surface_0_ncpcp` | `surface_0_avg_lsprate`와 **r = 1.000** 완전 중복 |
| `surface_0_snol` | `surface_0_lssrate`와 **r = 1.000** 완전 중복 |

### 상관 ≈ 0 이거나 중복

| 변수 | label 상관 | 근거 |
|---|---:|---|
| `etc_0_VLCDC` | -0.0016 | 65% 0값, `lcc`와 r=0.840 |
| `surface_0_avg_lsprate` | -0.0021 | 88% 0값, 신호 없음 |
| `meanSea_0_prmsl` | -0.0295 | `sp`(-0.2113)보다 열등. 해발 900m에서 해면 환산 시 가상 기층 온도 가정이 잡음을 주입 |
| `heightAboveGround_2_SWDIR` | -0.1986 | `NDNSW`와 **r=0.958** |
| `heightAboveGround_2_SWDIF` | -0.2281 | `NDNSW`와 r=0.706 |
| `heightAboveGround_2_dpt` | -0.2700 | `t`(r=0.932) + `q`(r=0.944)의 조합. 독립 정보 없음 |

### 축약

| 변수 | 처리 |
|---|---|
| `50MUmax/min`, `50MVmax/min` | → `gust50_ldaps` 1개로 (max−min 폭 = 난류강도) |

### 절대 넣으면 안 되는 것

| 변수 | 근거 |
|---|---|
| `forecast_kst_dtm` (숫자 변환) | train(2022-01~2025-01)과 test(2025-01~2026-01)가 **1초도 겹치지 않고 test가 전부 이후**. 트리는 외삽 불가 → 2025년 행이 모든 시간 분할에서 같은 쪽으로 떨어짐. **컬럼 자체는 키로 반드시 유지**하되 feature에서 제외 |
| `year` | 위와 동일. 2025는 학습에 없는 값 |
| `dow` | 잔차 변동폭 0.019 (`hour`의 1/11). 바람에 주간 주기 없음 |
| `lead_h` | `lead_h = 12 + ((hour-1) mod 24)` 결정적 관계 → `hour`와 완전 중복 |

키 컬럼은 삭제 대신 feature 목록을 명시적으로 정의해 제외한다:

```python
KEY_COLS = ['forecast_kst_dtm', 'data_available_kst_dtm']
FEATURE_COLS = [c for c in ldaps_group1_hourly.columns if c not in KEY_COLS]
```

---

## 5. 검증 기록

이 파이프라인의 설계 근거가 된 실측 결과. 재검토 시 참고.

### `hour`가 가장 강력한 변수

풍속을 10분위로 고정(= NWP가 이미 아는 정보를 통제)한 뒤 잔차를 시간 변수로 설명한 결과:

| 변수 | 잔차 변동폭 (이용률) | 판정 |
|---|---:|---|
| `hour` | **0.2092** | 필수 |
| `month` | **0.1751** | 유지 |
| `dow` | 0.0190 | 삭제 |

시간별 잔차 패턴 (같은 예보풍속 대비 실제 발전량):

```
새벽 04시 +0.0905   ← 안정 성층, 전단 강함
아침 08시 +0.0487
정오 12시 -0.0945
오후 15시 -0.1187   ← 대류 혼합, 전단 약함
저녁 20시 +0.0036
```

야간–오후 간 이용률 **21%p 격차**. `alpha`로 잡으려 한 안정도 효과가 그대로 남아 있으며, `hour`가 이를 메운다.

### 허브높이 외삽이 원본 풍속보다 못함

| 변수 | label 상관 (전체) | label 상관 (유효시간대) |
|---|---:|---:|
| `ws10_ldaps` | **0.7322** | **0.5806** |
| `ws50_ldaps` | 0.7213 | 0.5645 |
| `power_proxy_ldaps` | 0.7191 | — |
| `ws_hub_ldaps` | 0.7095 | 0.5507 |

외삽 단계를 거칠수록 상관이 단조 감소. 원인은 `alpha = ln(ws50/ws10)/ln(5)`가 잡음 있는 두 값의 비율이라 오차가 증폭되고, `(117/50)^alpha`(중앙값 1.23의 변동 배율)를 곱하면 ws50의 순서가 흐트러지기 때문. `alpha` 자체의 label 상관도 유효시간대에서 0.11에 불과.

**→ `ws10_ldaps`, `ws50_ldaps`를 지우지 말 것. `ws_hub`가 우월하다는 전제는 데이터로 기각됨.**

### 터빈별 feature를 노출하지 않는 이유

터빈별 IDW 풍속 간 상관:

| 그룹 | 터빈간 최대거리 | 비대각 상관 최소 | 터빈간 상대편차 |
|---|---:|---:|---:|
| Group 1 | 1.06 km | 0.9894 | 3.11% |
| Group 3 | 2.06 km | 0.9949 | 2.17% |

Group 1의 터빈 6개가 모두 같은 격자(g5, g6, g10, g11)를 가중치만 달리해 공유한다. Group 3는 지리적으로 더 넓은데도 상관이 더 높은데, 원동 능선이 격자 배열과 나란해 같은 격자 위를 미끄러지기 때문. 즉 **절대 거리가 아니라 격자 배열을 가로지르는지**가 관건.

→ 터빈별 6개 컬럼 대신 `ws_hub_std` 1개로 압축.

### 파워커브 클리핑을 하지 않는 이유

| | 클리핑 X | 클리핑 O |
|---|---:|---:|
| Spearman vs ws_hub | 0.999761 | 0.991592 |
| 고유값 비율 | **100.00%** | **63.61%** |
| 3600으로 뭉개진 행 | 0% | 24.30% |
| 0으로 뭉개진 행 | 0% | 12.09% |

클리핑은 단조변환이므로 트리는 클리핑 없이도 동일한 분할을 찾을 수 있다. 반면 단사가 아니므로 36.4%의 행을 두 값으로 뭉개 **정보를 파괴하기만 한다**. 특히 cut-out은 비단조 현상(고풍속 → 출력 0)이라, 클리핑하면 트리가 그 구간을 분할할 수 없게 된다.

`Cp`, `0.5`, `SWEPT_AREA`, `/1000`은 모두 상수배라 트리에 영향이 0. 물리 단위 유지는 산점도 진단용.

**단, 선형모델·NN을 쓴다면 클리핑이 결정적으로 중요해진다.**

### 중간 계산은 feature가 아님

| 변수 | 상관 | 역할 |
|---|---:|---|
| `heightAboveGround_2_t` vs `t_hub` | **1.000000** | 상수 이동 |
| `Tv_hub` vs `t_hub` | **0.999303** | rho 계산용 |
| `P_hub` vs `sp` | **0.997211** | 거의 상수배 (0.986) |
| `rho_hub` vs `t_hub` | **-0.982418** | rho는 사실상 기온의 변형 |
| `rho_hub` vs `P_hub` | 0.240360 | 기압은 rho에 흡수되지 않음 → `sp` 유지 |

`Tv_hub`, `P_hub`는 `rho_hub` 계산 후 폐기. 단 rho 계산에는 반드시 Tv를 써야 하며(Tv 사용 vs T 사용 시 rho가 평균 0.43%, 최대 1.30% 차이 → 발전량에 선형 반영).

**미해결**: `rho_hub` vs `t_hub`가 r=-0.982로 중복 수준. 검증에서 (rho만 / t_hub만 / 둘 다)를 비교해 결정할 것. `t_hub`를 남기는 근거는 착빙(icing)이며, rho는 이 비선형 단절을 표현하지 못한다.

### 착빙 조건

격자 고도 869~1001 m. 허브고도 기온 기준:

| 조건 | 비율 |
|---|---:|
| 0°C 미만 | 29.0% |
| 착빙 위험대 (-8~+2°C & RH>90%) | 8.3% |

`t_hub` + `heightAboveGround_2_r`을 남겨 트리가 스스로 경계를 찾게 한다. 임계값을 직접 지정한 `icing_risk` 플래그는 근거가 약하므로 검증으로 이득을 확인한 뒤에만 채택.

### 사용 격자

Group 1 터빈이 실제로 쓰는 격자는 **16개 중 6개** (1, 2, 5, 6, 10, 11). `how='inner'` merge에서 나머지 10개는 자동 제외된다. 공간 gradient feature를 만들 계획이라면 이 파이프라인 **바깥에서** `ldaps_train_group1` 원본을 따로 사용해야 한다.

---

## 6. 출력 컬럼 (30개)

| 분류 | 컬럼 |
|---|---|
| 키 | `forecast_kst_dtm`, `data_available_kst_dtm` |
| 타깃 근사 | `group1_power_proxy_ldaps` |
| 풍속 | `ws_hub_ldaps`, `ws_hub_std_ldaps`, `ws10_ldaps`, `ws50_ldaps`, `alpha`, `gust50_ldaps` |
| 풍향 | `wd10_sin_ldaps`, `wd10_cos_ldaps`, `heightAboveGround_10_10u`, `heightAboveGround_10_10v` |
| 열역학 | `rho_hub_ldaps`, `t_hub_ldaps`, `heightAboveGround_2_r`, `heightAboveGround_2_q`, `surface_0_sp` |
| 경계층 | `etc_0_blh`, `heightAboveGround_5_XBLWS`, `heightAboveGround_5_YBLWS` |
| 복사 | `surface_0_NDNSW`, `surface_0_NDNLW` |
| 운량 | `etc_0_hcc`, `etc_0_mcc`, `etc_0_lcc` |
| 강수 | `surface_0_lssrate`, `surface_0_SNOM` |
| 시간 | `hour`, `month` |

`heightAboveGround_10_10u/v`는 풍향 계산 후 삭제 가능하나, `10u`는 label 상관 0.708로 그 자체가 강한 신호(탁월풍이 서풍)라 유지도 타당. 검증으로 확인할 것.

`heightAboveGround_5_XBLWS/YBLWS`는 `ws10`과 r=0.672로 완전 중복은 아니고 label 상관도 0.42~0.49이나, 5m는 터빈과 무관한 고도. permutation importance로 확인 후 결정.

---

## 7. 남은 작업

- [ ] 동일 파이프라인을 `ldaps_test_group1`에 적용 (함수화 필요)
- [ ] `rho_hub` vs `t_hub` 택일 검증
- [ ] `10u/10v`, `5_XBLWS/YBLWS` 유지 여부를 permutation importance로 결정
- [ ] Group 2, 3 적용 시 상관 검사 재실행 (Group 3는 로터 직경 136m로 `SWEPT_AREA` 상이)

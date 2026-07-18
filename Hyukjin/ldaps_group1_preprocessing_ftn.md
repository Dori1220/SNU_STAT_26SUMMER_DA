# LDAPS 전처리 함수 (`preprocess_ldaps_group`)

LDAPS 격자별 long 데이터를 시각당 1행 wide 테이블로 변환하는 전처리 전 과정을 함수화한 것. **train / test 동일 적용**을 목적으로 하며, test 결측치 보간까지 포함한다.

- 입력: LDAPS CSV 경로, KPX 그룹 번호
- 출력: 시각당 1행 DataFrame (Group 1 기준 30컬럼)
- 관련 문서: 집계 로직의 상세 근거는 [`feature_ldaps.md`](feature_ldaps.md) 참조

---

## 1. 함수 전체 코드

```python
def preprocess_ldaps_group(
        ldaps_path: str,
        kpx_group: int,
        k: int = 4,
        cp: float = 0.35,
        info_path: str = "../data/info_include_lat_lon.csv",
) -> pd.DataFrame:
    """LDAPS 격자별 long 데이터 -> 시각당 1행 wide 테이블.
    train/test 동일 적용. test 결측은 예보 배치 내 시간 보간으로 처리.
    """
    # ── 물리 상수 ──
    REF_LOW, REF_HIGH, REF_T = 10.0, 50.0, 2.0
    LAPSE_RATE, G, R_D = 0.0065, 9.80665, 287.05

    # ── 0) 로드 및 터빈 제원 (group3는 로터 136m) ──
    ldaps = pd.read_csv(ldaps_path, encoding = 'utf-8-sig')
    info = pd.read_csv(info_path, encoding = 'utf-8-sig')
    group_info = info[info['KPX그룹'] == kpx_group].copy()

    hub_height = float(group_info['Hub Height(m)'].iloc[0])
    rotor_d    = float(group_info['Rotor Diameter(m)'].iloc[0])
    swept_area = np.pi * (rotor_d / 2) ** 2

    # ── 1) 결측치 보간 ──
    # 같은 예보 배치 안에서만 보간 -> 미래 배치 정보 유입(누수) 차단
    ldaps = ldaps.sort_values(['grid_id', 'data_available_kst_dtm', 'forecast_kst_dtm']).reset_index(drop = True)
    num_cols = [c for c in ldaps.select_dtypes(include = [np.number]).columns if c != 'grid_id']
    ldaps[num_cols] = (
        ldaps.groupby(['grid_id', 'data_available_kst_dtm'])[num_cols]
        .transform(lambda g: g.interpolate(method = 'linear', limit_direction = 'both'))
    )

    # ── 2) 터빈-격자 최근접 매핑 ──
    grids = ldaps.drop_duplicates('grid_id').sort_values('grid_id').reset_index(drop = True)

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0088
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    grid_lat, grid_lon = grids['latitude'].to_numpy(), grids['longitude'].to_numpy()
    grid_ids = grids['grid_id'].to_numpy()

    records = []
    for _, t in group_info.iterrows():
        dist = haversine_km(t['latitude'], t['longitude'], grid_lat, grid_lon)
        for rank, gi in enumerate(np.argsort(dist)[:k], start = 1):
            records.append({'호기': t['호기'], 'rank': rank,
                            'grid_id': grid_ids[gi], 'dist_km': dist[gi]})
    nearest_info = pd.DataFrame(records)

    # ── 3) 격자별 파생변수 ──
    ldaps['ws10_ldaps'] = np.hypot(ldaps['heightAboveGround_10_10u'], ldaps['heightAboveGround_10_10v'])
    u50 = (ldaps['heightAboveGround_50_50MUmax'] + ldaps['heightAboveGround_50_50MUmin']) / 2
    v50 = (ldaps['heightAboveGround_50_50MVmax'] + ldaps['heightAboveGround_50_50MVmin']) / 2
    ldaps['ws50_ldaps'] = np.hypot(u50, v50)

    ws10_safe = np.clip(ldaps['ws10_ldaps'], 0.5, None)
    ws50_safe = np.clip(ldaps['ws50_ldaps'], 0.5, None)
    ldaps['alpha'] = np.log(ws50_safe / ws10_safe) / np.log(REF_HIGH / REF_LOW)
    ldaps['alpha'] = ldaps['alpha'].clip(0.05, 0.4)
    ldaps['ws_hub_ldaps'] = ldaps['ws50_ldaps'] * (hub_height / REF_HIGH) ** ldaps['alpha']

    ldaps['t_hub_ldaps'] = ldaps['heightAboveGround_2_t'] - LAPSE_RATE * (hub_height - REF_T)

    q = ldaps['heightAboveGround_2_q']
    tv_sfc = ldaps['heightAboveGround_2_t'] * (1 + 0.61 * q)
    tv_hub = ldaps['t_hub_ldaps'] * (1 + 0.61 * q)
    tv_mean = (tv_sfc + tv_hub) / 2
    p_hub = ldaps['surface_0_sp'] * np.exp(-G * hub_height / (R_D * tv_mean))
    ldaps['rho_hub_ldaps'] = p_hub / (R_D * tv_hub)

    ldaps['power_proxy_ldaps'] = (
        0.5 * ldaps['rho_hub_ldaps'] * swept_area * ldaps['ws_hub_ldaps'] ** 3 * cp / 1000
    )
    ldaps['gust50_ldaps'] = np.hypot(
        ldaps['heightAboveGround_50_50MUmax'] - ldaps['heightAboveGround_50_50MUmin'],
        ldaps['heightAboveGround_50_50MVmax'] - ldaps['heightAboveGround_50_50MVmin'],
    )

    # ── 4) 터빈별 IDW 가중치 ──
    weight = nearest_info[['호기', 'grid_id', 'dist_km']].copy()
    weight['weight'] = 1 / weight['dist_km']
    weight['weight'] = weight['weight'] / weight.groupby('호기')['weight'].transform('sum')

    # ── 5) 터빈 단위 축약 ──
    IDW_COLS = [
        'power_proxy_ldaps',
        'ws_hub_ldaps', 'ws10_ldaps', 'ws50_ldaps', 'alpha', 'gust50_ldaps',
        'heightAboveGround_10_10u', 'heightAboveGround_10_10v',
        'rho_hub_ldaps', 't_hub_ldaps',
        'heightAboveGround_2_r', 'heightAboveGround_2_q',
        'surface_0_sp', 'etc_0_blh',
        'surface_0_NDNSW', 'surface_0_NDNLW',
        'etc_0_hcc', 'etc_0_mcc', 'etc_0_lcc',
        'surface_0_lssrate', 'surface_0_SNOM',
        'heightAboveGround_5_XBLWS', 'heightAboveGround_5_YBLWS',
    ]
    merged = ldaps[['forecast_kst_dtm', 'grid_id'] + IDW_COLS].merge(
        weight[['호기', 'grid_id', 'weight']], on = 'grid_id', how = 'inner'
    )
    for col in IDW_COLS:
        merged[col] = merged[col] * merged['weight']
    turbine_level = merged.groupby(['forecast_kst_dtm', '호기'])[IDW_COLS].sum()

    # ── 6) 그룹 단위 축약 ──
    MEAN_COLS = [c for c in IDW_COLS if c != 'power_proxy_ldaps']
    agg_spec = {c: 'mean' for c in MEAN_COLS}
    agg_spec['power_proxy_ldaps'] = 'sum'

    grouped = turbine_level.groupby('forecast_kst_dtm')
    out = grouped.agg(agg_spec)
    out['ws_hub_std_ldaps'] = grouped['ws_hub_ldaps'].std()
    out = out.rename(columns = {'power_proxy_ldaps': f'group{kpx_group}_power_proxy_ldaps'})

    # ── 7) 풍향 ──
    wd_rad = np.arctan2(-out['heightAboveGround_10_10u'], -out['heightAboveGround_10_10v'])
    out['wd10_sin_ldaps'] = np.sin(wd_rad)
    out['wd10_cos_ldaps'] = np.cos(wd_rad)

    # ── 8) 누수 검증용 시각 ──
    avail = ldaps.groupby('forecast_kst_dtm')['data_available_kst_dtm'].first()
    out = out.join(avail).reset_index()

    # ── 9) 시간 파생변수 ──
    dtm = pd.to_datetime(out['forecast_kst_dtm'])
    out['hour'] = dtm.dt.hour
    out['month'] = dtm.dt.month

    return out
```

---

## 2. 사용법

```python
ldaps_group1_hourly      = preprocess_ldaps_group('../data/ldaps_train_group1.csv', kpx_group = 1)
ldaps_group1_hourly_test = preprocess_ldaps_group('../data/ldaps_test_group1.csv',  kpx_group = 1)
```

---

## 3. 단계별 설명

| 단계 | 내용 |
|---|---|
| 0 | CSV 로드. 터빈 제원(허브높이·로터직경)을 `info`에서 읽어 소인면적 계산 |
| 1 | 결측치 보간 (배치 내 시간 보간) |
| 2 | 터빈–격자 최근접 k개 매핑 (haversine) |
| 3 | 격자별 파생변수: ws10/ws50/alpha/ws_hub, t_hub, rho_hub, power_proxy, gust50 |
| 4 | 터빈별 IDW 가중치 (거리 역수, 터빈별 합=1) |
| 5 | 터빈 단위 축약 (시각 × 터빈) |
| 6 | 그룹 단위 축약 — power_proxy는 sum, 나머지 mean, ws_hub는 std 추가 |
| 7 | 풍향 sin/cos (성분 평균 후 atan2) |
| 8 | 누수 검증용 `data_available_kst_dtm` 결합 |
| 9 | 시간 파생변수 (hour, month) |

파생변수 계산식과 집계 방식의 물리적 근거는 [`feature_ldaps.md`](feature_ldaps.md)에 상세히 기록되어 있다.

---

## 4. 노트북 셀 코드 대비 변경점

기존 노트북은 train만 처리하고 상수를 하드코딩했다. 함수화하며 세 가지를 바꿨다.

### (1) 결측치 보간 — 신규 추가

test 데이터에 **752개 결측**이 존재한다 (3개 시각 × 16격자 × 여러 변수). 처리하지 않으면 해당 시각 예측이 전부 NaN이 된다.

**핵심: `data_available_kst_dtm`을 groupby 키에 포함한다.**
`grid_id`별로만 보간하면 앞뒤 값이 다른 예보 배치에서 올 수 있고, 다음 배치는 예측기준시점(전날 14:00) 이후 발표이므로 **누수**가 된다. 배치 안에서만 보간하면 안전하다.

결측 3개 시각은 모두 배치 중간(17/24, 18/24, 6/24번째)이고 앞뒤 값이 존재하여 배치 내 선형보간으로 전부 메워진다.

| 결측 시각 | 예보 배치 | 배치 내 위치 |
|---|---|---|
| 2025-04-08 17:00 | 2025-04-07 13:00 | 17/24 |
| 2025-06-18 18:00 | 2025-06-17 13:00 | 18/24 |
| 2025-07-18 06:00 | 2025-07-17 13:00 | 6/24 |

### (2) 터빈 제원을 `info`에서 읽음 — 하드코딩 제거

`HUB_HEIGHT = 117`, `ROTOR_D = 126`을 고정하지 않고 `group_info`에서 가져온다.
**Group 3는 UNISON U136으로 로터 136m(소인면적 16.5% 큼)** 이므로, 하드코딩 시 `kpx_group=3` 호출에서 조용히 틀린 값이 나온다.

### (3) `nearest_info` 내부 계산 — 중복 로드 제거

기존 `get_nearest_ldaps_grid`는 CSV(129MB)를 재로드했다. 이미 로드한 `ldaps`에서 격자 좌표를 뽑아 haversine 계산.

---

## 5. 검증 결과

| 항목 | train | test |
|---|---|---|
| shape | (26304, 30) | (8760, 30) |
| 결측 | 0 | 752 → **0** |
| 컬럼 동일 | ✅ (train == test) | |
| 기간 | 2022-01~2025-01 | 2025-01~2026-01 |

**기존 노트북 결과와 완전 일치** (train 첫 행):

| 값 | 함수 결과 | 노트북 값 |
|---|---|---|
| `group1_power_proxy_ldaps` | 16044.1694 | 16044.1694 |
| `ws_hub_std_ldaps` | 0.455818 | 0.455818 |

전체 소요 약 9.7초 (train + test).

---

## 6. 주의사항

- `cp = 0.35`는 트리 모델에서 상수배라 결과 무영향. 선형모델·NN 사용 시에만 의미.
- Group 2, 3에 그대로 호출 가능하나, [`feature_ldaps.md`](feature_ldaps.md)의 삭제 변수 목록·상관 수치는 **Group 1 기준**. 그룹별 재확인 필요.
- 출력의 `forecast_kst_dtm`, `data_available_kst_dtm`은 키 컬럼이므로 모델 입력에서 제외:
  ```python
  KEY_COLS = ['forecast_kst_dtm', 'data_available_kst_dtm']
  FEATURE_COLS = [c for c in out.columns if c not in KEY_COLS]
  ```

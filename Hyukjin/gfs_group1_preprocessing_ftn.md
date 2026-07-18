# GFS 전처리 함수 (`preprocess_gfs_group`)

GFS 격자별 long 데이터를 시각당 1행 wide 테이블로 변환. **LDAPS와 상보적**으로 설계 — LDAPS가 못 주는 신호(직접 80·100m 허브풍, 상층 종관풍)를 GFS에서 뽑는다.

- 입력: GFS CSV 경로, KPX 그룹 번호
- 출력: 시각당 1행 DataFrame (Group 1 기준 22컬럼)
- 관련 문서: [`ldaps_group1_preprocessing_ftn.md`](ldaps_group1_preprocessing_ftn.md), [`feature_ldaps.md`](feature_ldaps.md)

---

## 1. LDAPS와의 근본적 차이

| | LDAPS | GFS |
|---|---|---|
| 해상도 | 1.5 km, 16격자 조밀 | 25 km, 9격자 (3×3 정규격자) |
| 격자 배치 | 터빈 위에 얹힘 | 터빈을 넓게 감쌈 |
| 집계 단위 | 터빈별 IDW → 그룹 통계 | **centroid 1점 IDW 보간** |
| power_proxy | 터빈 6개 합(sum) | centroid 단일값 |
| std/min/max | 있음 (터빈 간 편차) | **없음** (1점이라 편차 개념 소멸) |
| 고유 강점 | 국지 지형 정밀도 | 직접 80·100m 풍속 + 상층 종관 |

**핵심**: GFS는 25 km 격자라 터빈 6개(1 km 군집)를 구분하지 못한다. 따라서 터빈별로 나누지 않고, **터빈 군집 중심(centroid) 1점**으로 보간한다. 이 때문에 `std`는 계산할 수 없고 의미도 없다.

### 감싸는 격자와 centroid 위치

Group 1 centroid = (37.2871, 128.9520). GFS 3×3 격자 중 이 점을 감싸는 사각형은 **격자 1, 2, 4, 5**:

```
grid1(37.50,128.75)   grid2(37.50,129.00)
          ┌─────────────────┐
          │   ● centroid    │   ● = (37.287, 128.952)
          └─────────────────┘
grid4(37.25,128.75)   grid5(37.25,129.00)
```

centroid까지 거리와 IDW 가중치:

| 격자 | 거리 | IDW 가중치 |
|---|---:|---:|
| 5 | 5.92 km | 0.565 |
| 4 | 18.35 km | 0.182 |
| 2 | 24.05 km | 0.139 |
| 1 | 29.64 km | 0.113 |

grid 5가 가장 가깝지만 남서로 치우쳐 있어(centroid가 격자 5보다 북서), 감싸는 4격자 보간이 grid 5 단독보다 터빈 위치를 정확히 짚는다. 나머지 격자(3=바다, 6~9=원거리)는 제외.

---

## 2. label 상관 실측 (유효시간대, Spearman)

centroid IDW 집계 후 label과의 상관. **GFS만의 강점을 정량 확인**한 결과.

| 변수 | 상관 | 비고 |
|---|---:|---|
| `isobaricInhPa_850_u` | **+0.6304** | **지상풍보다 강함** — GFS 최대 강점 |
| `ws850` (850hPa 풍속) | +0.6256 | 상층 종관풍 |
| `ws10` | +0.5572 | LDAPS와 동일 패턴 (아래 참조) |
| `ws100` (100m) | +0.5364 | 허브 근접 직접풍 |
| `ws80` (80m) | +0.5372 | |
| `ws_hub` (80/100 보간) | +0.5362 | |
| `ws_pbl` (경계층풍) | +0.4670 | |
| `surface_0_gust` | +0.4913 | 돌풍 |
| `ws700` | +0.3483 | 중층 종관 |
| `isobaricInhPa_500_gh` | -0.2551 | 500hPa 지위고도 (기압계) |
| `heightAboveGround_2_2t` | -0.2734 | 기온 |
| `surface_0_sp` | -0.0520 | 신호 없음 → 제외 |
| `meanSea_0_prmsl` | +0.0667 | 신호 없음 → 제외 |
| `surface_0_prate` / `tp` | ≈ -0.07 | 강수, 약함 → 제외 |
| `lowCloudLayer_0_lcc` | -0.0507 | 약함 → 제외 |

### 핵심 발견 1: 850hPa 바람이 지상풍보다 강력

850hPa(약 1.5 km 고도)는 허브(117m)보다 훨씬 높은데도 label 상관이 가장 높다(0.63). GFS는 25 km 광역 모델이라 **경계층 잡음이 적은 상층에서 대규모 흐름을 가장 깨끗하게 표현**하기 때문. 이 상층 종관풍이 몇 시간 뒤 지상풍의 **선행지표**가 된다. LDAPS에는 없는 정보라 GFS를 쓰는 핵심 이유.

### 핵심 발견 2: 여기서도 ws10 > ws_hub

LDAPS와 동일하게 GFS에서도 생짜 `ws10`(0.557)이 허브 보간 `ws_hub`(0.536)보다 상관이 높다. 80/100m 전단 외삽도 잡음을 더한다. **직접 제공되는 `ws100`을 신뢰하되, 외삽 `ws_hub`를 맹신하지 말 것.**

---

## 3. 함수 전체 코드

```python
def preprocess_gfs_group(
        gfs_path: str,
        kpx_group: int,
        use_grids = (1, 2, 4, 5),
        cp: float = 0.35,
        info_path: str = "../data/info_include_lat_lon.csv",
) -> pd.DataFrame:
    """GFS 격자별 long 데이터 -> centroid 1점 IDW 집계 -> 시각당 1행.
    LDAPS와 상보적: 직접 80/100m 풍속 + 상층 종관풍 위주.
    """
    REF_T = 2.0
    LAPSE_RATE, G, R_D = 0.0065, 9.80665, 287.05

    gfs = pd.read_csv(gfs_path, encoding='utf-8-sig')
    info = pd.read_csv(info_path, encoding='utf-8-sig')
    group_info = info[info['KPX그룹'] == kpx_group].copy()
    hub_height = float(group_info['Hub Height(m)'].iloc[0])
    rotor_d    = float(group_info['Rotor Diameter(m)'].iloc[0])
    swept_area = np.pi * (rotor_d / 2) ** 2
    lat_c = group_info['latitude'].mean()   # 터빈 군집 중심
    lon_c = group_info['longitude'].mean()

    # ── 0) 결측 보간 (배치 내) — 안전장치 ──
    gfs = gfs.sort_values(['grid_id', 'data_available_kst_dtm', 'forecast_kst_dtm']).reset_index(drop=True)
    num_cols = [c for c in gfs.select_dtypes(include=[np.number]).columns if c != 'grid_id']
    gfs[num_cols] = gfs.groupby(['grid_id', 'data_available_kst_dtm'])[num_cols].transform(
        lambda g: g.interpolate(method='linear', limit_direction='both'))

    # ── 1) 감싸는 격자만 ──
    gfs = gfs[gfs['grid_id'].isin(use_grids)].copy()

    # ── 2) 격자별 파생 ──
    gfs['gfs_ws10']  = np.hypot(gfs['heightAboveGround_10_10u'],   gfs['heightAboveGround_10_10v'])
    gfs['gfs_ws80']  = np.hypot(gfs['heightAboveGround_80_u'],     gfs['heightAboveGround_80_v'])
    gfs['gfs_ws100'] = np.hypot(gfs['heightAboveGround_100_100u'], gfs['heightAboveGround_100_100v'])
    # 80/100m 두 점으로 전단지수 -> 허브 117m (100m 살짝 위, 짧은 외삽)
    a = np.log(np.clip(gfs['gfs_ws100'], 0.5, None) / np.clip(gfs['gfs_ws80'], 0.5, None)) / np.log(100/80)
    a = a.clip(0.05, 0.4)
    gfs['gfs_alpha'] = a
    gfs['gfs_ws_hub'] = gfs['gfs_ws100'] * (hub_height / 100.0) ** a

    # 상층 종관풍 (GFS 고유)
    gfs['gfs_ws850']  = np.hypot(gfs['isobaricInhPa_850_u'], gfs['isobaricInhPa_850_v'])
    gfs['gfs_ws700']  = np.hypot(gfs['isobaricInhPa_700_u'], gfs['isobaricInhPa_700_v'])
    gfs['gfs_ws_pbl'] = np.hypot(gfs['planetaryBoundaryLayer_0_u'], gfs['planetaryBoundaryLayer_0_v'])

    # 공기밀도 (GFS 2t/2sh/sp)
    gfs['gfs_t_hub'] = gfs['heightAboveGround_2_2t'] - LAPSE_RATE * (hub_height - REF_T)
    q = gfs['heightAboveGround_2_2sh']
    tv_sfc = gfs['heightAboveGround_2_2t'] * (1 + 0.61 * q)
    tv_hub = gfs['gfs_t_hub'] * (1 + 0.61 * q)
    tv_mean = (tv_sfc + tv_hub) / 2
    p_hub = gfs['surface_0_sp'] * np.exp(-G * hub_height / (R_D * tv_mean))
    gfs['gfs_rho_hub'] = p_hub / (R_D * tv_hub)

    gfs['gfs_power_proxy'] = 0.5 * gfs['gfs_rho_hub'] * swept_area * gfs['gfs_ws_hub'] ** 3 * cp / 1000

    # ── 3) centroid IDW 가중치 ──
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0088
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    gd = gfs.drop_duplicates('grid_id')[['grid_id', 'latitude', 'longitude']]
    dist = {int(r.grid_id): haversine_km(lat_c, lon_c, r.latitude, r.longitude) for _, r in gd.iterrows()}
    inv = {g: 1/dd for g, dd in dist.items()}
    s = sum(inv.values())
    wmap = {g: v/s for g, v in inv.items()}
    gfs['w'] = gfs['grid_id'].map(wmap)

    # ── 4) centroid 1점으로 IDW 집계 ──
    AGG_COLS = [
        'gfs_power_proxy',
        'gfs_ws_hub', 'gfs_ws100', 'gfs_ws80', 'gfs_ws10', 'gfs_alpha',
        'gfs_ws850', 'gfs_ws700', 'gfs_ws_pbl',
        'gfs_rho_hub', 'gfs_t_hub',
        'surface_0_gust', 'planetaryBoundaryLayer_0_VRATE',
        'isobaricInhPa_500_gh', 'isobaricInhPa_850_t', 'atmosphere_0_tcc',
        # 풍향 계산용 성분 (100m=허브 근접, 850=종관)
        'heightAboveGround_100_100u', 'heightAboveGround_100_100v',
        'isobaricInhPa_850_u', 'isobaricInhPa_850_v',
    ]
    for c in AGG_COLS:
        gfs[c] = gfs[c] * gfs['w']
    out = gfs.groupby('forecast_kst_dtm')[AGG_COLS].sum()

    # ── 5) 풍향 sin/cos (100m 허브풍 + 850hPa 종관풍) ──
    wd100 = np.arctan2(-out['heightAboveGround_100_100u'], -out['heightAboveGround_100_100v'])
    out['gfs_wd100_sin'] = np.sin(wd100)
    out['gfs_wd100_cos'] = np.cos(wd100)
    wd850 = np.arctan2(-out['isobaricInhPa_850_u'], -out['isobaricInhPa_850_v'])
    out['gfs_wd850_sin'] = np.sin(wd850)
    out['gfs_wd850_cos'] = np.cos(wd850)
    out = out.drop(columns=['heightAboveGround_100_100u', 'heightAboveGround_100_100v',
                            'isobaricInhPa_850_u', 'isobaricInhPa_850_v'])

    # ── 6) 누수 검증용 시각 ──
    avail = gfs.groupby('forecast_kst_dtm')['data_available_kst_dtm'].first()
    out = out.join(avail).reset_index()
    return out
```

---

## 4. 사용법

```python
gfs_group1_hourly      = preprocess_gfs_group('../data/gfs_train_group1.csv', kpx_group = 1)
gfs_group1_hourly_test = preprocess_gfs_group('../data/gfs_test_group1.csv',  kpx_group = 1)
```

LDAPS 테이블과 `forecast_kst_dtm` 기준으로 join하여 최종 학습 테이블 구성:

```python
train = ldaps_group1_hourly.merge(
    gfs_group1_hourly.drop(columns='data_available_kst_dtm'),
    on='forecast_kst_dtm', how='inner')
```

---

## 5. LDAPS 파이프라인 대비 차이

| 단계 | LDAPS | GFS |
|---|---|---|
| 터빈 매핑 | 터빈별 최근접 k격자 | **centroid 1점** |
| 가중치 | 터빈별 IDW (터빈별 합=1) | centroid까지 IDW (전체 합=1) |
| 축약 | 터빈 단위 → 그룹 sum/mean/std | **centroid 1점 (단일 IDW)** |
| power_proxy | 터빈 6개 sum | centroid 단일값 |
| std | `ws_hub_std` 있음 | **없음** |
| 풍향 | 10m 1종 | 100m(허브) + 850hPa(종관) 2종 |
| 상층변수 | 없음 | 850/700/500hPa 종관풍 |
| 허브풍 | 50m 외삽 | **80/100m 직접 → 짧은 외삽** |
| 시간변수 | hour, month 추가 | **미추가** (LDAPS 테이블에서 이미 제공) |

**시간변수를 GFS에서 만들지 않는 이유**: 최종 테이블에서 LDAPS와 join하면 `hour`/`month`가 중복된다. 한쪽(LDAPS)에서만 생성.

---

## 6. 채택/제외 변수

### 채택 (GFS 고유 강점 위주)

| 분류 | 컬럼 |
|---|---|
| 타깃 근사 | `gfs_power_proxy` |
| 허브 직접풍 | `gfs_ws_hub`, `gfs_ws100`, `gfs_ws80`, `gfs_ws10`, `gfs_alpha` |
| **상층 종관풍** | `gfs_ws850`, `gfs_ws700` |
| 경계층 | `gfs_ws_pbl`, `planetaryBoundaryLayer_0_VRATE` |
| 돌풍 | `surface_0_gust` |
| 종관 패턴 | `isobaricInhPa_500_gh`, `isobaricInhPa_850_t` |
| 열역학 | `gfs_rho_hub`, `gfs_t_hub` |
| 운량 | `atmosphere_0_tcc` |
| 풍향 | `gfs_wd100_sin/cos`, `gfs_wd850_sin/cos` |

### 제외

| 변수 | 근거 |
|---|---|
| `surface_0_sp`, `meanSea_0_prmsl` | label 상관 ≈ 0 |
| `surface_0_prate`, `surface_0_tp` | 강수, 상관 ≈ -0.07 |
| `lowCloudLayer_0_lcc`, `middleCloudLayer_0_mcc`, `highCloudLayer_0_hcc` | 약함. `tcc`로 대표 |
| `surface_0_dswrf`, `surface_0_dlwrf` | 복사. 풍력에 약함 |
| `heightAboveGround_2_2d/2r/2sh` | rho 계산 후 중복 |
| `isobaricInhPa_500_u/v/t`, `700_v` | 500hPa 풍속 약함(0.20). gh만 유지 |

**LDAPS와 중복되는 지상변수(2t, sp, 복사, 운량)는 GFS 쪽을 과감히 버린다.** LDAPS가 1.5 km로 더 정밀하므로, GFS는 자기만의 강점(직접 허브풍 + 상층 종관)에 집중.

---

## 7. 검증 결과

| 항목 | train | test |
|---|---|---|
| shape | (26304, 22) | (8760, 22) |
| 결측 | 0 | 0 (원래 결측 없음) |
| 컬럼 동일 | ✅ | |
| 풍향 sin²+cos² | = 1 | |
| 소요 | 약 5.5초 (train+test) | |

GFS test는 결측이 없지만, 0단계 보간은 LDAPS와 동일하게 무조건 실행된다(결측 없으면 no-op). train/test 코드 경로 통일 목적.

---

## 8. 주의사항

- `use_grids=(1,2,4,5)`는 **Group 1 centroid 기준**. Group 2, 3는 centroid가 달라 감싸는 격자가 바뀔 수 있으니 지도/거리로 재확인.
- `cp`는 트리 모델에서 상수배라 무영향.
- 상층풍(850hPa)이 최강 신호지만 **선행지표라 리드타임에 민감**. 예측기준시점(전날 14:00) 제약은 `data_available_kst_dtm`으로 이미 반영됨.
- 키 컬럼(`forecast_kst_dtm`, `data_available_kst_dtm`)은 모델 입력에서 제외.

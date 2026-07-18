# baseline_lgbm.ipynb 설명서

이 문서는 `baseline_lgbm.ipynb`의 전체 실행 흐름, 설정값, 전처리 방식, 검증 전략, 모델 학습 방식, 진단 산출물, 최종 제출 파일 생성 과정을 정리한 설명서입니다.

대상 노트북:

- `baseline_lgbm.ipynb`

현재 노트북의 기본 목적은 다음과 같습니다.

- LDAPS/GFS 기상예보 데이터를 그룹별 풍력발전량 예측용 feature로 변환합니다.
- LightGBM 기반 모델을 그룹별로 학습합니다.
- `basic` 또는 `ensemble` 방식 중 하나를 선택해 발전량을 예측합니다.
- validation score, residual, feature importance, permutation importance, SHAP을 계산합니다.
- 최종적으로 2025년 test 예측 결과를 `baseline_submit.csv`로 저장합니다.

## 1. 전체 실행 흐름

노트북은 아래 순서로 실행되도록 구성되어 있습니다.

1. 패키지 설치 및 import
2. 실험 Configuration 설정
3. 데이터 로드
4. Feature Engineering
5. Metric, Validation, Feature Selection, Grid Search utility 정의
6. LightGBM 모델 학습 및 Grid Search
7. Validation 진단
8. Final Train 및 Submission 생성

각 단계는 앞 단계에서 만든 객체를 그대로 사용하므로, 원칙적으로 위에서 아래로 순서대로 실행해야 합니다.

## 2. 실행 환경

### 2.1 패키지 설치

노트북 초반에서 아래 패키지가 설치되어 있는지 확인하고, 없으면 `pip install`을 수행합니다.

- `lightgbm`
- `shap`
- `seaborn`

Colab GPU 환경에서는 LightGBM GPU 학습을 위해 NVIDIA OpenCL ICD 파일을 자동으로 설정하려고 시도합니다.

### 2.2 주요 import

주요 라이브러리는 다음과 같습니다.

- `numpy`
- `pandas`
- `lightgbm`
- `shap`
- `matplotlib`
- `seaborn`

SHAP 또는 plotting 라이브러리 import가 실패하면, 해당 기능만 건너뛰도록 `SHAP_AVAILABLE`, `PLOT_AVAILABLE` 플래그를 둡니다.

## 3. Configuration 구조

Configuration은 `2.1`부터 `2.10`까지 하위 섹션으로 나뉘어 있습니다.

### 3.1 Runtime Basics

주요 설정:

- `RANDOM_STATE = 42`

역할:

- feature selection sampling
- grid search 조합 sampling
- random validation split
- SHAP sampling

등에서 재현성을 유지합니다.

### 3.2 Paths And Artifact Locations

Colab과 로컬 환경을 모두 지원합니다.

기본 경로:

- Colab: `/content/drive/MyDrive/BARAM2026/data`
- 로컬: `C:\Users\심현석\Documents\BARAM 2026`

주요 디렉터리:

| 변수 | 의미 |
| --- | --- |
| `TRAIN_DIR` | train 데이터 폴더 |
| `TEST_DIR` | test 데이터 폴더 |
| `OUTPUT_DIR` | 모든 산출물 저장 루트 |
| `FEATURE_OUTPUT_DIR` | 전처리 feature 목록 및 inventory 저장 |
| `EXP_DIR` | 모델링 실험 결과 저장 |
| `CHECKPOINT_DIR` | grid search, validation 결과, summary 저장 |
| `PLOT_DIR` | residual plot 저장 |
| `IMPORTANCE_DIR` | LightGBM feature importance, permutation importance 저장 |
| `SHAP_DIR` | SHAP importance 및 summary plot 저장 |
| `FEATURE_SELECTION_DIR` | feature selection cache 저장 |

주요 파일 경로:

| 변수 | 저장 파일 |
| --- | --- |
| `GRID_RESULT_PATH` | grid search 요약 결과 |
| `GRID_FOLD_RESULT_PATH` | fold별 grid search 결과 |
| `BEST_SUMMARY_PATH` | 그룹별 best parameter 요약 |
| `VALID_PRED_PATH` | validation prediction long table |
| `VALID_METRIC_PATH` | validation metric table |
| `RESIDUAL_LONG_PATH` | residual 분석용 long table |
| `WRONG_CASE_PATH` | ensemble classifier 오분류 case |
| `SUBMISSION_PATH` | 최종 제출 파일 |
| `FINAL_SUMMARY_PATH` | final train/test 예측 summary |
| `FEATURE_IMPORTANCE_PATH` | LightGBM built-in importance |
| `PERMUTATION_IMPORTANCE_PATH` | permutation importance |
| `SHAP_IMPORTANCE_PATH` | SHAP importance |

### 3.3 Competition Constants

예측 대상:

```python
TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
```

설비용량:

```python
CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}
```

이 값은 NMAE, FICR, clipping, sample weight 계산에 사용됩니다.

### 3.4 Preprocessing Feature Settings

집계 방식:

```python
WEATHER_AGGS = ["mean", "std", "max"]
SUBSET_AGGS = ["mean", "std", "max"]
```

LDAPS/GFS grid별 예보값을 시간 단위로 만들 때 평균, 표준편차, 최댓값을 사용합니다.

강수 bin:

```python
PRECIP_BIN_FEATURES = [
    ("precip_0_1mm", 0.0, 1.0),
    ("precip_1_3mm", 1.0, 3.0),
    ("precip_3_15mm", 3.0, 15.0),
]
```

15mm 이상은 reference bin처럼 두고 별도 flag를 만들지 않습니다.

그룹별 LDAPS raw grid:

| 그룹 | LDAPS raw grid |
| --- | --- |
| `kpx_group_1` | 5, 6 |
| `kpx_group_2` | 6, 11 |
| `kpx_group_3` | 6, 12 |

그룹별 LDAPS subset aggregation grid:

| 그룹 | subset aggregation |
| --- | --- |
| `kpx_group_1` | 5, 6, 10, 11 |
| `kpx_group_2` | 6, 7, 11, 12 |
| `kpx_group_3` | 6, 11, 12, 16 |

GFS raw grid:

```python
GFS_RAW_GRID_IDS = [3, 5]
```

Lag/Rolling:

```python
WEATHER_LAGS = [1, 2, 24]
WEATHER_ROLL_WINDOWS = [3, 6, 24]
```

각 시간의 예보 feature에 대해 1시간 전, 2시간 전, 24시간 전 lag와 3/6/24시간 rolling mean/std를 만듭니다.

주의:

- rolling은 현재 시점을 포함하지 않도록 `shift(1)` 후 계산합니다.
- 즉, rolling feature는 해당 시점 이전 feature만 사용합니다.

### 3.5 Validation And Modeling Mode

Validation option:

```python
VALIDATION_OPTION = "random"
```

지원하는 옵션:

- `full`
- `sliding`
- `expanding`
- `random`

Modeling mode:

```python
MODELING_MODE = "ensemble"
```

지원하는 옵션:

| 모드 | 설명 |
| --- | --- |
| `basic` | LightGBM regressor 하나로 발전량을 직접 예측 |
| `ensemble` | zero classifier, capacity classifier, regressor를 결합 |

`ensemble` 모드의 기준값:

```python
ZERO_THRESHOLD = 0.80
CAPACITY_THRESHOLD = 0.98
ZERO_LABEL_KWH = 100.0
CAPACITY_LABEL_RATIO = 0.98
```

의미:

- Model I: 발전량이 100kWh 미만인지 분류
- Model II: 발전량이 capacity의 98% 이상인지 분류
- Model III: 발전량 자체를 회귀 예측

### 3.6 Run Switches

주요 실행 스위치:

| 변수 | 의미 |
| --- | --- |
| `RUN_GRID_SEARCH` | grid search 실행 여부 |
| `RESUME_GRID_SEARCH` | 기존 grid checkpoint가 있으면 완료된 run을 건너뛸지 여부 |
| `FORCE_RERUN_GRID_SEARCH` | 기존 grid result 파일 삭제 후 다시 실행할지 여부 |
| `RUN_FINAL_TRAIN_AND_SUBMIT` | 최종 학습 및 submission 생성 여부 |
| `RUN_DIAGNOSTICS` | residual plot 생성 여부 |
| `RUN_BUILTIN_FEATURE_IMPORTANCE` | LightGBM 내장 importance 계산 여부 |
| `RUN_PERMUTATION_IMPORTANCE` | permutation importance 계산 여부 |
| `RUN_SHAP` | SHAP 계산 여부 |
| `RUN_SHAP_FEATURE_SELECTION` | feature selection을 수행할지 여부 |
| `REUSE_FEATURE_SELECTION_CACHE` | feature selection cache 재사용 여부 |
| `FORCE_RERUN_FEATURE_SELECTION` | feature selection을 강제로 다시 할지 여부 |
| `ALLOW_LEGACY_FEATURE_SELECTION_CACHE` | 예전 형식의 feature selection cache 허용 여부 |

중요:

- `FORCE_RERUN_GRID_SEARCH=True`이면 기존 grid-search 결과 파일을 삭제합니다.
- 이전 결과를 이어서 쓰고 싶으면 `FORCE_RERUN_GRID_SEARCH=False`, `RESUME_GRID_SEARCH=True`로 두는 것이 자연스럽습니다.

### 3.7 Feature Selection And Diagnostics

기본값:

```python
MAX_GRID_COMBOS_PER_GROUP = 12
FEATURE_SELECTION_TOPK_IMPORTANCE = 5000
FEATURE_SELECTION_TOPK_SHAP = 1500
FEATURE_SELECTION_SHAP_SAMPLE_SIZE = 2000
```

Feature selection은 2단계입니다.

1. LightGBM feature importance로 top 5000 후보 선정
2. 그 후보에 대해 SHAP을 계산하여 top 1500 feature 선정

진단 관련 설정:

| 변수 | 의미 |
| --- | --- |
| `CASE_EXPORT_MAX_FEATURES` | wrong case 저장 시 포함할 feature 수 |
| `PERMUTATION_MAX_FEATURES` | permutation importance 후보 feature 수 |
| `PERMUTATION_SAMPLE_SIZE` | permutation importance sample row 수 |
| `PERMUTATION_N_REPEATS` | feature별 permutation 반복 횟수 |
| `SHAP_SAMPLE_SIZE` | 진단용 SHAP sample row 수 |
| `SHAP_MAX_DISPLAY` | SHAP summary plot에 표시할 feature 수 |

### 3.8 LightGBM Runtime, Objective, And Weighting

LightGBM objective:

```python
REGRESSOR_OBJECTIVE = "l1"
```

지원 objective:

- `l1`
- `l2`
- `huber`

Sample weight:

```python
USE_POWER_SAMPLE_WEIGHT = True
MIN_NON_EVAL_WEIGHT = 0.35
HIGH_POWER_WEIGHT_STRENGTH = 6.0
```

회귀 모델 학습 시 발전량이 높은 시간대에 더 큰 weight를 부여합니다.

weight 계산 개념:

```text
ratio = actual / capacity
비평가 구간(ratio < 0.10): MIN_NON_EVAL_WEIGHT
평가 구간(ratio >= 0.10): 1.0
추가 고발전량 weight: HIGH_POWER_WEIGHT_STRENGTH * min(ratio, 1)^2
```

즉, 실제 발전량이 높을수록 모델이 더 신경 써서 맞추도록 유도합니다.

### 3.9 LightGBM Parameter Grids

Model I zero classifier:

- 기본적으로 parameter set 1개만 사용합니다.
- 필요하면 `ZERO_CLASSIFIER_PARAM_GRID`에 dict를 추가하여 확장할 수 있습니다.

Model II capacity classifier:

- `CAPACITY_CLASSIFIER_PARAM_GRID`에 여러 parameter set이 있으며, `ensemble` 모드에서 grid search 대상입니다.

Model III/basic regressor:

- `REGRESSOR_PARAM_GRID`에서 조합을 생성합니다.
- `MAX_GRID_COMBOS_PER_GROUP`만큼 sampling하여 group별 grid search를 수행합니다.

## 4. 데이터 로드

필수 입력 파일:

| 파일 | 설명 |
| --- | --- |
| `train/train_labels.csv` | 시간별 실제 발전량 label |
| `train/ldaps_train.csv` | train LDAPS 예보 |
| `train/gfs_train.csv` | train GFS 예보 |
| `test/ldaps_test.csv` | test LDAPS 예보 |
| `test/gfs_test.csv` | test GFS 예보 |
| `sample_submission.csv` | 제출 양식 |

시간 컬럼 처리:

- `kst_dtm`
- `forecast_kst_dtm`
- `data_available_kst_dtm`

모두 `datetime`으로 변환합니다.

Group 3 주의사항:

- group 3은 2022년 발전량 데이터가 없는 것이 정상입니다.
- 노트북에서는 2023년 이후 group 3 결측 여부를 별도로 확인합니다.
- 학습/검증 mask는 target이 NaN인 row를 제외하므로, group 3의 2022년 NaN은 모델 학습에 들어가지 않습니다.

## 5. Feature Engineering

### 5.1 Drop Columns

예측에 덜 직접적이거나 불필요하다고 판단한 일부 기상변수를 제거합니다.

LDAPS 제거 예:

- 단파/장파복사 관련 변수
- 운량 관련 변수
- 강설/적설/융설 관련 변수
- land/sea mask
- 고도 관련 변수

GFS 제거 예:

- 단파/장파복사 flux
- 강수율/총강수량 일부
- 운량 관련 변수

단, LDAPS `surface_0_ncpcp`는 강수 bin feature를 만들기 위해 유지합니다.

### 5.2 Physical Value Corrections

물리적으로 불가능한 값을 보정합니다.

보정 내용:

- 상대습도는 100%를 넘지 않도록 clipping
- 이슬점온도가 기온보다 높으면 기온으로 보정

적용 대상:

- LDAPS train/test
- GFS train/test

### 5.3 Wind Physics Features

LDAPS와 GFS의 u/v 바람 성분으로 풍속과 방향 관련 feature를 만듭니다.

공통 생성 feature:

- `wind_speed`
- `wind_speed_sq`
- `wind_speed_cubed`
- `wind_speed_pow_neg2`
- `wind_speed_pow_neg3`
- `angle_sin`
- `angle_cos`

여기서 `angle_sin`, `angle_cos`는 `arctan2(v, u)` 기반의 벡터 방향 표현입니다.

LDAPS 대상 바람:

- 10m wind
- 50m max wind
- 50m min wind
- 5m X/Y boundary layer wind

GFS 대상 바람:

- 10m wind
- 80m wind
- 100m wind
- planetary boundary layer wind
- 850hPa wind
- 700hPa wind
- 500hPa wind

### 5.4 Humid Air Density

습도를 고려한 공기밀도를 계산합니다.

공식:

```text
humid_air_density =
    surface_pressure / {287.05 * temperature_K * (1 + 0.61 * specific_humidity)}
```

추가로 역수 feature도 생성합니다.

- `humid_air_density`
- `humid_air_density_inv`

### 5.5 Wind Power Proxy

풍력 발전량의 물리적 구조를 반영하기 위해 power proxy를 만듭니다.

기본 형태:

```text
0.5 * humid_air_density * swept_area * effective_speed^3
```

effective speed 처리:

- cut-in 미만: 0
- rated 이상 cut-out 미만: rated speed로 고정
- cut-out 이상: 0

그룹별 발전기 사양:

| 그룹 | cut-in | rated | cut-out | rotor radius |
| --- | --- | --- | --- | --- |
| group 1 | 3.0 | 12.0 | 22.5 | 63m |
| group 2 | 3.0 | 12.0 | 22.5 | 63m |
| group 3 | 3.0 | 12.5 | 22.0 | 68m |

### 5.6 Wind Regime Flags

각 풍속 변수에 대해 풍속 구간 flag를 생성합니다.

생성 구간:

- `below_cutin`
- `cutin_to_rated`
- `rated_to_cutout`

`cutout 이상` 구간은 별도 column으로 만들지 않습니다. 회귀분석 관점에서 reference category처럼 남기는 구조입니다.

### 5.7 Precipitation Bin Flags

LDAPS `surface_0_ncpcp`를 사용하여 강수량 bin flag를 만듭니다.

생성 flag:

- `precip_0_1mm`
- `precip_1_3mm`
- `precip_3_15mm`

15mm 이상은 생략된 기준 구간입니다.

### 5.8 BLH Difference

LDAPS `etc_0_blh`에 대해 grid별 시간 순서 차분을 만듭니다.

생성 feature:

- `etc_0_blh_diff1`
- `etc_0_blh_diff2`

경계층 높이의 변화량을 반영하기 위한 feature입니다.

### 5.9 GFS Pressure Difference

GFS grid 3과 grid 5의 지표면 기압 차이를 만듭니다.

```text
gfs_grid3_surface_pressure - gfs_grid5_surface_pressure
```

생성 feature 이름:

- `gfs_g03_minus_g05_surface_0_sp_pressure_diff`

### 5.10 Spatial Features

기상 데이터는 grid별 행으로 주어지므로, 모델 입력을 시간별 한 행으로 바꿉니다.

구성:

1. LDAPS all-grid aggregation
2. GFS all-grid aggregation
3. GFS raw grid 3, 5 wide feature
4. GFS grid 3-5 pressure difference
5. LDAPS group-specific raw grid wide feature
6. LDAPS group-specific subset aggregation

All-grid aggregation:

- 모든 LDAPS 또는 GFS grid를 `forecast_kst_dtm` 단위로 묶고 `mean/std/max`를 계산합니다.

Raw grid wide:

- 특정 grid의 원본 feature를 `forecast_kst_dtm` 단위 wide column으로 펼칩니다.
- 예: `ldaps_g05_...`, `ldaps_g06_...`

Subset aggregation:

- 그룹별로 지정한 LDAPS grid subset에 대해 `mean/std/max`를 계산합니다.

### 5.11 Calendar Features

시간 주기성을 반영하기 위해 sin/cos feature를 만듭니다.

생성 feature:

- `hour_sin`, `hour_cos`
- `day_sin`, `day_cos`
- `month_sin`, `month_cos`
- `season_sin`, `season_cos`

### 5.12 Lag/Rolling Features

`choose_lag_roll_cols()`가 lag/rolling을 만들 후보 feature를 고릅니다.

후보 우선순위:

- wind/speed/gust/power_proxy
- precip
- pressure_diff
- blh
- planetaryBoundaryLayer
- VRATE
- humid_air_density
- surface pressure
- raw grid feature
- subset aggregation feature
- wind regime flag

생성 방식:

- lag: `1`, `2`, `24`
- rolling mean/std: `3`, `6`, `24`

중요:

- rolling은 `shift(1)` 후 계산하므로 현재 시점 feature를 rolling에 직접 포함하지 않습니다.
- target 발전량 lag는 만들지 않습니다. 즉, 실제 발전량 history를 feature로 쓰지 않습니다.

### 5.13 Feature Inventory 저장

그룹별 feature 생성 후 아래 파일이 저장됩니다.

- `{target}_feature_columns.csv`
- `{target}_lag_rolling_base_columns.csv`
- `feature_inventory.csv`

이 파일들은 어떤 feature가 생성되었는지 확인할 때 사용합니다.

## 6. Metric

### 6.1 Group Metric

`group_metric()`은 그룹별 score를 계산합니다.

평가 대상 row:

```text
actual >= capacity * 0.10
```

즉, 실제 발전량이 설비용량의 10% 이상인 시간만 평가에 사용합니다.

계산 항목:

- `group_nmae`
- `nmae_eval`
- `one_minus_nmae`
- `ficr`
- `selection_score`
- `mae_all_kwh`
- `mae_eval_kwh`
- `ficr_6pct_ratio`
- `ficr_8pct_ratio`
- `ficr_fail_ratio`
- `kge`

`selection_score`:

```text
selection_score = 0.5 * (1 - group_nmae) + 0.5 * ficr
```

### 6.2 Official-like Metric

`official_like_metric_from_long()`은 세 그룹의 결과를 합쳐 대회 산식과 유사한 전체 score를 계산합니다.

```text
1-NMAE_g = 1 - 평균(group_nmae)
FICR_g = 평균(group_ficr)
total_score = 0.5 * (1-NMAE_g) + 0.5 * FICR_g
```

### 6.3 KGE

`kling_gupta_efficiency()`도 계산합니다.

KGE는 대회 공식 점수는 아니지만, 예측값의 다음 특성을 함께 확인하기 위해 사용합니다.

- correlation
- variance ratio
- mean ratio

## 7. Validation Strategy

`VALIDATION_OPTION`으로 검증 방식을 선택합니다.

### 7.1 full

훈련:

- 2024-01-01 00:00:00 이전 labeled row

검증:

- 2024-01-01 01:00:00부터 2025-01-01 00:00:00까지

### 7.2 sliding

검증 구간은 2024년 분기별입니다.

Fold:

- 2024 Q1 시작 전까지 365일 학습, Q1 검증
- 2024 Q2 시작 전까지 365일 학습, Q2 검증
- 2024 Q3 시작 전까지 365일 학습, Q3 검증
- 2024 Q4 시작 전까지 365일 학습, Q4 검증

각 fold에서 validation 시작 직전 365일을 train window로 사용합니다.

예:

```text
[ train ][ val ] 
        [ train ][ val ]
                [ train ][ val ]
                        [ train ][ val ]
```

### 7.3 expanding

검증 구간은 2024년 분기별입니다.

Fold:

- 2024 Q1 시작 전까지 전체 학습, Q1 검증
- 2024 Q2 시작 전까지 전체 학습, Q2 검증
- 2024 Q3 시작 전까지 전체 학습, Q3 검증
- 2024 Q4 시작 전까지 전체 학습, Q4 검증

각 fold에서 validation 시작 직전 시점까지 누적된 모든 labeled data를 train window로 사용합니다.

예:

```text
[ train ][ val ]
[     train     ][ val ]
[         train         ][ val ]
[             train             ][ val ]
```

### 7.4 random

무작위 validation trial을 10번 생성합니다.

기간 기준:

- group 1, 2: 2022-2024 전체 1096일 기준
- group 3: 2023-2024 전체 731일 기준

각 trial:

- train window: 전체 기간의 60%
- validation window: 전체 기간의 10%

선택 가능한 범위 안에서 validation 시작 시점을 random하게 뽑습니다.

예:

```text
             [ train ][ val ] 
  [ train ][ val ]
                        [ train ][ val ]
       [ train ][ val ]
```

## 8. Feature Selection

### 8.1 Feature Selection Train Period

feature selection은 validation 방식과 별도로 고정된 과거 구간을 사용합니다.

| 그룹 | feature selection 기간 |
| --- | --- |
| group 1 | 2022-01-01 01:00:00 ~ 2023-01-01 00:00:00 |
| group 2 | 2022-01-01 01:00:00 ~ 2023-01-01 00:00:00 |
| group 3 | 2023-01-01 01:00:00 ~ 2024-01-01 00:00:00 |

group 3은 2022년 label이 없으므로 2023년을 사용합니다.

### 8.2 Mode별 Feature Selection Component

`MODELING_MODE = "basic"`일 때:

- `model3_regressor` 기준으로만 feature selection을 합니다.

`MODELING_MODE = "ensemble"`일 때:

- `model1_zero`
- `model2_capacity`
- `model3_regressor`

세 component별 SHAP ranking을 종합합니다.

### 8.3 2-stage Selection

현재 FEATURE_SELECTION_TOPK_SHAP의 기본값은 1500 입니다.

단계:

1. 모든 후보 feature로 LightGBM importance 계산
2. importance top 5000 선택
3. top 5000 feature로 다시 모델 학습
4. SHAP 계산
5. SHAP top 1500 저장
6. component별 ranking을 종합해 그룹별 최종 feature set 생성

### 8.4 Cache

feature selection 결과는 mode별 cache 파일로 저장됩니다.

예:

- `kpx_group_1_ensemble_combined_shap_top1500.csv`
- `kpx_group_1_ensemble_feature_selection_cache_meta.json`

cache signature에는 다음 정보가 반영됩니다.

- group
- modeling mode
- top-k importance 수
- top-k SHAP 수
- feature selection component 목록

validation option이나 train row 수는 cache 재사용 조건에 포함하지 않습니다.

## 9. Modeling

### 9.1 basic 모드

`MODELING_MODE = "basic"`이면 각 그룹마다 하나의 LightGBM regressor만 학습합니다.

흐름:

1. 선택된 feature로 `X_train`, `X_valid` 생성
2. target 발전량을 그대로 회귀 학습
3. 예측값을 `[0, capacity]`로 clipping
4. final prediction으로 사용

classifier는 만들지 않습니다.

### 9.2 ensemble 모드

`MODELING_MODE = "ensemble"`이면 세 모델을 학습합니다.

Model I: zero classifier

- target: `actual < 100`
- 출력: zero probability
- threshold: `ZERO_THRESHOLD = 0.80`

Model II: capacity classifier

- target: `actual >= capacity * 0.98`
- 출력: capacity probability
- threshold: `CAPACITY_THRESHOLD = 0.98`

Model III: regressor

- target: actual generation
- objective: `REGRESSOR_OBJECTIVE`
- 현재 기본값: LightGBM `regression_l1`

최종 의사결정:

```text
zero_flag=True, capacity_flag=False -> prediction = 0
capacity_flag=True, zero_flag=False -> prediction = capacity
둘 다 True 또는 둘 다 False -> regressor prediction 사용
```

그 후 최종 예측값은 `[0, capacity]`로 clipping됩니다.

### 9.3 Classifier 학습

classifier는 class imbalance를 완화하기 위해 `class_balance_weight()`를 사용합니다.

positive class weight:

```text
positive_weight = negative_count / positive_count
```

### 9.4 Regressor 학습

regressor는 `create_regressor()`에서 생성합니다.

지원 objective:

- `l1`: `regression_l1`
- `l2`: `regression_l2`
- `huber`: `huber`

`huber`일 경우:

```text
alpha = group capacity * HUBER_ALPHA_RATIO
```

sample weight는 `USE_POWER_SAMPLE_WEIGHT=True`일 때 적용됩니다.

## 10. Grid Search

### 10.1 Combo 생성

`make_combo_list()`가 parameter 조합을 만듭니다.

`basic` 모드:

- regressor grid만 사용합니다.
- `MAX_GRID_COMBOS_PER_GROUP`만큼 random sampling합니다.

`ensemble` 모드:

- zero classifier grid
- capacity classifier grid
- regressor grid

의 product를 만든 뒤, `MAX_GRID_COMBOS_PER_GROUP`만큼 sampling합니다.

capacity classifier parameter set이 최대한 골고루 포함되도록 stratified sampling을 시도합니다.

### 10.2 Checkpoint

grid search 결과는 아래 파일에 누적 저장됩니다.

- `regime_lgbm_grid_results.csv`
- `regime_lgbm_grid_fold_results.csv`

`run_key`는 다음 요소를 포함합니다.

- group
- modeling mode
- validation option
- validation signature
- parameter hash

따라서 다른 validation option 또는 modeling mode의 결과가 섞이는 것을 방지합니다.

### 10.3 Best Parameter 선택

grid search 완료 후 `status == "ok"`인 row 중에서 그룹별 best row를 고릅니다.

정렬 기준:

1. `selection_score` 내림차순
2. `ficr` 내림차순
3. `one_minus_nmae` 내림차순

선택 결과는 `BEST_SUMMARY_PATH`에 저장됩니다.

## 11. Refit Best Validation Models

best parameter를 선택한 뒤, validation fold별로 다시 모델을 학습하고 예측합니다.

저장되는 주요 객체:

- `validation_predictions`
- `validation_metrics`
- `wrong_regime_cases`
- `validation_artifacts`

`validation_predictions`에는 다음 정보가 포함됩니다.

- `forecast_kst_dtm`
- `group`
- `fold`
- `actual_kwh`
- `zero_prob`
- `capacity_prob`
- `zero_flag`
- `capacity_flag`
- `model1_zero_pred_kwh`
- `model2_capacity_pred_kwh`
- `model3_reg_pred_kwh`
- `final_pred_kwh`
- 각 모델별 residual

Residual 정의:

```text
residual = prediction - actual
```

따라서 residual이 양수이면 과대예측, 음수이면 과소예측입니다.

## 12. Validation Diagnostics

### 12.1 Residual Distributions

다음 plot을 생성합니다.

- 그룹별 final residual KDE
- 평가 row만 대상으로 한 final residual KDE
- 그룹별 final residual boxplot
- 평가 row만 대상으로 한 final residual boxplot
- 시간대별 residual boxplot
- actual vs final prediction time series
- model-wise residual boxplot
- model-wise residual KDE

평가 row:

```text
actual >= capacity * 0.10
```

### 12.2 Wrong Regime Cases

`ensemble` 모드에서만 의미가 있습니다.

저장 대상:

- Model I zero classifier가 잘못 맞힌 case
- Model II capacity classifier가 잘못 맞힌 case

저장 파일:

- `wrong_regime_cases_with_features.csv`

`basic` 모드에서는 classifier가 없으므로 이 진단은 비어 있을 수 있습니다.

### 12.3 Feature Importance

LightGBM 내장 feature importance를 계산합니다.

저장 파일:

- `lgbm_builtin_feature_importance.csv`

### 12.4 Permutation Importance

validation sample에서 feature를 섞었을 때 score가 얼마나 떨어지는지 계산합니다.

metric:

- classifier: ROC-AUC 또는 accuracy
- regressor: `selection_score`

저장 파일:

- `permutation_importance.csv`

### 12.5 SHAP

SHAP TreeExplainer를 사용합니다.

저장 파일:

- `shap_importance.csv`
- SHAP summary plot PNG

`MODELING_MODE`에 따라 SHAP 계산 component가 달라집니다.

- `basic`: regressor만
- `ensemble`: zero classifier, capacity classifier, regressor

## 13. Final Train And Submission

`RUN_FINAL_TRAIN_AND_SUBMIT=True`이면 최종 submission을 생성합니다.

절차:

1. 그룹별 best parameter row를 읽습니다.
2. 각 그룹에 대해 전체 labeled train data로 모델을 재학습합니다.
3. 2025년 test feature를 예측합니다.
4. 예측값을 `[0, capacity]`로 clipping합니다.
5. `sample_submission`의 column 순서를 보존해 merge합니다.
6. `baseline_submit.csv`를 저장합니다.

저장 파일:

- `baseline_submit.csv`
- `final_model_summary.csv`

`final_model_summary.csv`에는 다음 정보가 포함됩니다.

- group
- modeling mode
- train rows
- test rows
- feature count
- zero fire rate
- capacity fire rate
- prediction min/mean/max

## 14. 주요 산출물 정리

| 산출물 | 설명 |
| --- | --- |
| `feature_inventory.csv` | 그룹별 feature 생성 현황 |
| `{target}_feature_columns.csv` | 그룹별 최종 후보 feature 목록 |
| `{target}_lag_rolling_base_columns.csv` | lag/rolling 생성 기준 feature 목록 |
| `regime_lgbm_grid_results.csv` | grid search 요약 |
| `regime_lgbm_grid_fold_results.csv` | fold별 grid search 결과 |
| `regime_lgbm_best_summary.csv` | 그룹별 best parameter |
| `validation_predictions.csv` | validation 예측 결과 |
| `validation_metrics.csv` | validation metric |
| `official_like_group_metrics.csv` | 공식 산식 유사 그룹별 metric |
| `validation_residuals_long.csv` | residual 분석용 long table |
| `wrong_regime_cases_with_features.csv` | classifier wrong case |
| `lgbm_builtin_feature_importance.csv` | LightGBM 내장 importance |
| `permutation_importance.csv` | permutation importance |
| `shap_importance.csv` | SHAP importance |
| `baseline_submit.csv` | 최종 제출 파일 |
| `final_model_summary.csv` | final model/test summary |

## 15. 자주 바꿀 설정값

### 15.1 모델 구조 변경

단일 회귀 모델:

```python
MODELING_MODE = "basic"
```

세 모델 결합:

```python
MODELING_MODE = "ensemble"
```

### 15.2 Validation 변경

```python
VALIDATION_OPTION = "full"
VALIDATION_OPTION = "sliding"
VALIDATION_OPTION = "expanding"
VALIDATION_OPTION = "random"
```

### 15.3 Grid Search 조합 수 변경

```python
MAX_GRID_COMBOS_PER_GROUP = 16
```

값을 키우면 더 많은 parameter 조합을 실험하지만, 실행 시간이 늘어납니다.

### 15.4 Feature Selection 수 변경

```python
FEATURE_SELECTION_TOPK_IMPORTANCE = 5000
FEATURE_SELECTION_TOPK_SHAP = 1500
```

이 값을 바꾸면 feature selection cache 조건도 바뀝니다.

### 15.5 Objective 변경

```python
REGRESSOR_OBJECTIVE = "l1"
REGRESSOR_OBJECTIVE = "l2"
REGRESSOR_OBJECTIVE = "huber"
```

현재 custom loss는 사용하지 않습니다.

### 15.6 Sample Weight 변경

```python
USE_POWER_SAMPLE_WEIGHT = True
MIN_NON_EVAL_WEIGHT = 0.35
HIGH_POWER_WEIGHT_STRENGTH = 6.0
```

고발전량 구간을 더 강하게 학습하고 싶으면 `HIGH_POWER_WEIGHT_STRENGTH`를 키웁니다.

## 16. 실행 시 주의사항

### 16.1 Cache와 재실행

feature selection cache를 재사용하려면:

```python
REUSE_FEATURE_SELECTION_CACHE = True
FORCE_RERUN_FEATURE_SELECTION = False
```

feature selection을 다시 하고 싶으면:

```python
FORCE_RERUN_FEATURE_SELECTION = True
```

grid search를 이어서 하고 싶으면:

```python
FORCE_RERUN_GRID_SEARCH = False
RESUME_GRID_SEARCH = True
```

grid search를 처음부터 다시 하고 싶으면:

```python
FORCE_RERUN_GRID_SEARCH = True
```

### 16.2 전처리 변경 후 Cache

전처리 구조가 바뀌면 feature 이름이나 feature 값이 바뀔 수 있습니다.

이 경우 기존 feature selection cache가 논리적으로 맞지 않을 수 있으므로, 다음 설정을 권장합니다.

```python
FORCE_RERUN_FEATURE_SELECTION = True
```

### 16.3 GPU 사용

Colab GPU 환경에서는 `COLAB_GPU` 환경변수에 따라 `USE_LGBM_GPU`가 자동으로 True가 될 수 있습니다.

다만 LightGBM GPU는 OpenCL 환경에 따라 불안정할 수 있습니다. 오류가 발생하면:

```python
USE_LGBM_GPU = False
```

로 바꾸고 CPU로 실행하는 것이 안전합니다.

### 16.4 결측치 처리

원본 LDAPS/GFS 기상 예보 데이터에는 결측치가 없지만, lag, rolling, inverse feature, merge 과정에서 NaN 또는 inf가 생길 수 있으므로

최종 feature matrix를 모델에 넣기 전에 clean_matrix()에서 아래와 같이 처리합니다.

처리 방식:

1. `inf`, `-inf`를 NaN으로 변환
2. train median으로 결측치 대체
3. 남은 NaN은 0으로 대체
4. `float32`로 변환

target label의 NaN은 보간하지 않습니다. 학습/검증 mask에서 제외됩니다.

### 16.5 Data Leakage 관련

이 baseline은 기상예보 데이터의 `forecast_kst_dtm` 기준 feature를 사용합니다.

이 코드는 현재 대회 데이터가 이미 “예측 대상 시각별 사용 가능한 예보값” 형태로 제공된다는 전제에서 작성된 baseline입니다.

## 17. 추천 실행 순서

일반적인 실험 순서는 다음과 같습니다.

1. Configuration에서 `MODELING_MODE`, `VALIDATION_OPTION`, grid 수를 정합니다.
2. 전처리 셀까지 실행하여 feature inventory를 확인합니다.
3. feature selection cache를 사용할지 결정합니다.
4. grid search를 실행합니다.
5. best summary와 validation score를 확인합니다.
6. residual plot에서 과소/과대예측 패턴을 확인합니다.
7. feature importance, permutation importance, SHAP을 확인합니다.
8. final train 및 submission 생성 셀을 실행합니다.

빠른 실험:

```python
MAX_GRID_COMBOS_PER_GROUP = 3
RUN_PERMUTATION_IMPORTANCE = False
RUN_SHAP = False
```

정식 실험:

```python
MAX_GRID_COMBOS_PER_GROUP = 16
RUN_PERMUTATION_IMPORTANCE = True
RUN_SHAP = True
RUN_FINAL_TRAIN_AND_SUBMIT = True
```

## 18. 요약

이 노트북은 풍력발전량 예측을 위한 LightGBM baseline입니다.

핵심 특징:

- 그룹별 공간 feature를 다르게 생성합니다.
- LDAPS/GFS의 풍속, 공기밀도, power proxy, pressure difference, BLH 변화량 등을 사용합니다.
- lag/rolling feature로 시간적 연속성을 일부 반영합니다.
- feature importance와 SHAP을 이용해 feature 수를 줄입니다.
- validation option 4가지를 지원합니다.
- 단일 regressor `basic` 모드와 세 모델 결합 `ensemble` 모드를 모두 지원합니다.
- 최종 제출 파일과 진단 plot/importance 파일을 자동 저장합니다.

현재 기본값은 다음 실험 설정에 해당합니다.

```python
MODELING_MODE = "ensemble"
VALIDATION_OPTION = "random"
REGRESSOR_OBJECTIVE = "l1"
FEATURE_SELECTION_TOPK_IMPORTANCE = 5000
FEATURE_SELECTION_TOPK_SHAP = 1500
MAX_GRID_COMBOS_PER_GROUP = 12
```

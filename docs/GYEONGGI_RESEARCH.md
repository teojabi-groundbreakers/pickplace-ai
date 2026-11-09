# 경기도 카드매출 성장률 연구용 모델

## 목적과 제한

기존 경기도 월별 검토 후보의 지역·업종별 완전한 분기 매출 네 개를 사용하여 다음 분기의 전년 동기 대비 성장률을 예측하는 후향 연구 모델이다. 월별 금액을 3개월 합산할 수 있다는 가정은 검증되지 않았고 단위는 미확인이다. 코드가 같다는 것은 H10 출처와 경계 동일성을 증명하지 않는다. 과거 공개 버전·공개 시점을 확인하지 못한 2026-11-09 취득 수정 스냅샷을 사용하므로 당시 실시간 예측의 성능이 아니다. 운영이나 기존 API에서 사용하지 않는다.

## 자료와 분할

- 동일한 원본 지역·83개 업종 코드 조합에서 서로 다른 3개월이 모두 있는 경우만 분기를 구성한다. 월별 중복과 미확인 업종 등 기존 제외 행은 복원하지 않는다. 2024Q1을 보간하지 않으며 2022~2023 행은 학습에 포함되지 않는다.
- 목표는 `sales[t+1] / sales[t-3] - 1`이다. 분모 0·입력 불완전 조합은 제외한다. 4분기 공통 특징 계산은 `pickplace_ai.features.gyeonggi_four_quarter_features`를 사용한다.
- 학습 목표 분기 2025Q2, 검증 2025Q4, 시험 2026Q2로 분리하고 2025Q3·2026Q1은 평가 모델 선정에 사용하지 않는다. 각 학습·검증·시험은 단일 분기로 기간 일반화·계절성을 검증하기에 부족하다. 공백 분기는 실제 당시 공개일을 보장하지 않는다.
- 같은 표본에서 0% 성장 기준과 HistGradientBoosting 회귀를 비교하고 검증 MAE로 선정한다. 시험 점수는 선정에 사용하지 않는다. 극단적인 성장률을 임의 삭제하지 않는다.

## 실행 결과 (2026-11-09)

| 후보 | 검증 MAE | 시험 MAE | 시험 중앙 절대오차 |
|---|---:|---:|---:|
| 0% 성장 기준 | 49.29%p | 66.63%p | 17.74%p |
| HistGradientBoosting | 63.81%p | 71.30%p | 14.20%p |

검증에서 0% 기준이 선택됐다. 학습 30,465건, 검증 30,524건, 시험 30,469건이며 각 분기에 83업종이 포함됐다. 시험 중 966건은 평가 모델의 학습 분기에 등장하지 않은 지역·업종 조합이다. 시험 목표 성장률 최대값은 약 2,282배로 평균에 영향을 주므로 중앙값과 p90도 메타데이터에 기록했다. %p는 성장률의 오차 단위이며 정확도·성공률이 아니다.

평가가 끝난 뒤 2025Q2~2026Q2의 5개 목표 분기 152,405개 표본으로 HGB를 별도 재학습했다. 이 최종 모델은 평가 모델과 다르므로 위 시험 점수를 최종 모델의 독립 성능으로 주장할 수 없다.

산출물 `artifacts/gyeonggi-retrospective-20261109/`의 `selected-model.joblib`은 검증에서 선택된 0% 기준, `learned-model.joblib`은 2025Q2 평가용 학습 HGB, `final-learned-model.joblib`은 5분기 재학습 HGB다. `metadata.json`에 분할·입력/코드/모델 지문, 표본 범위, 지표, 제약이 있다. 예측 하한 -100%를 적용한 평가와 동일한 결과를 얻으려면 공통 예측 함수를 사용해야 한다. 로컬의 신뢰할 수 있는 모델 파일만 읽는다.

## 재실행

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.train_gyeonggi_research \
  --candidate data/processed/gyeonggi/monthly-review-20261109/candidates.csv \
  --review-manifest data/processed/gyeonggi/monthly-review-20261109/manifest.json \
  --snapshot 2026-11-09 \
  --output artifacts/gyeonggi-retrospective-new
```

기존 디렉터리는 덮어쓰지 않는다. 원본 데이터·모델·결과는 Git 제외 대상이다. 매출 집계 정의, 행정동·업종 연계, 실제 공개 시점 및 중복 오류를 제공기관에 확인하고 새로운 시점의 독립 평가를 수행해야 한다.


## 절대오차 손실 개선 실험 (2026-11-09)

기존 동일 특징·학습/검증 분할·표본을 사용하고, 극단치를 삭제하지 않은 채 HGB 손실을 `absolute_error`로 바꾼 후보를 비교했다. 검증에서만 평균 절대오차로 후보를 선택했다. 검증 자료도 여러 후보 탐색에 사용했으므로 독립적인 성능 보장은 아니다. 2026Q2 시험은 기존 실험에서 이미 결과를 확인한 자료여서 여기서는 **재사용한 탐색 비교**이며 새 독립 시험이 아니다.

| 후보 | 2025Q4 검증 MAE | 재사용 2026Q2 MAE | 재사용 2026Q2 중앙 절대오차 |
|---|---:|---:|---:|
| 0% 성장 기준 | 49.29%p | 66.63%p | 17.74%p |
| 제곱오차 HGB | 63.81%p | 71.30%p | 14.20%p |
| 절대오차 HGB | 39.28%p | 54.74%p | 12.81%p |

검증상 절대오차 HGB가 선택됐다. 결과는 극단값과 분모가 작은 행의 영향을 크게 받으며, 기존 시험의 0% 기준에서는 절대오차 상위 1%가 총 절대오차의 53.87%를 차지했다. 금액 단위·집계 기준이 확인되지 않아 임의의 금액 기준으로 행을 제외하지 않았다.

검증·재사용 시험과는 별도로 최종 모델을 2025Q2~2026Q2 5분기 152,405행으로 재학습했다. 이 모델은 시험 정답까지 학습했으므로 위 수치를 최종 모델의 독립 성능으로 표현하지 않는다. `artifacts/gyeonggi-improved-research-20261109/`에 평가용 세 모델, 최종 선택·절대오차 모델, 메타데이터가 있다. 대상 지역·업종 범위는 호출자가 메타데이터에서 확인해야 하며, 기존 API에는 연결되어 있지 않다.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.train_gyeonggi_improved \
  --candidate data/processed/gyeonggi/monthly-review-20261109/candidates.csv \
  --review-manifest data/processed/gyeonggi/monthly-review-20261109/manifest.json \
  --snapshot 2026-11-09 \
  --output artifacts/gyeonggi-improved-research-new
```

새로 확보한 완전한 분기와 과거 공개 버전으로 별도 평가하기 전에는 운영 성능으로 사용하지 않는다. FE·BE 응답 점검은 [응답 계약 점검](GYEONGGI_CONTRACT_AUDIT.md)을 참조한다.

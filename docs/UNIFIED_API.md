# 서울·경기 통합 성장률 연구 API

## 범위와 주의

서울 행정동 추정매출과 경기도 행정동 카드매출을 함께 학습한 단일 연구 모델을 내부 API로 시험한다. 원천별 값과 업종·지역 코드가 같은 의미라고 가정하지 않으며 응답의 `source`와 코드 체계를 반드시 보존한다. 서울은 `seoul-adstrd` 8자리와 `CS` 업종 코드, 경기는 `gyeonggi-admdong-10-unverified` 10자리 원천 코드와 3자리 중분류 업종 코드를 사용한다. FE의 H10 코드를 임의로 자르거나 업종명으로 변환하지 않는다.

서울 원본의 단위 표기는 KRW이나 호출 입력의 실제 단위는 검증되지 않는다. 경기의 매출 단위와 월별 합산 방식, 두 원천의 지역 경계·과거 공개 버전·실제 전향 성능은 확인되지 않았다. 호출자가 보낸 이력값과 코드 대응의 진위는 검증되지 않는다. 수치는 운영·사업 판단에 승인되지 않았고 창업 성공 확률이 아니다. FE의 현행 연구 화면은 서울 추정매출이라는 설명이 고정되어 있으므로 경기도 결과를 그대로 표시하면 안 된다. 이 응답은 **BE 연동용 내부 DTO**이며 FE 분석 응답 전체를 대체하지 않는다.

모델은 2026년 2분기까지의 라벨로 학습했으며 `2026Q2` 기준 다음 분기인 `2026Q3`만 지원한다. 이는 모델 작성 당시 이미 끝난 분기의 연구용 사후 추정이다. `2026Q4`나 과거 대상은 `unavailable`과 `null`, 이유를 반환한다. 재학습 모델의 독립 시험값은 없으므로 `retrospective_test`는 null이다. 점수·영향 요인·통계·유동인구·경쟁·지도·요약·추천·예측구간은 모두 null이며 `score_unavailable_reasons` 또는 `unavailable_reasons`에서 이유를 확인한다. 요청이 올바르지만 데이터 부족·범위 미지원이면 HTTP 200에서 `status=unavailable`; 인증 오류는 401, 형식 오류는 422, 모델 미준비 또는 내부 오류는 503과 `null`·사유를 반환한다.

## 실행

모델 파일과 metadata는 운영자가 신뢰하는 경로에서 직접 준비한다. 코드·파일 해시는 손상 및 불일치를 확인하며, 악성 joblib을 안전하게 만들지 않는다. API 키는 별도 설정하고 저장소·대화에 남기지 않는다. 원본 데이터, 실자료 예시와 모델은 Git에 포함하지 않는다.

```bash
export PICKPLACE_UNIFIED_MODEL_ROOT="$PWD/artifacts/unified-growth-research-20261109"
read -rs PICKPLACE_UNIFIED_API_KEY
export PICKPLACE_UNIFIED_API_KEY
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/uvicorn pickplace_ai.unified_api:app --host 127.0.0.1 --port 18003 --workers 1
```

다른 터미널에서 `/health`의 `model_ready`를 확인한다. `/research/unified/model-info`와 `/research/unified/predict`에는 `X-API-Key`가 필요하다. `/docs`와 `/openapi.json`은 모델 요청 및 응답 형식 참고용이다.

## 실제 자료로 로컬 호출 예시 만들기

아래 명령은 로컬에 확보한 원본을 모델의 데이터 지문과 비교한 뒤, 학습 범위 안에서 2025Q3~2026Q2의 네 개 완전한 분기를 가진 서울·경기 조합을 하나씩 선택한다. `data_available_by`는 자료가 **그 날짜까지 보유됐음을 확인한 날짜**이며, 최초 공개일이나 정확한 취득일을 뜻하지 않는다. 예시는 실제 집계값으로 작성되므로 외부에 그대로 게시하지 않는다. 이미 존재하는 출력 폴더는 덮어쓰지 않는다.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.make_unified_requests \
  --seoul data/processed/seoul/seoul-2021-2026q2-20261104.csv \
  --gyeonggi data/processed/gyeonggi/monthly-review-20261109/candidates.csv \
  --model-root artifacts/unified-growth-research-20261109 \
  --output-dir reports/unified-research-examples-new
```

다른 터미널에서 같은 연구용 API 키를 환경변수에 설정하고 호출한다.

```bash
curl -sS http://127.0.0.1:18003/research/unified/predict \
  -H "X-API-Key: $PICKPLACE_UNIFIED_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @reports/unified-research-examples-new/seoul-request.json
curl -sS http://127.0.0.1:18003/research/unified/predict \
  -H "X-API-Key: $PICKPLACE_UNIFIED_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @reports/unified-research-examples-new/gyeonggi-request.json
```

`history`는 서로 다른 4개의 달력 분기 `2025Q3`, `2025Q4`, `2026Q1`, `2026Q2`와 해당 원천 기준 매출을 받는다. 각 입력과 `data_available_by`가 분석 기준일 `as_of`까지 사용 가능해야 하고, 분석 기준일은 모델의 작성일과 두 원천의 보유 확인일보다 빠를 수 없다. 증명되지 않은 과거 공개 시점을 만들어 넣지 않는다. 기간 누락, 결측, 중복, 비양수 분모, 범위 외 코드는 숫자를 만들어 대체하지 않는다.

## Docker

로컬 모델 디렉터리를 읽기 전용으로 연결한다. 먼저 API 키를 환경변수로 설정하고 프로젝트 루트에서 실행한다. 기본 Compose는 내부 네트워크만 생성하며, 로컬 포트 테스트가 필요할 때만 loopback 전용 overlay를 추가한다.

```bash
export PICKPLACE_UNIFIED_MODEL_SOURCE="$PWD/artifacts/unified-growth-research-20261109"
read -rs PICKPLACE_UNIFIED_API_KEY
export PICKPLACE_UNIFIED_API_KEY
docker compose -f compose.unified.yaml -f compose.unified.local.yaml up --build -d
```

호스트 접근은 `127.0.0.1:18003`, 같은 Docker 네트워크의 BE에서는 `http://unified-api:8000`을 사용한다. BE 컨테이너에 네트워크를 연결하고 타임아웃·인증 및 응답 변환을 별도로 구현해야 한다. 실제 BE·FE 통합과 클라우드 배포 완료를 뜻하지 않는다.


## 공동 모델 학습 및 검토 결과

서울·경기의 서로 다른 매출 정의를 유지하고 출처 식별값과 공통 비율 특징으로 단일 HGB 모델을 학습했다. 학습 시 원천별 가중치 총합을 같게 두었고 최종 모델은 서울 80,131행, 경기 152,405행을 사용했다. 원본 업종·지역 코드를 상호 변환하지 않았다.

| 2025Q4 검증 MAE | 공동 모델 | 같은 조건의 지역별 모델 |
|---|---:|---:|
| 서울 | 70.28%p | 69.33%p |
| 경기 | 38.83%p | 39.28%p |

검증은 각 원천별로 같은 표본에서 계산한 후향 탐색값이며 공동 모델이 모든 지역에서 우수하지 않다. 2026Q2 시험도 이전에 결과를 살펴본 재사용 자료다. 이후 최종 모델은 해당 분기까지 재학습했으므로 위 지표를 최종 모델의 독립 성능으로 사용하지 않는다. 2026Q3 추정은 이미 지난 분기에 대한 연구용 사후 추정이고 실제 정확도를 입증하지 않는다.

`metadata.json`에 분할, 출처별 범위·확인일, 데이터·코드·모델 지문, 지역별 평가 및 제한을 기록한다. 파일의 지문은 신뢰할 수 없는 joblib을 안전하게 만들지 않는다. 모델 생성과 API 처리는 동일한 `unified_features.py`와 기존 `features.py`의 비율 특징을 사용한다. 기존 서울 near-term API는 해당 모델이 기록한 원본 feature 코드 버전을 선택하도록 복구했으며 무결성 검사를 유지한다.


## 서울 매출 이력 보류 정책 (2026-11-09)

`POST /research/unified/predict`는 서울의 입력값이 완전하고 코드·시점·모델 범위를 통과한 뒤, 모델을 호출하기 전에 네 개의 분기 이력을 확인한다. 입력 매출은 호출자가 전달한 값이며 API는 그 진위나 단위를 독립 검증하지 않는다. 아래는 2025Q2 학습 입력 분포에서 계산한 서울 원화 연구 기준이다. 기준과 같은 값은 보류하지 않는다.

| 신호 | 보류 조건 |
|---|---|
| `low_reference_sales` | 목표 분기의 전년 동기 매출이 약 1,054만 원 미만 |
| `low_latest_sales` | 가장 최근 분기 매출이 약 810만 원 미만 |
| `rapid_sales_increase` | 이전 연속 분기 중 어느 한 번이라도 직전 분기 대비 약 206.66% 초과 증가 |
| `rapid_sales_decrease` | 이전 연속 분기 중 어느 한 번이라도 직전 분기 대비 약 72.79% 초과 감소 |

하나라도 해당하면 HTTP 200에 `status: unavailable`, `estimated_yoy_growth: null`, `reason: seoul_sales_history_unstable`, 한국어 `reason_message`를 반환한다. `risk_signals`와 `risk_signal_messages`는 해당된 모든 코드와 한국어 설명을 같은 순서로 담는다. 정상 서울 요청은 `risk_policy_status: evaluated`와 빈 신호 배열을 반환한다. 예측 시점의 네 분기만 사용하며 예측 대상의 실제 매출·오차는 사용하지 않는다.

경기는 원천 매출 단위와 기준을 검증하지 못했으므로 이 서울 원화 기준을 적용하지 않는다. 경기 응답은 `risk_policy_status: not_applicable`, `risk_policy_reason: gyeonggi_risk_policy_unverified` 및 한국어 이유를 반환한다. 이는 경기 매출이 안정적이라고 검증됐다는 뜻이 아니다.

정책은 기존 공동 모델과 서울 원천/평가 모델 지문에 묶여 있다. 기준과 다른 모델에서 서울 요청은 `seoul_guard_policy_mismatch`로 보류한다. `/health`의 `seoul_risk_policy_ready`, 인증된 `/research/unified/model-info`의 `seoul_risk_policy`에서 버전·기준과 검증 상태를 확인한다. 원본 데이터와 다른 API는 변경하지 않는다.

이 기준은 이미 살펴본 자료에 대한 사후 연구로 정했으며 신규 기간의 전향 검증은 없다. 2025Q4에서 네 신호 중 하나를 보류한 연구는 라벨과 4분기 입력이 있는 표본 16,002건 중 1,890건(11.81%)을 보류했고, 남은 14,112건의 조건부 MAE는 20.19%p였다. 이는 전체 요청의 제공률이나 최종 재학습 모델의 성능을 뜻하지 않는다. 급변과 저매출은 실제 데이터 오류의 증거가 아니다.

BE는 `reason`을 안정적인 기계 코드로 보존하고, FE에 표시할 사유로 `reason_message`와 필요한 `risk_signal_messages`를 전달한다. 현 FE/BE에는 이 매핑이 아직 구현되지 않았으며, 경기 출처를 서울로 표시하는 FE 고정 문구도 별도로 수정해야 한다.

연구 기준 재현은 다음과 같다. 원본·코드·모델 지문을 확인하며 기존 출력은 덮어쓰지 않는다.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.audit_seoul_guard \
  --seoul data/processed/seoul/seoul-2021-2026q2-20261104.csv \
  --model-root artifacts/unified-growth-research-20261109 \
  --output reports/seoul-guard-policy-new
```


## 보류 근거 상세

보류 응답의 `risk_details`는 각 신호의 `signal`, `from_period`, `to_period`, `observed_value`, `threshold_value`, `unit`, `comparison`을 담는다. 저매출은 `from_period=null`과 해당 관측 분기, 급증·급감은 연속된 이전·이후 분기를 표시한다. 여러 변화 중 가장 큰 증가 또는 감소 하나를 사용하며 동률은 가장 이른 구간이다. 정상·경기·정책 불일치·모델 오류 응답에는 빈 배열이다.

`unit=ratio`의 값 1.0은 100% 증가, -0.7은 70% 감소다. `unit=KRW`는 호출자의 단위 선언일 뿐 입력값이나 단위를 독립 확인했다는 뜻이 아니다. 네 분기의 월별 합계 여부를 API에서 검증하지 않는다. 서울 원본은 분기별 제공이지만 매출 열이 `당월_매출_금액`이므로 3개월 합계를 자동 가정하거나 3배하지 않는다. 별도의 원본 의미 확인 전 `risk_details`를 원인이나 확정 오류로 제시하지 않는다. 기존 보류 기준과 경기 미적용 정책은 유지한다.

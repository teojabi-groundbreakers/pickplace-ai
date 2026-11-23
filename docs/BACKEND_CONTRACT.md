# Spring Boot와 AI 서비스 연결

AI 서비스는 **내부 계산 API**다. FE가 요청하는 `/analyses`의 전체 결과를 대신하지 않는다. Spring Boot는 사용자 인증, 지역·업종 코드 매핑, DB 이력 조회, 통계·지도 응답 및 AI 결과 통합을 담당한다.

지표별 최신 분기를 제공하는 다음 공개 응답과 FE 작업은 [지표별 기간 안내](PER_INDICATOR_PERIOD_HANDOFF.md)를 참고한다. 현재 기본 `/predict`는 한 요청에 `base_period` 하나를 사용하며, 승인된 서로 다른 분기마다 별도 요청을 받아 각각의 결과를 반환한다.

## 파트별 책임

| 담당 | 책임 |
|---|---|
| Spring Boot | 원본 요청 코드 보존, 공식 코드 대응 확인, 데이터 조회, 내부 AI 호출, 타임아웃과 부분 실패 처리 |
| FastAPI | 입력 검증, 공통 특징 생성, 모델 추론, 진단지수, 결과의 단위·버전·가용성 반환 |
| FE | 점수·예측값·기준 시점 표시, 데이터 부족과 미지원 상태 구분 |

현재 FE는 카카오에서 얻은 원본 10자리 행정동 H 코드를 그대로 전달한다. AI의 서울 CSV는 `seoul-adstrd` 코드를 보존한다. Spring Boot는 **공식 대응표와 기준일을 확인해** 서로 연결하고, 최종 FE 응답에는 원래 요청 코드를 보존해야 한다. 이름 매칭이나 자릿수 절단, 임의 `00` 추가를 운영 코드 변환 규칙으로 사용하지 않는다. `coffee` 같은 FE 업종 코드도 서울의 원본 업종 코드와 공식 대응을 합의한다.

## 내부 API

| 경로 | 역할 |
|---|---|
| `GET /health` | 프로세스·기본진단과 모델 준비 상태 확인 |
| `GET /model-info` | 로딩된 모델과 지원 범위 확인 |
| `POST /predict` | 같은 지역·코드 체계·업종의 시계열로 가능한 예측과 진단 반환 |

실행 중인 서버의 `/openapi.json`과 `/docs`가 구현된 요청 구조의 기준이다. `/predict`에는 다음을 전달한다.

- `schema_version`: `1.0`
- `region_code`, `region_code_system`, `industry_code`
- `base_period`: `YYYYQn`
- `as_of`: 분석일 `YYYY-MM-DD`
- `source_kind`: `official` 또는 명시적으로 허용한 `synthetic`
- `history`: 공통 Observation 스키마의 관측 행 목록

모든 관측은 같은 지역·코드 체계·업종·출처 구분에 속해야 한다. 중복 분기, 기준 분기 이후 관측, 분석일에 아직 공개되지 않은 행, NaN·무한대 및 음수는 거부한다. 모델 특징 생성은 학습과 API에서 같은 코드를 사용한다.

## 응답 해석

- 성장 예측의 `estimated_yoy_growth=0.05`는 전년 동기 대비 예상 증가율 5%다.
- `risk_indicators.sales_yoy_change`와 `risk_indicators.closure_burden`는 요청 이력에서 계산한 별도 관측 지표다. 화면이나 보고서에는 두 값을 각 기준 분기와 함께 표시한다. [분리 지표 계약](#매출-변화와-폐업-부담)
- `fit_indicators`는 기준일과 지역 대응이 확인된 서울 자료에서 점포당 주민 수와 점포 밀집도의 원래 값과 업종 내 백분위를 제공한다. [참조표 생성과 응답 계약](SEOUL_FIT_REFERENCE.md)
- `risk.score`는 높을수록 위험 신호가 크고, `fit.score`는 높을수록 정의된 지표상 적합도가 높다.
- 초안 규칙 지수는 정책 버전과 초안 상태를 함께 해석한다. 검증된 폐업확률·창업 성공확률이 아니다.
- 성장 예측의 `target_period`, `comparison_period`, 모델·정책 버전을 표시하거나 저장한다.
- 예측 불가·미지원·모델 미준비는 `status`, `reason`, `null` 값으로 처리한다. `null`을 0점으로 바꾸지 않는다.
- 합성 입력·모델은 응답에서도 `source_kind=synthetic`으로 유지한다.
- 요청 분석일보다 늦은 정보를 학습·선택·평가에 사용한 모델은 `model_information_after_as_of`로 예측을 제한한다. 모델 시간 메타데이터가 부족하면 `invalid_model_temporal_metadata`를 반환한다. 과거 날짜를 넣는 것만으로 최신 모델이 당시 예측 모델이 되지는 않는다.

FE의 현재 계약 초안에는 일부 점수와 수치형 신뢰도를 필수로 기대하는 부분이 있다. AI의 부분 가용성·`null` 결과를 연동하려면 BE·FE 계약을 조정해야 한다. 검증하지 않은 신뢰도 숫자나 종합점수를 만들어 기존 타입을 채우지 않는다. 이번 AI 준비 작업은 기존 FE 소스·계약을 수정하지 않는다.

### 서울 2026Q4 성장 예측의 이전 분기 전환

연구 모드의 기본 `POST /predict`에서 `official` 서울 원본 코드와 `base_period=2026Q3`으로 요청하면 Q4를 대상으로 삼는다. Q3 기준 요청에 연속 매출 이력이 있으면 기본 1분기 모델의 판단을 따른다. Q3 매출이 없거나 이력이 부족하면 서버에 설정한 공식 서울 스냅샷에서 같은 행정동·업종의 Q2까지 연속 5분기를 확인하고, Q4용 2분기 앞 모델로 직접 계산한다. 이 요청에서는 Q3 관측 행이 빠지거나 `history`가 빈 배열이어도 정상 처리한다.

| 성장 예측 결과 | 필드 |
|---|---|
| Q2 입력으로 Q4 추정 | `growth.base_period=2026Q2`, `growth.target_period=2026Q4`, `growth.horizon_quarters=2`, `growth.fallback_reason=missing_sales` 또는 `insufficient_history`, `growth.release_status=experimental_research_only` |
| Q2 이력도 부족 | `growth.estimated_yoy_growth=null`, `growth.base_period=null`, `growth.target_period=2026Q4`, `growth.reason=fallback_insufficient_history` 또는 `fallback_missing_sales` |
| Q4 종료 후 요청 | `growth.estimated_yoy_growth=null`, `growth.reason=target_not_future` |

최상위 `base_period`는 요청 기준 Q3이며, 위험도·적합도는 계속 그 분기를 사용한다. `growth.fallback_reason`은 원래 요청의 부족 사유이며, `growth.reason`은 최종 결과를 제공할 수 없을 때의 사유다. 백엔드가 자료 부족으로 호출 자체를 `SKIPPED` 처리하면 이 전환은 실행되지 않는다.

로컬에서 검증된 모델과 스냅샷을 지정해 시작한다. 스냅샷 CSV와 같은 위치에 `.manifest.json` 파일이 있어야 한다. `/health`의 `seoul_growth_fallback_ready`와 `seoul_growth_fallback_reason`으로 로드 상태를 확인한다.

```bash
PICKPLACE_ALLOW_EXPERIMENTAL=true \
PICKPLACE_SEOUL_FALLBACK_MODEL_DIR=artifacts/seoul-near-term-2026q2-20261113 \
PICKPLACE_SEOUL_FALLBACK_SNAPSHOT=data/processed/seoul/seoul-2021-2026q2-20261104.csv \
PYTHONPATH=src:. .venv/bin/uvicorn pickplace_ai.api:app --host 127.0.0.1 --port 18000
```

### 매출 변화와 폐업 부담

기본 `POST /predict` v1.0은 기존 `risk` 객체 옆에 `risk_indicators`를 반환한다. 두 관측 지표는 종합지수나 성장 모델 준비 상태와 독립적으로 계산한다.

| 응답 필드 | 계산과 표시 | 필요한 관측 |
|---|---|---|
| `risk_indicators.sales_yoy_change` | 기준 분기 매출 ÷ 전년 동기 매출 − 1. 예를 들어 `-0.20`은 전년 동기 대비 20% 감소 | 기준 분기와 정확히 네 분기 전의 같은 지역·업종 매출 |
| `risk_indicators.closure_burden` | 기준 분기 폐업 수 ÷ 그 분기의 전체 점포 수. `0.05`는 관측한 전체 점포 대비 폐업 비중 5% | 같은 기준 분기의 폐업 수와 전체 점포 수 |

각 지표에서 `status`, `value`, `unit=ratio`, `reason`, `base_period`를 읽는다. 매출에는 `comparison_period`, `current_sales_amount`, `comparison_sales_amount`가 포함되고, 폐업 부담에는 `closing_count`와 `store_count`가 포함된다. 관측값이 부족하거나 분모가 유효하지 않으면 해당 지표의 `value`가 `null`이고 `reason`으로 이유를 전달한다. 기준 분기 매출이나 폐업 수가 0인 경우는 유효한 관측이다.

| 사유 | `reason` |
|---|---|
| 기준·비교 매출 누락 | `missing_current_sales`, `missing_comparison_sales` |
| 비교 매출 0 이하 | `nonpositive_comparison_sales` |
| 전체 점포 수 또는 폐업 수 누락 | `missing_store_count`, `missing_closing_count` |
| 전체 점포 수 0 이하 | `nonpositive_store_count` |
| 폐업 수가 전체 점포 수를 초과 | `closing_count_exceeds_store_count` |
| 비율을 유한한 값으로 계산할 수 없음 | `nonfinite_derived_indicator` |

표시 문구는 매출 **변화**와 폐업 **부담**을 구분한다. 폐업 부담은 해당 분기 관측 비중이다. 개별 점포의 미래 폐업 여부나 확률을 표시하려면 별도 검증이 필요하다. 지표별 표시를 사용하는 쪽은 두 필드의 가용성을 개별 확인한다.

## 운영·통합

- `PICKPLACE_SEOUL_FIT_REFERENCE_DIR`에 검증된 서울 비교표 디렉터리 또는 `YYYYQn`별 비교표를 담은 상위 디렉터리를 설정할 수 있다. 미설정 시 `fit_indicators`의 두 값과 백분위는 null이다.
- `PICKPLACE_MODEL_DIR`: 배포할 모델 산출물 디렉터리
- `PICKPLACE_API_KEY`: 내부 요청 키. 설정하면 `/predict`·`/model-info` 호출의 `X-API-Key` 헤더에 함께 전달한다.
- `PICKPLACE_REQUIRE_API_KEY`: 운영 시 `true`. 비밀키가 비어 있으면 시작을 중단한다. 로컬 개발의 무키 실행과 구분한다.
- `PICKPLACE_ALLOW_SYNTHETIC`: 합성 데모 요청·모델 허용 여부. 기본 비활성
- `PICKPLACE_ALLOW_EXPERIMENTAL`: 평가·승인이 끝나지 않은 실험 모델 허용 여부. 기본 비활성

제공 Compose는 AI 컨테이너 준비용이며 운영 내부키를 요구한다. Spring Boot 컨테이너도 `pickplace-ai-internal` 네트워크에 연결한 후 `http://ai-api:8000`으로 호출한다. 다른 Compose에서 이미 만든 네트워크를 함께 사용한다면 `external: true` 등 실제 네트워크 연결 구성을 확인한다. AI 포트를 호스트나 외부 인터넷에 공개하지 않는다. Spring Boot는 AI 장애 시 통계 결과를 유지하고 AI 부분만 사용 불가로 표시한다.

`/health`는 기본진단 가능 상태(`basic_diagnosis_ready`)와 성장 모델 준비 상태(`growth_model_ready`)를 구분한다. 모델이 없더라도 기본진단 서비스가 실행 중이면 HTTP 200일 수 있으므로 성장 모델의 준비 여부를 상태 필드로 확인한다. 서울 참조표 상태와 로드한 분기는 `fit_reference_ready`, `fit_reference_reason`, `fit_reference_periods`로 확인한다.

경기 2026Q2의 `fit_indicators`를 조회할 때는 행안부 H10 코드 체계 `mois-resident-admin-10`과 소진공 업종 소분류를 사용한다. 경기 응답의 `store_count_definition`은 `sbiz_listed_active_businesses`이며 경기의 수록 영업 업소를 분모·분자로 사용한다. 경기 참조표의 상태는 `/health`의 `gyeonggi_fit_reference_ready`, `gyeonggi_fit_reference_reason`, `gyeonggi_fit_reference_periods`로 확인한다. [경기 원천과 적용 범위](GYEONGGI_FIT_REFERENCE.md)

외부 SGIS 경계 조회, 프론트엔드 직접 연결 및 전국 데이터 수집 API는 이 서비스의 초기 구현 범위에 포함하지 않는다.

## 후향 실험 모델 연동 테스트

63개 업종 연구 모델은 별도의 `pickplace_ai.research_api` 서비스로만 시험한다. `GET /research/industries`, `POST /research/predict`에 연구용 `X-API-Key`가 필요하다. 호출자는 공통 관측 이력과 실제 확보 기준일, 모델 종류(`selected` 또는 `learned`)를 전달한다. 완료된 업종별 시험 분기만 재현하며, 응답은 `retrospective_research_only`와 검증되지 않은 지역·입력 범위를 표시한다. 운영 `/predict`, 실시간 미래 예측 또는 성공 확률로 취급하지 않는다. Docker·요청 생성·BE 네트워크는 [후향 테스트 API](RESEARCH_API.md)를 참고한다.


## 서울·경기 통합 연구 API (2026-11-09)

`POST /research/unified/predict`는 서울과 경기를 함께 학습한 별도 후향 연구 모델을 호출한다. 지역별 원천과 코드 체계를 명시하고 최근 4개 완전 분기의 이력을 BE가 전달한다. FE 분석 전체 응답이 아닌 내부 DTO다. 성장률만 연구용으로 반환하며 점수·요인·지도 등 제공 불가 항목은 null과 사유로 표시한다. 지원 범위 밖, 이력 누락, Q4 등 미지원 기간도 null과 사유다. [통합 API 문서](UNIFIED_API.md)에 요청·인증·실행 방법을 기록한다.

공동 모델의 검증 오차는 서울 70.28%p, 경기 38.83%p이며 재사용된 과거 시험과 재학습 이후 최종 모델의 성능은 독립 검증이 아니다. 기존 FE의 연구 예측 문구는 서울로 고정돼 경기 출처를 표시하도록 FE·BE 계약을 조정하기 전 그 화면에 직접 연결하지 않는다. API의 원본 코드와 FE H10·업종 간 자동 대응이나 실제 FE/BE 통합은 구현하지 않았다.


### 서울 연구용 매출 이력 보류

통합 API의 서울 요청이 전년·최근 분기 저매출이나 급증·급감 기준에 걸리면 `status=unavailable`, `estimated_yoy_growth=null`, `reason=seoul_sales_history_unstable`, `reason_message`와 `risk_signals`/`risk_signal_messages`를 반환한다. BE는 코드를 보존하고 한국어 메시지를 FE에 전달해야 하며 이를 0이나 점수로 변환하지 않는다. 경기 요청은 `risk_policy_status=not_applicable`과 미검증 이유를 반환한다. 해당 기준은 탐색적이며 실제 FE/BE 연결은 아직 검증하지 않았다. [통합 API 문서](UNIFIED_API.md) 참고.


보류 응답의 `risk_details`는 사유별 관측 분기·값·기준값·단위를 전달한다. `ratio=1.0`은 100% 변화이며 KRW는 호출자의 단위 선언이다. 이는 원인 확정이나 입력 검증이 아니다. BE가 사유 문구와 함께 전달할 수 있지만 실제 FE/BE 통합은 아직 수행하지 않았다. [통합 API 설명](UNIFIED_API.md) 참고.

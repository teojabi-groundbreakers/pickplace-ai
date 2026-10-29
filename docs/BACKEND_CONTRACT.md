# Spring Boot와 AI 서비스 연결

AI 서비스는 **내부 계산 API**다. FE가 요청하는 `/analyses`의 전체 결과를 대신하지 않는다. Spring Boot는 사용자 인증, 지역·업종 코드 매핑, DB 이력 조회, 통계·지도 응답 및 AI 결과 통합을 담당한다.

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
- `risk.score`는 높을수록 위험 신호가 크고, `fit.score`는 높을수록 정의된 지표상 적합도가 높다.
- 초안 규칙 지수는 정책 버전과 초안 상태를 함께 해석한다. 검증된 폐업확률·창업 성공확률이 아니다.
- 성장 예측의 `target_period`, `comparison_period`, 모델·정책 버전을 표시하거나 저장한다.
- 예측 불가·미지원·모델 미준비는 `status`, `reason`, `null` 값으로 처리한다. `null`을 0점으로 바꾸지 않는다.
- 합성 입력·모델은 응답에서도 `source_kind=synthetic`으로 유지한다.
- 요청 분석일보다 늦은 정보를 학습·선택·평가에 사용한 모델은 `model_information_after_as_of`로 예측을 제한한다. 모델 시간 메타데이터가 부족하면 `invalid_model_temporal_metadata`를 반환한다. 과거 날짜를 넣는 것만으로 최신 모델이 당시 예측 모델이 되지는 않는다.

FE의 현재 계약 초안에는 일부 점수와 수치형 신뢰도를 필수로 기대하는 부분이 있다. AI의 부분 가용성·`null` 결과를 연동하려면 BE·FE 계약을 조정해야 한다. 검증하지 않은 신뢰도 숫자나 종합점수를 만들어 기존 타입을 채우지 않는다. 이번 AI 준비 작업은 기존 FE 소스·계약을 수정하지 않는다.

## 운영·통합

- `PICKPLACE_MODEL_DIR`: 배포할 모델 산출물 디렉터리
- `PICKPLACE_API_KEY`: 내부 요청 키. 설정하면 `/predict`·`/model-info` 호출의 `X-API-Key` 헤더에 함께 전달한다.
- `PICKPLACE_REQUIRE_API_KEY`: 운영 시 `true`. 비밀키가 비어 있으면 시작을 중단한다. 로컬 개발의 무키 실행과 구분한다.
- `PICKPLACE_ALLOW_SYNTHETIC`: 합성 데모 요청·모델 허용 여부. 기본 비활성
- `PICKPLACE_ALLOW_EXPERIMENTAL`: 평가·승인이 끝나지 않은 실험 모델 허용 여부. 기본 비활성

제공 Compose는 AI 컨테이너 준비용이며 운영 내부키를 요구한다. Spring Boot 컨테이너도 `pickplace-ai-internal` 네트워크에 연결한 후 `http://ai-api:8000`으로 호출한다. 다른 Compose에서 이미 만든 네트워크를 함께 사용한다면 `external: true` 등 실제 네트워크 연결 구성을 확인한다. AI 포트를 호스트나 외부 인터넷에 공개하지 않는다. Spring Boot는 AI 장애 시 통계 결과를 유지하고 AI 부분만 사용 불가로 표시한다.

`/health`는 기본진단 가능 상태(`basic_diagnosis_ready`)와 성장 모델 준비 상태(`growth_model_ready`)를 구분한다. 모델이 없더라도 기본진단 서비스가 실행 중이면 HTTP 200일 수 있으므로 성장 모델의 준비 여부를 상태 필드로 확인한다.

외부 SGIS 경계 조회, 프론트엔드 직접 연결 및 전국 데이터 수집 API는 이 서비스의 초기 구현 범위에 포함하지 않는다.

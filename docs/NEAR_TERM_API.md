# 2026년 단기 추정 연구용 FastAPI

2026Q2까지의 서울 행정동·업종 이력을 이용해 2026Q3 또는 2026Q4 전년 동기 대비 추정매출 성장률을 계산하는 내부 테스트 API다. 모델은 수정 스냅샷으로 학습했고 후향 비교만 수행했다. 운영 또는 창업 판단에 승인된 모델이 아니다. 2026Q3은 2026-11-04 기준 이미 끝난 분기의 사후 추정이다.

## 준비 및 실행

Git에 포함되지 않는 신뢰된 `artifacts/seoul-near-term-2026q2-20261104/`의 `model.joblib`, `summary.json`, `forecast-2026-q3-q4.csv`가 필요하다. 모델의 지문은 파일 무결성만 확인하며 신뢰할 수 없는 joblib을 안전하게 만들지 못한다. 제공된 범위 파일은 학습 범위 기록 확인용이며, 입력의 원본 진위나 공식 코드 매핑을 검증하지 않는다.

프로젝트 루트에서 Docker Desktop을 켜고 로컬 테스트용 키를 별도로 생성한다.

```bash
export PICKPLACE_NEAR_TERM_API_KEY="$(openssl rand -hex 32)"
docker compose -f compose.near-term.yaml -f compose.near-term.local.yaml up -d --build
curl http://127.0.0.1:18001/health
curl -H "X-API-Key: $PICKPLACE_NEAR_TERM_API_KEY" \
  http://127.0.0.1:18001/research/near-term/model-info
```

기본 Compose는 포트를 외부에 게시하지 않고 `pickplace-ai-near-term-internal` 내부 네트워크를 사용한다. 로컬 오버레이는 추가 브리지와 `127.0.0.1:18001`만 사용한다. Docker로 실행하는 Spring Boot는 내부 네트워크에 연결해 `http://near-term-api:8000`을 호출할 수 있다. 호스트에 직접 실행한 Spring Boot는 로컬 오버레이의 `http://127.0.0.1:18001`을 사용한다. 다른 PC나 인터넷에는 노출하지 않는다.

다른 모델 경로가 필요할 때 `PICKPLACE_NEAR_TERM_MODEL_SOURCE`를 신뢰된 호스트 디렉터리로 설정한다. API 키는 BE와 AI 서버만 보유하며 React의 `VITE_` 변수·코드·Git에 넣지 않는다. `/health` 외의 API는 `X-API-Key`가 필요하다. 모델 지문·코드·라이브러리 버전이 다르거나 모델 파일이 없으면 `model_ready=false`와 함께 추론 요청은 503이다.

## 실제 데이터 요청 예시

결합 공통 CSV를 가진 로컬 환경에서 예시 요청 파일을 만든다. 실제 데이터는 Git에 넣지 않고 기존 파일을 덮어쓰지 않는다.

현재 개발 기기에는 검증용 `reports/near-term-example-2026Q3.json`과 `reports/near-term-example-2026Q4.json`도 있다. 두 파일은 집계된 실제 데이터를 포함하고 Git에는 올라가지 않는다.

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.make_near_term_request \
  --input data/processed/seoul/seoul-2021-2026q2-20261104.csv \
  --region-code 11110515 --industry-code CS100010 \
  --target-period 2026Q3 --as-of 2026-11-04 \
  --output reports/near-term-request.json

curl -X POST http://127.0.0.1:18001/research/near-term/predict \
  -H "X-API-Key: $PICKPLACE_NEAR_TERM_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @reports/near-term-request.json
```

`as_of`는 데이터 취득일 이후이면서 한국 현재 날짜 이하여야 한다. 실제 사용 시 현재 분석일을 지정한다. BE가 `2025Q2~2026Q2` 연속 5분기의 동일 행정동·업종 이력을 제공한다. 서울 `seoul-adstrd` 8자리 코드와 원본 `CS...` 업종 코드만 지원한다. 카카오 H10을 자르거나 이름으로 추정하지 않고, 공식 대응 관계가 확인된 경우에만 BE가 변환한다.

응답의 `estimated_yoy_growth=0.05`는 전년 동기 대비 5% 추정이고 점수나 성공 확률이 아니다. `retrospective_test`의 `mae_pp`·`median_absolute_error_pp`는 과거 시험의 오차이며 미래 신뢰구간이 아니다. 결측·지원하지 않는 범위 등은 HTTP 200에서 `status=unavailable`, 성장률 `null`, `reason`을 반환한다. 잘못된 형식은 422, 인증 오류는 401, 모델 장애는 503이다. 어떤 실패도 데모나 임의의 0으로 대체하지 않는다.

`input_values_verified=false`, `region_scope_verified=false`와 `warnings`를 유지한다. `scope_seen_in_training`은 해당 조합의 학습 이력 여부이지 코드 매핑이나 모델 적합성 인증이 아니다. Spring Boot는 이 값을 사용자에게 과장 없이 전달해야 한다.

```bash
docker compose -f compose.near-term.yaml -f compose.near-term.local.yaml down
```

실제 Spring Boot와 FE의 통합은 아직 수행하지 않았다. 현재 API의 대상 기준은 2026Q2, 예측 분기는 2026Q3/Q4로 고정돼 있으며, 새 분기 자료를 쓰려면 별도 학습·검증·버전 갱신이 필요하다.

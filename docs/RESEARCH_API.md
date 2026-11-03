# 서울 업종별 모델 후향 테스트 API

이 API는 서울시 **현재 확보한 수정 스냅샷의 과거 시험 분기 재현**과 Spring Boot 연동 시험을 위한 별도 서비스다. 학습 당시 공개된 자료를 재현한 백테스트, 현재·미래 예측, 실제 사업성 판단에 사용할 수 없다. 운영 `/predict`와 앱·네트워크·설정을 분리한다.

## 모델과 요청 데이터 준비

- 신뢰할 수 있는 운영자가 만든 `artifacts/seoul-all-industries-retrospective-20261103/`의 `summary.json`과 `industries/<업종코드>/`를 준비한다. 모델과 원본·가공 CSV는 Git에 포함되지 않으므로 테스트 기기에 별도 전달해야 한다.
- 모델 목록은 서버 시작 시 읽으며, 개별 모델은 요청 시 코드·버전·해시를 검증한 후 불러온다. SHA-256은 손상 확인이지 신뢰할 수 없는 joblib의 안전을 보장하지 않는다. 출처가 검증된 모델만 마운트한다.
- 요청의 `history`는 호출자가 제공한다. 서버가 DB나 CSV에서 자동 조회하지 않는다. 공통 스냅샷 CSV를 보유한 개발자는 아래 생성 도구를 사용할 수 있다.
- 예측 대상은 업종별 `test_periods`의 완료된 분기로 제한된다. 요청 `as_of`와 각 행의 `available_at`은 스냅샷 취득일 이후여야 한다. 과거 최초 공개일로 값을 소급하지 않는다.
- 모델 데이터에 지역별 학습 범위가 기록돼 있지 않아 지역 범위·요청 원본의 진위는 검증하지 못한다. 응답의 `region_scope_verified=false`, `input_values_verified=false`와 연구 전용 경고를 유지한다.

## 로컬 Docker 실행

Docker Desktop 또는 Docker Engine이 실행된 상태에서 프로젝트 루트의 터미널에서 실행한다. 키는 예시가 아닌 별도 연구용 값을 설정하고 코드나 Git에 넣지 않는다.

```bash
export PICKPLACE_RESEARCH_API_KEY='replace-with-your-own-ascii-key'
docker compose -f compose.research.yaml -f compose.research.local.yaml up -d --build
```

기본 Compose는 외부에 포트를 공개하지 않고 `pickplace-ai-research-internal` 내부망에만 연결한다. 로컬 오버레이는 추가 브리지를 연결하고 `127.0.0.1:18000`에만 포트를 게시한다. 호스트 직접 실행 Spring Boot에서도 같은 호스트의 이 주소를 사용할 수 있다. 키는 공유 보안 채널로 별도로 전달한다.

컨테이너는 기본적으로 호스트의 `./artifacts/seoul-all-industries-retrospective-20261103`을 `/app/research-models`에 읽기 전용으로 마운트한다. 다른 위치를 쓰면 `PICKPLACE_RESEARCH_MODEL_SOURCE`에 **호스트 경로**를 지정한다. 이미지 자체에는 모델이 포함되지 않는다.

```bash
curl http://127.0.0.1:18000/health
curl -H "X-API-Key: $PICKPLACE_RESEARCH_API_KEY" \
  http://127.0.0.1:18000/research/industries
```

`/health`의 `registry_ready`는 모델 목록을 읽었는지 뜻하며 `model_artifacts_verified=false`는 개별 모델 전체를 미리 검증하지 않았다는 뜻이다. 개별 모델은 호출 시 검증한다. 모델 목록·예측에는 `X-API-Key`가 필수다. 준비되지 않은 레지스트리나 모델은 오류를 반환하며 정상 결과로 대체하지 않는다.

## 실제 스냅샷으로 요청 만들기

`data/processed/seoul/seoul-all-snapshot-20261103.csv`를 가진 환경에서만 다음 예제를 만들 수 있다. 결과에는 선택한 지역·업종의 실제 집계값 5행이 포함된다. 공유 시 서울시 출처·이용 조건을 함께 확인한다. `reports/`는 Git 제외 대상이며 기존 파일은 덮어쓰지 않는다.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_research_request \
  --input data/processed/seoul/seoul-all-snapshot-20261103.csv \
  --region-code 11110515 --industry-code CS100010 \
  --base-period 2025Q3 --as-of 2026-11-03 \
  --model-variant learned --output reports/research-request.json

curl -X POST http://127.0.0.1:18000/research/predict \
  -H "X-API-Key: $PICKPLACE_RESEARCH_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @reports/research-request.json
```

이 예는 카페·음료 업종의 `2025Q4` 시험 대상 재현이다. `model_variant=selected`는 해당 업종의 검증 우승 후보이며 기준 예측일 수 있다. `learned`는 학습된 Ridge 또는 HistGradientBoosting 후보다. 응답의 `estimated_yoy_growth=0.05`는 전년 동기 대비 5%이며, 신뢰도나 성공 확률이 아니다.

다른 업종의 시험 분기와 저표본 경고는 `/research/industries`에서 확인한다. 고시원처럼 시험 분기가 다른 업종도 있다. 서울 행정동 코드는 8자리 `seoul-adstrd`를 사용하며, 카카오 H10과의 변환은 공식 대응표를 확인해야 한다. 코드 자르기나 이름 추정은 사용하지 않는다.

## Spring Boot가 Docker 컨테이너일 때

Spring Boot Compose에서 기존 테스트 네트워크에 연결하고 `http://research-api:8000`으로 호출한다. 연구 서버를 먼저 시작해야 한다.

```yaml
services:
  backend:
    networks:
      - research_internal

networks:
  research_internal:
    external: true
    name: pickplace-ai-research-internal
```

위 코드는 기존 BE 서비스의 네트워크 설정에 추가하는 예시다. 기존 서비스가 사용하는 다른 네트워크가 있다면 함께 유지한다. 다른 PC에서 직접 접속하려면 별도의 인증된 접근 경로가 필요하며, 이 Compose는 원격 공개를 구성하지 않는다.

## 종료

```bash
docker compose -f compose.research.yaml -f compose.research.local.yaml down
```

모델의 과거 버전·실제 공개 시점, 최신 분기 자료, 지역별 범위·요청 데이터 검증, 전향 평가와 실제 BE 통합은 별도로 남아 있다.

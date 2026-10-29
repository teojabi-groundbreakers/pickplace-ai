# PickPlace AI

Spring Boot에서 호출하는 FastAPI 기반 내부 AI 서비스와 데이터 준비·학습 도구입니다.

**현재 단계는 실데이터 모델 개발을 시작할 수 있는 로컬 기반 구현입니다.** 서울시 원본 CSV·API 키는 아직 제공되지 않았으며, 합성 자료의 학습·테스트 결과를 실제 상권 예측 성능으로 표시하지 않습니다. 기존 `pickplace-frontend`는 수정하지 않습니다.

## 포함한 기능

- 공통 데이터 스키마와 출처·공개 시점 검증
- 서울시 추정매출·점포 CSV 가져오기 및 품질 검사
- 학습·추론에서 재사용하는 특징 생성 코드
- 분기별 시간순 분할과 공개 시점에 맞는 라벨 제외
- 단순 기준 예측·Ridge·HistGradientBoosting 비교 및 모델 저장
- 선택적 LightGBM 학습 경로
- 초안 위험·적합도 지수 및 모델 입력 민감도 설명
- `/health`, `/model-info`, `/predict` 내부 API
- Docker 배포 설정과 자동 테스트

실제 데이터·합성 데이터, 미검증 모델·운영 승인 모델을 구분합니다. 기본 설정에서는 합성 요청과 미검증 모델을 차단하고, 성장 예측 불가 시 점수를 조작하지 않고 `null`과 사유를 반환합니다.

## 시작하기

Python 3.13을 권장합니다. 아래 명령은 새 프로젝트 폴더에서 실행합니다.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.lock -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check .
```

`requirements.lock`은 검증한 패키지 버전의 제약 파일입니다. 재현 시 설치 명령에 `-c requirements.lock`을 포함합니다. 패키지·Python 버전이 다르면 모델 로더가 호환되지 않는 산출물을 거부할 수 있습니다.

## 합성 자료로 전체 과정 확인

다음 명령이 만드는 데이터는 허구의 지역 코드·업종을 사용하는 예시입니다. 실제 지역 자료가 없다고 자동 실행하거나 실제 요청의 대체 결과로 사용하지 않습니다.

```bash
pickplace-ai make-demo --output data/demo.csv --request-output reports/demo-request.json
pickplace-ai inspect --input data/demo.csv --output reports/demo-quality.json
pickplace-ai prepare --input data/demo.csv --output data/demo-features.csv
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pickplace-ai train \
  --input data/demo.csv --output artifacts/demo --model hist
```

출력 경로에 기존 파일·모델이 있으면 덮어쓰지 않습니다. 반복 실험은 `data/demo-v2.csv`, `artifacts/demo-v2` 등 새로운 경로를 사용합니다.

명시적으로 예시 모드를 허용한 로컬 API를 실행합니다.

```bash
PICKPLACE_ALLOW_SYNTHETIC=true \
PICKPLACE_ALLOW_EXPERIMENTAL=true \
PICKPLACE_MODEL_DIR=artifacts/demo \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uvicorn pickplace_ai.api:app --host 127.0.0.1 --port 8000 --workers 1
```

다른 터미널에서 다음 명령으로 응답을 확인합니다.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @reports/demo-request.json
```

`examples/`의 JSON은 저장소에 포함한 합성 참고 자료입니다. 위 명령은 재실행 시 참고 자료를 덮어쓰지 않도록 새 요청을 Git 제외 대상인 `reports/`에 생성합니다.

`as_of`는 분석에 사용할 정보 기준일입니다. 예시 요청의 과거 기준일은 의도적으로 작성한 합성 테스트이며 현재 시점의 실제 예측이 아닙니다. 운영에서는 Spring Boot가 실제 분석일을 지정하고 예측 대상 분기를 확인해야 합니다.

## 실제 데이터 준비

1. [데이터 준비 문서](docs/DATA.md)의 공식 사이트에서 매출·점포 CSV를 받습니다.
2. 원본의 업종·지역·단위·관측기간과 실제 공개 시점을 확인합니다.
3. 동일한 공개 기준으로 처리할 수 있는 파일 묶음별로 가져옵니다.
4. 정규화한 자료를 검사하고 유효한 과거 예측 표본을 확인합니다.
5. 합성 자료와 분리된 새 경로에 모델을 학습합니다.

```bash
pickplace-ai import-seoul \
  --sales data/raw/seoul-sales.csv \
  --stores data/raw/seoul-stores.csv \
  --available-at 2026-10-29 \
  --sales-basis as_published \
  --output data/processed/seoul-snapshot.csv

pickplace-ai inspect --input data/processed/seoul-snapshot.csv
```

위 날짜는 **명령 형식 예시**입니다. 확인한 실제 공개·사용 가능일로 바꾸세요. 오늘 내려받은 과거 파일 모두에 오늘 날짜를 넣었다면, 그 자료는 과거에 사용할 수 있었다는 증거가 아니므로 과거 예측 학습 표본이 만들어지지 않을 수 있습니다. 과거 공개 시점을 임의로 생성하지 않습니다.

실제 학습 가능한 자료를 확보한 후 실행합니다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pickplace-ai train \
  --input data/processed/seoul-history.csv --output artifacts/seoul-v1 --model hist
```

LightGBM은 선택적입니다. 설치 후 `--model lightgbm`을 명시합니다. 환경에 따라 OpenMP 런타임이 필요하며 이번 기본 환경 검증과 구분합니다.

```bash
python -m pip install -c requirements.lock -e '.[dev,lightgbm]'
```

## 구조

```text
src/pickplace_ai/
  schemas.py     # 요청·관측 스키마
  periods.py     # 분기 연산
  data.py        # 공통 CSV/JSON 입출력·품질 검사
  ingest.py      # 서울시 원본 CSV 어댑터
  features.py    # 공통 특징·미래 라벨 생성
  training.py    # 시간순 학습·평가·후보 선택
  artifacts.py   # 메타데이터·체크섬·환경 확인·추론
  scoring.py     # 초안 진단지수
  api.py         # 내부 FastAPI
  demo.py        # 명시적 합성 자료 생성
  cli.py         # 준비·학습 명령
config/data_sources.json
examples/
tests/
docs/
Dockerfile
compose.yaml
```

API는 FE의 `/analyses` 응답을 직접 대신하지 않습니다. Spring Boot가 통계·지도·AI 결과를 통합합니다. 기존 FE가 요구하는 점수·신뢰도와 AI의 부분 응답 사이의 차이는 [BE 계약 문서](docs/BACKEND_CONTRACT.md)를 확인하고 합의해야 합니다.

## 문서

- [문서 색인](docs/README.md)
- [모델 및 진단지표 설계](docs/MODEL_DESIGN.md)
- [실제 데이터 확보·반입](docs/DATA.md)
- [Spring Boot 연동 계약](docs/BACKEND_CONTRACT.md)
- [실행·배포·운영](docs/OPERATIONS.md)
- [작업 이력과 남은 일](docs/WORKLOG.md)

# 서울 열린데이터광장 API 수집

[서울시 추정매출·행정동](https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do?tab=A)의 `VwsmAdstrdSelngW`와 [점포·행정동](https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do?tab=A)의 `VwsmAdstrdStorW`를 기준년분기로 조회한다. 두 서비스의 응답을 원본 영문 필드명 그대로 UTF-8 BOM CSV로 보관하며, 기존 `import-seoul`은 이 필드와 인코딩을 지원한다.

## 키 전송 주의

서울시 공식 API 안내 주소는 `http://openapi.seoul.go.kr:8088`이며 인증키가 URL 경로에 포함된다. 이 환경에서 시험한 HTTPS 8088은 TLS 오류, 443은 연결 시간초과로 **HTTPS를 확인하지 못했다**. HTTP 호출은 키와 응답이 네트워크에서 암호화되지 않는다. 수집기는 기본적으로 호출을 거부하며 이 사실을 이해하고 명시적으로 `--allow-insecure-http`를 지정했을 때만 실행한다. 출력·메타데이터에는 키와 요청 URL을 기록하지 않는다.

키를 채팅이나 Git에 넣지 않고 실행할 터미널에서 환경변수로 설정한다. 다음 명령은 키를 입력할 때 화면에 표시하지 않는다.

```bash
printf '서울 열린데이터광장 API 키: '
read -rs SEOUL_OPEN_DATA_API_KEY
export SEOUL_OPEN_DATA_API_KEY
printf '\n'
```

`sample` 키는 최대 5건만 반환하므로 전체 수집에 사용할 수 없다. 수집기는 `.env`를 자동으로 읽지 않는다.

## 수집

프로젝트 루트에서 요청한다. 아래 명령은 실제 API에 HTTP로 키를 전송한다.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.fetch_seoul_api \
  --periods 2026Q1 2026Q2 --allow-insecure-http
```

매출과 점포가 각각 `data/raw/seoul/sales/`, `data/raw/seoul/stores/`에 `seoul-api-sales-2026Q1.csv`, `seoul-api-stores-2026Q1.csv` 같은 이름으로 저장된다. 각 CSV 옆의 `.metadata.json`에 데이터셋 페이지, 수집 시작·완료 UTC 시각, 분기, 행 수와 파일 SHA-256을 기록한다. 이 파일들은 모두 Git 제외 대상이다.

요청은 최대 1,000건씩 분할하며 전체 건수 변경, 누락된 페이지, 중복 키, 다른 분기, 필드 변경, API 오류를 검사한다. 검증 실패 시 해당 분기의 CSV와 메타데이터는 완성본으로 기록하지 않는다. 각 자료·분기는 독립적으로 저장되므로 이후 요청이 실패해도 앞서 성공한 파일은 남는다. 기존 파일은 덮어쓰지 않으므로 새 수집은 별도의 `--output-root`를 사용한다. 수집 중 총건수가 같고 중복이 없어도 서버가 페이지 사이에 데이터를 수정하지 않았음을 보장할 수는 없다.

2026년 11월 4일 확인 당시 1·2분기는 제공됐고 3분기는 `INFO-200`으로 조회되지 않았다. 이는 조회 시점 상태다. 데이터가 없는 분기는 0으로 채우지 않는다.

같은 날 실제 키로 수집한 결과는 매출 1분기·2분기 각각 16,659건, 점포 1분기 35,313건·2분기 35,301건이다. 네 파일의 지문을 확인했으며 기존 import-seoul에서 매출 33,318건 모두에 점포 정보가 연결됐다. 이 확인은 2026년 예측 모델을 재학습하거나 평가했다는 뜻은 아니다.

## 가공과 사용

수집한 두 CSV는 기존 `import-seoul`의 `--sales`, `--stores` 입력으로 사용할 수 있다. `--available-at`은 매출과 점포를 모두 확보한 실제 시각을 고려한 날짜를 지정한다. 수집 시각을 과거 최초 공개일로 바꿔 기록하지 않는다. 원본의 `THSMON_SELNG_AMT`는 제공된 값을 유지하며 임의로 3배하지 않는다.

2026년 데이터를 이용하기 전에 기존 2026년 추정 결과를 보존하고 1·2분기 실제 값과 비교한다. 기존 연간 추정 스크립트는 2021~2025년 입력만 허용하며 새 CSV를 자동 반영하지 않는다. 최신 자료를 반영한 [연구용 단기 추정 모델](NEAR_TERM_FORECAST.md)과 [FastAPI](NEAR_TERM_API.md)는 별도로 구현했다. 실제 Spring Boot 연동과 운영 검증은 아직 완료하지 않았다.

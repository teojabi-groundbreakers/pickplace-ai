# 서울 분기별 상권 현황 연구 API

서울에서 확보한 매출·점포·상주인구·길단위인구 원본을 각각 보존하고, 원본 행정동과 업종 코드 및 분기로 조회하는 내부 연구 API다. 이는 모델 예측이나 FE 전체 분석 응답이 아니다. 기존 서울·경기 통합 성장률 API도 변경하지 않았다.

## 원본과 제공 범위

| 자료 | 실제 로컬 수집 원본 범위 | API 제공 항목 | 해석 시 주의 |
|---|---|---|---|
| [서울 추정매출·행정동](https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do) | 2021Q1~2026Q2, 분기·행정동·업종 | 원본 `당월_매출_금액` | 원본 값을 가공 없이 반환한다. 분기별 자료의 월평균인지 등 시기별 집계 의미 확인 전 월평균·분기합계로 단정하지 않는다. |
| [서울 점포·행정동](https://data.seoul.go.kr/dataList/OA-22172/A/1/datasetView.do) | 2021Q1~2026Q2, 분기·행정동·업종 | 원본 전체·개업·폐업 점포 수 | [서울시 안내](https://golmok.seoul.go.kr/introduce.do)의 전체 점포 수는 해당 기간 폐업 점포도 포함한다. 현재 영업 중인 점포 수와 구분한다. 과거 `유사_업종_점포_수`를 현 자료의 `전체_점포_수`와 함께 사용하며 세부 정의 이력은 추가 확인한다. |
| [서울 상주인구·행정동](https://data.seoul.go.kr/dataList/OA-22183/A/1/datasetView.do) | 2021Q1~2026Q2, 분기·행정동 | 원본 총 상주인구 | 업종별 소비자나 유동인구가 아니다. 연간 갱신된 값이 다음 분기에 반복될 수 있다. |
| [서울 길단위인구·행정동](https://data.seoul.go.kr/dataList/OA-22178/A/1/datasetView.do) | 2021Q1~2026Q2, 분기·행정동 | 원본 총 유동인구 | 서울 생활인구 기반 원본 집계값이다. 정확한 일평균인지, 특정 점포 방문자 수인지는 확인하지 않았다. |

위 기간은 **파일에서 확인한 관측 분기**다. 모든 지역·업종에서 매 분기 자료가 있다는 뜻은 아니다. 최신 수정 스냅샷이며 과거 분기의 최초 공개일·정정 이력이 확인되지 않아 과거 예측 당시 사용 가능했던 자료로 간주할 수 없다. 2026Q3의 원본은 이 스냅샷에 없다.

서울 [공식 서비스 안내](https://golmok.seoul.go.kr/introduce.do)는 분기 단위 제공을 설명한다. 원본을 월별로 복제하거나 분기를 나눠 월 추이를 만들지 않는다. FE가 요구하는 점포당 월평균 매출, 일평균 유동인구, 월별 추이는 현재 `null`과 사유로 반환한다. 지역 평균·점수·예측·추천은 이 API의 범위가 아니다.

2026-11-09 실제 원본으로 별도 인덱스를 만든 결과 매출 376,485행, 점포 775,084행, 상주인구 9,350행, 길단위인구 9,350행이다. 각각 22개 분기에 값이 나타나지만 지역·업종별 완전성과 지리적 경계의 시점별 일치는 검증하지 않았다. 각 지표를 별도 테이블에 보관하므로 매출이 없는 업종의 점포 관측도 남는다.

## 수집 자료로 새 인덱스 만들기

AI 프로젝트 루트에서 실행한다. `*.csv`에 다른 시점이나 중복 파일을 추가했다면 명시적으로 필요한 파일만 선택해야 한다. 동일한 원천 키의 중복, 유효하지 않은 값 및 처리 중 원본 변경은 오류로 중단한다. 기존 출력 디렉터리는 덮어쓰지 않는다. 원본·인덱스는 Git에 포함하지 않는다.

```bash
cd /Users/factory/Documents/Projects/PickPlace/pickplace-ai
PYTHONPATH=src:. .venv/bin/python scripts/prepare_seoul_stats.py \
  --sales data/raw/seoul/sales/*.csv \
  --stores data/raw/seoul/stores/*.csv \
  --resident data/raw/seoul/resident_population/*.csv \
  --floating data/raw/seoul/floating_population/*.csv \
  --output-dir data/processed/seoul/indicators-new
```

`snapshot.sqlite3`과 `manifest.json`을 새 디렉터리에 생성한다. 매니페스트는 원본 파일 이름·SHA-256, 자료별 행 수·관측 분기, 생성 시각, DB 해시와 미검증 공개일 상태를 기록한다. 해시는 파일 손상·불일치를 확인할 뿐 제공기관의 진위나 자료의 사용 승인을 증명하지 않는다.

출력 폴더 생성 뒤 DB와 매니페스트를 순서대로 게시하므로 처리 중에는 불완전한 폴더가 잠시 보일 수 있다. 강제 종료로 일부만 남았다면 내용을 확인한 뒤 별도 새 경로로 실행한다. 서비스는 파일 두 개와 해시·구조를 시작 시 확인한다. 시작 후 인덱스 파일을 변경하지 말고, 새 버전은 새 경로로 만든 뒤 서버를 재시작한다.

## 로컬 API 실행

```bash
cd /Users/factory/Documents/Projects/PickPlace/pickplace-ai
export PICKPLACE_SEOUL_STATS_DIR="$PWD/data/processed/seoul/indicators-20261109"
read -rs PICKPLACE_SEOUL_STATS_API_KEY
export PICKPLACE_SEOUL_STATS_API_KEY
PYTHONPATH=src:. .venv/bin/uvicorn pickplace_ai.seoul_stats_api:app \
  --host 127.0.0.1 --port 18004
```

키는 현재 터미널에서 입력하며 Git이나 채팅에 기록하지 않는다. 다른 터미널에서 같은 키를 설정한 뒤 호출한다. `11110515`와 `CS100001`은 원본 코드 형식의 조회 예시이며, FE의 10자리 H코드나 `coffee` 코드가 이 값에 대응한다는 의미가 아니다.

```bash
curl -sS http://127.0.0.1:18004/health
curl -sS -G http://127.0.0.1:18004/research/seoul/indicators \
  -H "X-API-Key: $PICKPLACE_SEOUL_STATS_API_KEY" \
  --data-urlencode 'region_code=11110515' \
  --data-urlencode 'industry_code=CS100001' \
  --data-urlencode 'start_period=2025Q3' \
  --data-urlencode 'end_period=2026Q2'
curl -sS http://127.0.0.1:18004/research/seoul/metadata \
  -H "X-API-Key: $PICKPLACE_SEOUL_STATS_API_KEY"
```

`/health` 외 조회는 `X-API-Key`를 요구한다. 조회 범위는 최대 40개 분기다. 형식 오류는 422, 인증 오류는 401, 준비되지 않은 인덱스는 503이다. 자료가 없는 유효한 분기·지역·업종은 200 응답 안에 값 `null`, 상태와 사유를 둔다. 0인 원본 값과 누락은 구분한다.

`observations`의 각 분기는 `sales`, `stores`, `resident`, `floating`을 독립적으로 담는다. 각 항목은 `status`, `reason`, `values`를 포함하며 개별 값도 `value`, `reason`, `reason_message`를 갖는다. 모든 값이 있으면 `available`, 일부가 없으면 `partial`, 원본 행이나 모든 값이 없으면 `unavailable`이다. 상주·길단위인구는 행정동 기준으로 업종과 무관하게 동일한 값을 반환한다. 응답의 `source_columns`, 분기 기준, 스냅샷 생성 시각·공개일 검증 상태와 `warnings`를 함께 해석한다.

## FE·BE 연동 시 확인

- FE의 H10을 서울 8자리 코드로 자르거나 FE 업종명을 서울 CS 코드에 임의로 연결하지 않는다. 공식 코드 대응·기간별 경계와 BE 요청 계약을 확인한다.
- FE의 기존 추이 DTO는 월 단위다. 분기 자료 표시 및 `null` 사유 표시는 FE·BE의 별도 계약과 구현이 필요하다. AI 데이터의 보유가 실제 FE 연결 완료를 뜻하지 않는다.
- 현 API는 원본을 설명하는 연구용 내부 DTO다. 매출액 추정인지, 주민등록·상주·길단위인구 중 무엇인지와 기준 분기를 구분해 표시한다.
- 원본의 과거 개편, 자료 공개 및 정정 이력·이용조건과 실제 제공 가능 범위를 검토한다. 원본만으로 임대료·이익·창업 성공률은 제공하지 않는다.

# 데이터 확보·입력 가이드

현재 구현은 공식 파일을 **내려받은 뒤 공통 형식으로 변환하는 CSV 가져오기**와 공통 관측 스키마를 제공한다. 서울 매출·점포·상주인구·길단위인구·영역 CSV가 로컬에 확보되어 있다. 실제 전국 데이터 수집, 인증키 발급, 과거 공개일 복원 및 실제 데이터의 모델 성능 검증이 완료된 상태는 아니다. 출처 후보와 미확인 조건은 [데이터 목록](../config/data_sources.json)에 기록한다.

## 데이터 범위

| 범위 | 기능 방향 | 현 상태 |
|---|---|---|
| 전국 공통 점포·인구 | 수요·경쟁 등 기본진단 | 공통 입력 스키마 준비. 전국 수집 어댑터는 후속 작업 |
| 서울 행정동×업종×분기 | 추정매출 성장 예측 | 매출·점포 CSV 확보 및 가져오기 점검. 과거 공개일·버전 검증 필요 |
| 다른 지역 매출·상권 지표 | 지역별 확장 | 공개목록 후보만 정리. 단위·기간·정의·지역별 오차 검증 필요 |

상가정보의 최신 현황 한 번으로 성장 예측 라벨이나 과거 폐업률을 만들 수 있다고 가정하지 않는다. 과거 분기별 이력, 출처 변경 및 코드 변경을 확인한다. 유동인구를 주민등록 인구로 바꾸거나 없는 매출을 0으로 채우지 않는다.

## 자료 보관 폴더

지역과 데이터 기준별로 원본을 분리한다. 원본 ZIP은 해당 폴더에 그대로 보관하고, 압축을 푼 CSV도 같은 폴더에 둔다. 지역별 원본의 파일명과 수집 시점은 유지한다.

```text
data/
├── raw/                         # 공식 원본 ZIP·CSV
│   ├── seoul/                   # 서울 행정동 기준
│   │   ├── sales/               # 추정매출
│   │   ├── stores/              # 점포·개업·폐업
│   │   ├── resident_population/ # 상주인구
│   │   ├── working_population/  # 직장인구
│   │   ├── floating_population/ # 길단위·생활인구
│   │   └── boundaries/          # 행정동 경계·공간 정보
│   ├── nationwide/              # 전국 공통 자료
│   │   ├── stores/              # 소상공인 상가정보
│   │   ├── resident_population/ # 주민등록 인구
│   │   ├── restaurant_licenses/ # 일반음식점 인허가
│   │   └── sgis/                # SGIS 공간통계
│   ├── gyeonggi/card_sales/     # 경기도 카드매출
│   ├── busan/commercial_change/ # 부산 상권변화지표
│   └── gyeongnam/tourism/       # 경남 관광상권
└── processed/                   # 원본에서 변환한 공통 형식
    ├── seoul/
    ├── nationwide/
    ├── gyeonggi/
    ├── busan/
    └── gyeongnam/
```

`data/` 전체는 Git에서 제외하므로 복제 시 빈 폴더도 전달되지 않는다. 필요한 폴더는 새 환경에서 아래 명령으로 만들 수 있다.

```bash
mkdir -p data/raw/seoul/{sales,stores,resident_population,working_population,floating_population,boundaries} \
  data/raw/nationwide/{stores,resident_population,restaurant_licenses,sgis} \
  data/raw/gyeonggi/card_sales data/raw/busan/commercial_change data/raw/gyeongnam/tourism \
  data/processed/{seoul,nationwide,gyeonggi,busan,gyeongnam}
```

이 구조는 전국 데이터 어댑터 구현을 의미하지 않으며, 기존 합성 예제 파일도 별도로 유지한다. 변환된 파일은 원본에 덮어쓰지 않는다.

## 확보한 서울 CSV 점검 (2026-11-03)

- 매출 5개 파일 343,167행과 점포 5개 파일 704,470행은 모두 CP949이며, 각각 2021Q1~2025Q4의 20개 분기가 있다. 원본 키(분기·행정동 코드·업종 코드) 중복은 없었다.
- 매출 행 343,167건 모두 점포 키와 일치했다. 점포에만 있는 361,303개 키는 매출 기준 결합에서 제외된다. 매출은 매년 63개 업종, 점포는 100개 업종이며 범위 차이의 이유는 확인되지 않았다. 누락값을 0으로 가정하지 않는다.
- 확보한 점포 CSV의 `유사_업종_점포_수`는 전 행에서 `점포_수 + 프랜차이즈_점포_수`와 일치했다. 현행 [공식 API 명세](https://data.seoul.go.kr/dataList/OA-22172/A/1/datasetView.do)는 `SIMILR_INDUTY_STOR_CO`를 `전체_점포_수`로 표기하고 일반·프랜차이즈를 합한 값이라고 설명한다. 다만 과거 명칭의 공식 매핑 이력은 미확인이다.
- 상주인구와 길단위인구는 각각 2021Q1~2026Q2, 9,350행이며, 분기·행정동 키 중복은 없었다. 영역 파일은 425행, 좌표·면적만 제공하고 다각형 경계나 기준일 열은 없다. [공식 안내](https://data.seoul.go.kr/dataList/OA-22160/A/1/datasetView.do)의 좌표계는 EPSG:5181이며 면적 단위는 확인되지 않았다.
- 현재 어댑터는 매출·점포만 읽는다. 인구·면적은 자동 결합되지 않으며, 공개 시점·코드·면적 단위 확인과 결합 구현이 필요하다.
- 확보한 최신 CSV에 2026-11-03을 일괄 사용 가능일로 입력한 커피·음료 가져오기 점검은 8,396행을 반환했지만, 과거의 다음 분기는 이미 끝났으므로 학습 가능 표본은 0건이다. 이는 과거 공개일의 증거가 없을 때 정상적인 차단이다. 당시 공개 버전과 시점을 확보하기 전 과거 예측 검증으로 해석하지 않는다.

## 공통 관측 파일

CSV 한 행은 **동일한 코드 체계의 지역×업종×분기**다. 코드는 문자열로 보존한다.

| 필드 | 필수 | 의미 |
|---|---|---|
| `region_code` | 예 | 원본 지역 코드. 앞자리 0과 자릿수 보존 |
| `region_code_system` | 예 | 코드 체계. 서울 가져오기는 `seoul-adstrd` |
| `industry_code` | 예 | 원본 또는 합의한 업종 코드. FE의 `coffee`와 자동 동일시하지 않음 |
| `period` | 예 | 관측 분기 `YYYYQ1`~`YYYYQ4` |
| `available_at` | 예 | 해당 행에 포함한 자료가 사용 가능해진 실제 날짜 |
| `source_kind` | 예 | `official` 또는 `synthetic` |
| `source_dataset` | 예 | 출처·버전 식별 이름 |
| `sales_unit` | 예 | `KRW` |
| `sales_amount` | 아니오 | 집계 정의를 확인한 추정매출. 빈 값과 0 구분 |
| `store_count` | 아니오 | 전체 점포 수 |
| `opening_count`, `closing_count` | 아니오 | 제공되는 개업·폐업 수 |
| `resident_population`, `working_population` | 아니오 | 지역 인구. 관측 시점·범위 별도 확인 |
| `floating_population` | 아니오 | 정의를 확인한 유동인구 |
| `area_km2` | 아니오 | 같은 지역 범위의 면적 |

서로 다른 매출 집계 정의를 같은 학습 파일에 혼합하지 않는다. 데이터셋별 정의와 버전은 확보 기록에 보관하며, 모델 산출물에도 식별 정보를 남긴다. 개인정보와 API 인증키는 파일·Git·로그에 넣지 않는다.

## 서울 공식 CSV 가져오기

공식 출처:

- [추정매출·행정동](https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do)
- [점포·행정동](https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do)

공식 파일 확보 절차:

1. 두 상세 페이지의 데이터 설명·이용조건을 확인한다. 매출·점포 모두 **행정동** 자료인지 확인한다.
2. `파일내려받기`와 `전체 파일보기`에서 공통으로 제공되는 연도별 ZIP을 확보한다. 현재 목록에서 확인한 2021~2025년을 우선 조사하되 파일 내용의 실제 분기 범위를 확인한다.
3. 원본 ZIP을 그대로 보존하고 파일명·다운로드 일시·SHA-256·상세 페이지 URL·자료 수정일을 기록한다. ZIP 내부 CSV만 추출한 경우에도 원본을 함께 보관한다.
4. 시트 미리보기는 일부 행만 보일 수 있으므로 전체 자료로 간주하지 않는다. 공식 안내는 Sheet의 1,000건 제한과 전체 CSV 이용을 구분한다.
5. ZIP 내부 매출·점포 CSV의 실제 헤더·코드·금액 단위·분기를 확인하고 다음 가져오기 명령을 사용한다.

사용자가 전달한 CSV를 로컬에 보관했다. 공식 다운로드 버튼의 전송 경로와 파일별 과거 공개 버전은 독립 검증하지 않았다. 자동 다운로드 URL을 임의로 만들거나 비공식 복제본을 공식 원본으로 등록하지 않는다. Open API 수집이 필요하면 공식 인증키 발급·접근 권한과 전체 건수·페이징 명세를 확인한 뒤 별도 수집기를 연결한다.

서울시 공개 안내에는 2026년 7월 제공 기준 변경과 2021년 이후 자료 제공 내용이 있으므로 오래된 복제 파일과 현재 원본을 섞기 전에 산정·공간 기준의 일관성을 확인한다. 데이터 이용조건은 다운로드 시점의 원문을 보존한다.

서울 매출 CSV는 `data/raw/seoul/sales/`, 점포 CSV는 `data/raw/seoul/stores/`에 보관한 뒤 공통 CSV로 변환한다. 영문 OpenAPI 열 이름 또는 한글 다운로드 열 이름을 지원하고, UTF-8(BOM 포함)·CP949를 읽는다.

| 의미 | 영문 열 | 한글 열 |
|---|---|---|
| 관측 분기 | `STDR_YYQU_CD` | `기준_년분기_코드` |
| 지역 | `ADSTRD_CD` | `행정동_코드` |
| 업종 | `SVC_INDUTY_CD` | `서비스_업종_코드` |
| 추정매출 | `THSMON_SELNG_AMT` | `당월_매출_금액` |
| 전체 점포 | `SIMILR_INDUTY_STOR_CO` | `전체_점포_수` 또는 확보한 CSV의 `유사_업종_점포_수` |
| 개업 | `OPBIZ_STOR_CO` | `개업_점포_수` |
| 폐업 | `CLSBIZ_STOR_CO` | `폐업_점포_수` |

`20251`은 `2025Q1`로 변환한다. 지역 코드는 길이를 바꾸지 않는다. 매출에 점포 정보를 **분기·지역·업종이 모두 일치하는 경우에만** 연결하며, 없는 점포 수는 `null`로 남긴다. 전체 점포 수에 일반·프랜차이즈 점포 수를 다시 더하지 않는다. 같은 키가 여러 파일에 중복되면 자동 합산·삭제하지 않고 오류로 중단한다.

`당월_매출_금액`이라는 열 이름만 보고 월 매출이라고 단정하거나 3을 곱하지 않는다. 가져오기는 `sales_basis=as_published`로 원본 값을 보존한다. 실제 원·월평균·분기합계 여부와 자료 개편 전후의 일관성은 공식 설명을 확인한 후 기록한다.

Python 사용 예:

```python
from datetime import date
from pathlib import Path

from pickplace_ai.data import write_observations
from pickplace_ai.ingest import import_seoul

rows = import_seoul(
    sales_paths=[Path("data/raw/seoul/sales/seoul-sales.csv")],
    stores_paths=[Path("data/raw/seoul/stores/seoul-stores.csv")],
    available_at=date.fromisoformat("2026-10-29"),
    industry_code=None,
    sales_basis="as_published",
)
write_observations(rows, Path("data/processed/seoul/seoul.csv"))
```

CLI 사용 예:

```bash
pickplace-ai import-seoul \
  --sales data/raw/seoul/sales/seoul-sales.csv \
  --stores data/raw/seoul/stores/seoul-stores.csv \
  --available-at 2026-10-29 \
  --sales-basis as_published \
  --output data/processed/seoul/seoul.csv
```

여러 연도 파일은 `--sales`·`--stores` 뒤에 경로를 나열한다. 특정 업종으로 제한하려면 CLI의 업종 필터 옵션에 실제 원본 업종 코드를 전달한다. `coffee` 같은 서비스용 별칭으로 원본 코드를 추정하지 않는다. 이미 존재하는 출력 파일은 덮어쓰지 않으므로 새 버전 파일명을 사용한다.

이 예제의 날짜는 형식 설명용이다. 실제 수집일을 과거 최초 공개일로 복사하지 않는다. 과거 파일의 당시 공개일을 모르면 현재 확인한 공개 시점을 사용하고, **과거 예측 당시의 자료로 재현한 검증이 아님**을 평가 기록에 남긴다. 매출과 점포의 공개일이 다르면 모두 사용 가능해진 날짜를 사용한다. 버전별 공개일이 다른 원본은 나눠 가져온 뒤 공통 스키마로 연결한다. 모든 과거 행에 같은 최신 공개일을 넣으면 시간 검증에서 충분한 학습·검증 시점을 구성하지 못할 수 있으며, 이때 임의의 분기말 날짜를 만들어 검증을 통과시키지 않는다.

`source_kind=official`은 입력 출처를 구분하는 기록이며, 파일의 진위·공개일·사용권한을 프로그램이 독립 검증했다는 뜻은 아니다. 합성 예시는 `synthetic`으로 유지하고 학습 파일에서 실제 자료와 섞지 않는다.

## 실제 학습 전 확보 기록

- 공식 URL, 제공기관, 라이선스·재배포 조건
- 실제 관측기간, 공개일, 수집일, 원본 파일명 및 체크섬
- 공간 단위, 코드 체계·기준일과 변경 이력
- 업종 분류·코드 대응표와 변경 이력
- 매출 추정 방식, 금액 단위, 집계 정의
- 결측·비공개·0값 구분, 중복 및 누락 분기
- 지역별 표본 수와 최소 이력 충족 여부

외부 지역에는 전국 공통 특징만으로 검증한 별도 모델이 필요할 수 있다. 시군구 매출을 행정동마다 복제하거나 서울 모델의 결측 입력을 임의로 채워 전국 예측을 제공하지 않는다.

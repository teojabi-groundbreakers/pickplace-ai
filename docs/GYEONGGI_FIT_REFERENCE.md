# 경기도 2026년 2분기 업종별 비교표

2026년 6월 30일 경계, 주민등록인구, 영업 중 수록 업소에서 점포당 주민 수와 1㎢당 점포 수를 계산한다. 각 백분위는 같은 분기에 해당 소분류의 수록 업소가 한 곳 이상 있는 경기 읍면동과 비교한 값이다. 동률은 평균 순위로 계산하고 비교 대상이 한 곳이면 원래 값과 백분위 제공 불가 사유를 반환한다.

## 원천과 적용 범위

| 원천 | 확인한 기준 |
|---|---|
| [행정안전부 구역의 도형](https://business.juso.go.kr/jsi/jsiInquiryDetails?qaPstUnqMgno=134797&qaPstRfrncMgno=134797) | 6월 전체분의 `TL_SCCO_GEMD`. ZIP에는 PRJ가 없어 [제공처 안내](https://business.juso.go.kr/jsi/jsiInquiryDetails?qaPstUnqMgno=125841&qaPstRfrncMgno=125841)의 EPSG:5179로 계산 |
| [행정안전부 주민등록 인구통계](https://jumin.mois.go.kr/statMonth.do) | 6월 말 읍면동 총인구. 경기 H10 604개와 상위 집계를 대조해 합계 13,761,783명을 확인 |
| [소상공인시장진흥공단 상가정보](https://www.data.go.kr/data/15083033/fileData.do) | 주기성 과거 데이터의 2026-06-30 회차. 2026-08-05 등록 후 11월 수정 이력이 있는 파일의 경기 CSV 사용 |

경계 H10 602개와 인구 H10 604개 중 600개가 정확히 일치한다. [7월 1일 행정동 명칭 변경](https://www.anyang.go.kr/main/contents.do?key=289)이 반영된 명학동·병목안동 도형과 6월 인구의 안양8동·안양9동은 연결 대상에서 제외했다. 점포 원본에 남아 있는 7월 변경 코드의 621곳도 제외했다. 인구 원천의 화도읍 본체·동부·남부 출장소는 한 개의 화도읍 도형과 집계 범위가 달라 제외했다.

김포 구래동은 원본 도형이 자기교차한다. 이 도형의 외접 사각형과 겹치는 점포는 배정을 보류했다. 이 과정에서 양촌읍 368곳과 마산동 97곳도 누락되므로 두 지역을 비교에서 함께 제외했다. 원본에서 고리 방향 경고가 난 도형 29개의 기록은 로컬 감사에 남겼다. 단일 고리 또는 서로 겹치지 않는 바깥 고리는 좌표를 변경하지 않고 읽었으며 유효하지 않은 도형은 사용하지 않았다.

점포 위치는 원본 경도·위도(WGS84)를 EPSG:5179로 변환한 뒤 도형 안에 속하는지를 확인한다. 원본 8자리 점포 행정동 코드는 변환 없이 공간 배정 결과 점검에 사용한다. 점포는 소진공 소분류의 영업 중 수록 업소 단위다. 서울의 분기 전체 점포 집계 및 CS 업종 코드와 집계 기준이 다르므로 백분위의 비교 대상은 지역별 참조표 안에서 정한다. [소진공 위치·업종 필드 안내](https://www.data.go.kr/data/15012005/openapi.do)

공간 범위를 점검하는 데 한 도형에 속한 비율 95%를 적용했다. 원본의 같은 행정동에 속한 업소가 한 도형에 95% 미만으로 모인 17개 코드를 확인했다. 해당 업소가 나뉘어 들어간 도형 중 6월 인구와 연결 가능했던 36개 읍면동을 비교에서 보류했다. 고양 송포동으로 수록된 업소 중 1,031곳이 대화동 도형에 있었고 이 가운데 922곳은 다섯 좌표에 집중됐다. 공간 배정 차이가 큰 경우는 원천 주소의 집계 범위를 추가 확인해야 한다. [일산서구 행정동 관내도](https://www.goyang.go.kr/ilswgu/ilswgu01/ilswgu01_5/ilswgu01_5_3.jsp)

## 생성 및 조회 결과

경기 원본 672,679곳에서 업소번호 중복이 없음을 확인했다. 위 조건을 적용해 560개 읍면동과 247개 소분류에서 81,037개 행정동·업종 조합을 만들었다. 비교표에 사용한 617,906곳과 제외된 업소의 사유별 수는 `source-audit.json`에 기록한다.

| 업종 | 비교 동 수 | 수원 인계동 점포당 주민 수 | 수원 인계동 1㎢당 점포 수 |
|---|---:|---:|---:|
| 카페 `I21201` | 560 | 239.29명 · 10.20백분위 | 68.08곳 · 96.60백분위 |
| 분식류 `I21007` | 541 | 741.39명 · 11.30백분위 | 21.97곳 · 91.30백분위 |
| 빵·도넛 `I21001` | 535 | 1,370.45명 · 20.41백분위 | 11.89곳 · 86.70백분위 |

## FastAPI

- `PICKPLACE_GYEONGGI_FIT_REFERENCE_DIR`를 경기 비교표 디렉터리나 분기별 상위 경로로 설정한다. `/health`의 `gyeonggi_fit_reference_periods`에서 로드한 분기를 확인한다.
- `POST /predict` 요청과 이력에 `region_code_system: "mois-resident-admin-10"` 및 해당 경기 H10, 소진공 소분류 코드, `base_period: "2026Q2"`를 넣는다. 경기 비교표가 준비된 분석 기준일은 2026-11-20부터다.
- 응답 `fit_indicators`는 `reference_date`, `reference_available_at`, `cohort_size`, 주민등록인구·면적·수록 업소 수, 두 관측값과 백분위를 포함한다. `store_count_definition: "sbiz_listed_active_businesses"`는 수록 업소 수의 정의를 나타낸다.
- 미수록 지역·업종, 다른 분기, 확보일 이전 요청은 해당 값에 `null`과 사유를 반환한다. 점포 좌표 집계는 규칙 기반 위험도나 성장 예측의 입력으로 사용하지 않는다.

## 로컬 재현

경기 원본과 출처·지문 증빙은 `data/raw/gyeonggi/fit-reference/20260630/`에 보관한다. 월말 인구의 전국 원본은 서울 2분기 준비 때 확보한 파일을 재사용한다. 공간 전처리는 pyshp 3.1.8, pyproj 3.8.0, Shapely 2.2.0에서 확인했다. FastAPI는 완성한 비교표만 읽는다.

```bash
raw=data/raw/gyeonggi/fit-reference/20260630
population=data/raw/seoul/fit-reference/20260630/mois-population/mois-resident-households-202606-national-eupmyeondong.csv
PYTHONPATH=src:. .venv/bin/python scripts/build_gyeonggi_fit_reference.py \
  --archive "$raw/shape/gyeonggi-zone-shapes.zip" \
  --population "$population" \
  --stores "$raw/stores/sbiz-stores-gyeonggi-202606.csv" \
  --evidence "$raw/reference-evidence.json" \
  --output data/processed/gyeonggi/fit-references/2026Q2

PICKPLACE_GYEONGGI_FIT_REFERENCE_DIR=data/processed/gyeonggi/fit-references \
  .venv/bin/uvicorn pickplace_ai.api:app --host 127.0.0.1 --port 8000
```

빌더는 원천의 기준일·지문을 확인하고 새로운 디렉터리에만 쓴다. 결과의 `source-audit.json`에서 경계와 점포 제외 사유를 검토할 수 있다. 원본 ZIP·CSV와 생성한 비교표는 Git에서 제외한다.

# 날짜를 맞춘 서울 적합도 지표

## 자료와 기준

| 자료 | 기준 | 용도 |
|---|---|---|
| [국가데이터처 SGIS 행정구역 통계 및 경계](https://www.data.go.kr/data/15129688/fileData.do) | 2025-06-30, DBF의 `BASE_DATE` 확인 | 서울 행정동 426개 폴리곤 면적. EPSG:5179의 투영 면적을 ㎡로 계산 |
| [행정안전부 주민등록 인구통계](https://jumin.mois.go.kr/statMonth.do) | 2025년 6월 말, 읍면동별 전체 주민등록인구 | 점포당 주민 수의 분자. 주민등록 통계 정의에 따른 값으로 거주불명자와 주민등록 재외국민을 포함하며 외국인을 제외 |
| [서울시 상권분석서비스 점포](https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do) | 2025Q2 서울 원천의 `전체_점포_수` | 점포당 주민 수의 분모와 점포 밀집도의 분자. 서울 원천 분기 집계 |

SGIS ZIP에 함께 들어 있는 인구 통계의 기준연도는 2024년이다. 날짜가 맞는 주민등록인구는 행정안전부 월간 CSV에서 읽는다. 분석 대상은 서울 커피 전문점(`CS100010`), 분식전문점(`CS100008`), 제과점(`CS100005`)이다.


### 2025년 2분기 확인 결과

6월 말 SGIS 행정동 426개 전부를 4월 1일자 공식 연계표의 행정기관 코드에 연결했다. 7월 1일자 변경표의 서울 용신동 분동 시행일도 확인했다. 주민등록 CSV도 서울 동 426개이며 H10 코드 집합이 모두 일치한다. [국가데이터처 4월 연계표](https://kssc.mods.go.kr:8443/ksscNew_web/kssc/common/CommonBoardView.do?strCategoryNameCode=019&strBbsId=kasctnr&nttId=1294&pageIndex=1&plus=Y&categoryMenu=011)

서울 점포 자료의 8자리 원본은 서울시 영역 자료에서 직접 확인했다. 양쪽 면적의 교집합 비중이 각각 99.90% 이상이고 서로 최대 교집합인 도형으로 공간 대조한 뒤, 공식 H10과 서울이 안내하는 8자리 형식에도 일치한 395개 동을 선택했다. 공간 조건 396개 중 공식 코드가 일치하지 않은 한 동은 제외했다. 적용일이 확인되지 않은 서울시 도형은 이 대조에만 사용했고, 면적은 2025년 6월 말 SGIS 도형으로 계산했다. [서울시 행정동 영역](https://data.seoul.go.kr/dataList/OA-22160/S/1/datasetView.do) · [서울시 8자리 코드 형식 FAQ](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingPopulation.do)

| 업종 | 백분위 비교에 포함된 행정동 |
|---|---:|
| 커피 전문점 | 395 |
| 분식전문점 | 395 |
| 제과점 | 392 |

점포 원천 후보 1,273건 중 코드 대응 조건을 충족하지 않은 90건과 점포 수가 유효하지 않은 1건을 제외했다. 2026년 11월 20일 확보본으로 만들어 `reference_available_at=2026-11-20`을 사용한다.

청운효자동 커피 전문점의 실제 FastAPI 응답에서는 점포 115곳·주민등록인구 10,964명·면적 2.568560㎢로 **점포당 95.34명(7.11백분위)**, **44.77곳/㎢(47.97백분위)**를 확인했다.

### 2026년 2분기 원천 점검

| 입력 | 2026-06-30 자료 확인 결과 |
|---|---|
| [행정안전부 주민등록인구](https://jumin.mois.go.kr/statMonth.do) | 6월 말 원본 확보. 서울 427동, 9,289,813명. 시·구·동 합계 일치 |
| [국가데이터처 행정구역분류 연계표](https://kssc.mods.go.kr:8443/ksscNew_web/upload/%ED%95%9C%EA%B5%AD%ED%96%89%EC%A0%95%EA%B5%AC%EC%97%AD%EB%B6%84%EB%A5%98_2026.4.1.%EA%B8%B0%EC%A4%80_20260401014049.xlsx) | SGIS와 H10 공식 대응 427쌍. 7월 10일판과 서울 코드쌍이 동일하고 6월 인구의 H10 427개 모두 일치 |
| [서울시 상권분석서비스 점포](https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do) | 2분기 양수 점포 집계 후보는 커피 전문점 425동, 분식전문점 425동, 제과점 421동 |
| [SGIS 행정동 경계](https://sgis.mods.go.kr/view/pss/openDataIntrcn) | 공개 목록의 최신 행정동 경계 기준은 2025년 6월 말 |
| [서울시 동별 면적 통계](https://data.seoul.go.kr/dataList/DT201004K060005/S/2/datasetView.do) | 매년 12월 기준 |
| [행안부 구역의 도형](https://www.juso.go.kr/externalLink/goUrl?menuId=DT06) | 공식 목록에 2026년 6월 전체분이 있다. 본인인증과 제공 승인이 필요한 항목이며, 서울 파일의 기준일과 도형은 원본 수령 후 확인할 수 있다. [공식 신청 안내](https://eng.juso.go.kr/addrlink/qna/qnaDetail.do?bulletinRefSn=122642&noticeMgtSn=122642&noticeType=QNA) |

인구 원본·공식 연계표·검증 결과·다운로드 목록 응답은 `data/raw/seoul/fit-reference/20260630/`에 보관했다. 서울의 2026년 6월 행정동 SHP(`TL_SCCO_GEMD`)를 받으면 기준일, 좌표계, 동별 코드와 도형 범위를 확인하고 2분기 점포 집계와 연결할 수 있다.

서울 원본 API CSV와 가공 스냅샷의 2026Q2 전체점포·개업·폐업 35,301행이 일치했다. 전체 점포 수는 일반 점포와 프랜차이즈 점포를 포함한다. 점포 원본의 425개 지역 코드 집합은 2021Q1부터 2026Q2까지 같았다. 서울시 문서의 주민등록 행정기관 코드 형식으로 점포 코드의 6월 말 인구 대응 여부만 점검했다. 커피 전문점·분식전문점 각 422동과 제과점 418동이 남았다. 이 수는 경계 검증 전 후보 수다. 일원2동(`11680740`), 상일동(`11740520`), 용신동(`11230536`)의 옛 점포 코드는 자동 연결에서 제외한다. 공식 범위가 확인될 때까지 새 행정동으로 재배분을 보류한다. 코드 형식만 일치하는 추가 28동은 2025년 공간 대조 기준을 통과하지 못했으므로 새 경계에서 재검토한다. [서울시 코드 형식 안내](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingPopulation.do)

2026Q2 면적의 적용일을 확인하기 전까지는 이 분기의 참조표를 발행 보류한다. 2025Q2 참조표를 설정한 FastAPI에 실제 2026Q2 점포 관측을 넣어 두 지표가 `null`과 `fit_reference_period_mismatch`를 반환하는 것을 확인했다.

### 경계 수령 후 면적 정규화

제공 안내로 경계 기준일을 확인하고 ZIP을 보존한다. 파일명에 표시된 배포월, ZIP 수정 시각, DBF 수정일은 경계의 적용일과 구분한다. 서울 전체분의 `TL_SCCO_GEMD.shp`와 같은 이름의 DBF·SHX·PRJ가 필요하다. 공식 가이드의 `EMD_CD`는 행정기관 코드 10자리다. [행안부 구역의 도형](https://business.juso.go.kr/jst/jstAddressDownload?menu=24)

면적 전처리는 FastAPI와 별도로 실행한다. 검증에 사용한 GIS 라이브러리는 pyshp 3.1.8, pyproj 3.8.0, Shapely 2.2.0이다. 경계 기준일 검증 자료에는 `reference_date`, `reference_date_verified`, `source_url`, `reference_date_basis`의 공식 `url`과 설명, 실제 `acquired_at`을 기재한다. `components`에는 `.shp`, `.dbf`, `.shx`, `.prj` 각각의 SHA-256을 적는다. `.cpg`가 있으면 이 파일의 지문도 함께 기록한다. `bridge`에는 공식 URL, 정규화한 연계표 SHA-256과 같은 기준일을 기록한다. 제공처의 자료로 적용일을 확인한 뒤 검증 여부를 기록한다.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/normalize_seoul_gemd_areas.py   --shapefile data/raw/seoul/fit-reference/20260630/shape/TL_SCCO_GEMD.shp   --metadata data/raw/seoul/fit-reference/20260630/boundary-evidence.json   --bridge data/raw/seoul/fit-reference/20260630/crosswalk/kssc-sgis-h10-seoul-20260630.csv   --output data/processed/seoul/gemd-areas-20260630
```

이 단계는 `areas.csv`와 원본 지문·좌표계·기준일이 들어 있는 `manifest.json`, 제외 목록을 만든다. 매니페스트의 `area_source`를 참조표 생성용 `evidence.json`의 `sources.area`에 사용한다. 점포의 서울 8자리 원본 코드와 경계 범위를 대조한 뒤 대응이 확인된 행만 `mapping.csv`에 넣는다.

## FastAPI 응답

기본 `POST /predict` v1.0의 `fit_indicators`에 두 값을 반환한다. 조회 조건은 공식 `seoul-adstrd` 원본 지역·업종 코드, 참조표와 동일한 `base_period`, 실제 참조표 확보일 또는 그 이후의 `as_of`다. 요청의 인구·점포 수·면적을 참조표의 원천값 대신 사용하지 않는다. 기존 진단 및 성장 결과는 별도로 계산한다.

| 경로 | 값 | 단위 |
|---|---|---|
| `fit_indicators.residents_per_store` | 6월 말 주민등록인구 ÷ 2분기 전체 점포 수 | `persons_per_store` |
| `fit_indicators.stores_per_km2` | 2분기 전체 점포 수 ÷ 6월 말 행정동 면적 | `stores_per_km2` |

각 지표에 `value`, `unit`, `percentile`, `status`, `reason`, `percentile_reason`을 담는다. `fit_indicators`의 `base_period`, `reference_date`, `reference_available_at`, `cohort_size`, `store_count`, `resident_population`, `area_km2`로 적용 시점과 계산 근거를 확인할 수 있다.

백분위는 같은 참조표에 포함된 서울 행정동 중 **같은 업종**의 비율을 오름차순으로 정렬해 0~100에 놓는다. 동률은 평균 순위를 사용한다. 밀집도 백분위가 높으면 비교 집단에서 점포가 더 밀집돼 있다는 뜻이다. 두 백분위의 해석은 관측값의 상대적 위치로 제한한다.

| 상태 | 사유 코드 |
|---|---|
| 참조표 미설정 또는 파일·지문 검증 실패 | `fit_reference_not_configured`, `fit_reference_invalid` |
| 비공식 자료 또는 지원하지 않는 코드 체계 | `unsupported_fit_reference_scope` |
| 요청 기준 분기가 다르거나 분석일 이후 확보된 자료 | `fit_reference_period_mismatch`, `fit_reference_after_as_of` |
| 비교표에 해당 지역·업종이 없음 | `fit_scope_not_in_reference` |
| 해당 업종의 참조 행정동이 하나뿐임 | `insufficient_reference_areas`로 백분위만 null |

## FastAPI에 분기별 참조표 연결

`PICKPLACE_SEOUL_FIT_REFERENCE_DIR`에는 개별 참조표 디렉터리 또는 여러 분기 디렉터리를 담은 상위 경로를 설정할 수 있다. 분기 폴더명(`YYYYQn`)과 매니페스트의 `base_period`가 일치해야 한다.

```text
fit-references/
  2025Q2/
    manifest.json
    reference.csv
  2026Q2/          # 원천 검증 후 생성
    manifest.json
    reference.csv
```

FastAPI는 요청의 `base_period`와 정확히 일치하는 표로 두 값과 백분위를 계산한다. 해당 분기가 없으면 `fit_reference_period_mismatch`, 해당 분기의 파일이나 지문 검증에 실패하면 `fit_reference_invalid`를 반환한다. `GET /health`의 `fit_reference_periods`는 시작 시 로드한 분기 목록이다. 일부 분기의 로딩이 실패해도 정상 분기는 조회할 수 있다. 이 작업 환경에서는 `data/processed/seoul/fit-references/`에 기존 2025Q2 표를 연결했다.

## 참조표 만들기

`areas.csv`는 `reference_date,sgis_code,area_m2`, `mapping.csv`는 `reference_date,sgis_code,h10_code,seoul_code` 열을 사용한다. 공식 연계표와 서울시 원본 코드를 기준일과 함께 대조하고, 한 코드가 여러 코드에 대응하는 행은 입력에서 제외한다. `evidence.json`은 `reference_date`와 `sources.area`, `sources.population`, `sources.mapping`을 갖는다. 각 자료에 공식 `url`, 정규화한 입력 파일의 `sha256`, 실제 `acquired_at`, 자료의 `reference_date`를 기록한다. SGIS 원본 ZIP과 대응의 증빙은 로컬 자료와 함께 보관한다.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/build_seoul_fit_reference.py \
  --snapshot data/processed/seoul/indicators-20261109 \
  --areas data/raw/seoul/fit-reference/20250630/areas.csv \
  --population data/raw/seoul/fit-reference/20250630/mois-resident-households-202506-national-eupmyeondong.csv \
  --mapping data/raw/seoul/fit-reference/20250630/mapping.csv \
  --evidence data/raw/seoul/fit-reference/20250630/evidence.json \
  --output data/processed/seoul/fit-reference-2025Q2

PICKPLACE_SEOUL_FIT_REFERENCE_DIR=data/processed/seoul/fit-reference-2025Q2 \
  PYTHONPATH=src .venv/bin/uvicorn pickplace_ai.api:app
```

생성된 `manifest.json`에 원천 지문, 기준일, 확보일, 업종별 비교 동 수, 누락 사유별 제외 건수를 보관한다. API는 시작할 때 `reference.csv`와 지문을 검증한다. 공식 원본과 생성한 표는 로컬 `data/`에 둔다. 이번 원본, 코드·공간 대조 감사표와 재현 스크립트는 `data/raw/seoul/fit-reference/20250630/`에 있다. 참고 기간은 2025Q2이며, 자료가 맞는 후속 분기에 같은 절차를 다시 적용한다.

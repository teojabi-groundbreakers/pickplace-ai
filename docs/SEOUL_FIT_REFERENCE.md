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

# 인구·정책 API 수집과 CSV 저장

확인일: 2026-11-09. API 응답을 로컬 CSV와 수집 정보로 저장하는 도구다. 실제 데이터 수집·학습 반영은 아직 수행하지 않았다. 서울 기존 인구 CSV와 중복 수집하지 않고 전국 인구의 과거 이력을 별도로 준비한다.

| 자료 | API | 인증 | 결과와 현재 상태 |
|---|---|---|---|
| 전국 월별 주민등록인구 | [KOSIS 통계자료](https://kosis.kr/openapi/devGuide/devGuide_0201List.do), 표 `101/DT_1B04005N` | KOSIS API 키와 자료등록 ID | 선택한 월·분류의 CSV 및 메타데이터. 실제 호출과 전체 범위 미검증 |
| 지원사업 공고 | [기업마당 지원사업정보](https://www.bizinfo.go.kr/apiDetail.do?id=bizinfoApi) | 기업마당 API 키 | 최신 공고의 JSON 원본, CSV 및 메타데이터. 이용조건 확인과 실제 호출 미완료 |
| 경기도 지역화폐 | [지역화폐 발행·이용 현황](https://www.data.go.kr/data/15057930/openapi.do) | 제공기관별 이용 신청 확인 필요 | API 기술 명세와 실제 조회 범위 확인 전으로 수집기 미구현 |

## 공통 준비

프로젝트 루트 `/Users/factory/Documents/Projects/PickPlace/pickplace-ai`에서 실행한다. 아래 명령의 키 입력은 터미널에 표시되지 않고 현재 터미널 세션에만 설정된다. 키를 채팅이나 Git에 올리지 않는다. 새 터미널을 열면 다시 설정해야 한다.

```bash
cd /Users/factory/Documents/Projects/PickPlace/pickplace-ai
source .venv/bin/activate
```

수집기는 HTTPS만 사용하고 인증키가 포함된 요청 URL을 출력하지 않는다. `data/`는 Git 제외 대상이다. 기존 결과는 덮어쓰지 않으며 각 수집 결과의 SHA-256과 수집 시각을 기록한다.

## 1. KOSIS 인구

1. [KOSIS 공유서비스](https://kosis.kr/openapi/introduce/introduce_01List.do)에서 활용 신청 후 인증키를 발급받는다.
2. [월별 주민등록인구 통계표](https://kosis.kr/statisticsList/mass/mass_list.jsp?list_id=A_7&org_id=101&process=statHtml&tbl_id=DT_1B04005N&vw_cd=MT_ZTITLE)를 확인한다. [통계자료 개발 가이드](https://kosis.kr/openapi/devGuide/devGuide_0201List.do)의 **자료등록**에서 `101 / DT_1B04005N` 표와 필요한 지역·연령·성별 항목을 선택해 `userStatsId`를 발급받는다. 사용자 등록 내용이 원하는 범위인지 직접 확인한다.
3. API는 요청당 최대 4만 셀이다. 한 달치도 너무 크면 지역 또는 연령별로 자료등록을 나눠 별도 ID·출력 파일로 수집한다. 수집기는 4만 행 이상 또는 API 오류를 거부한다.
4. 아래 예시의 ID는 실제 등록 ID로 바꾼다. 월별 수집 결과가 무엇을 포함하는지는 등록 조건과 응답을 검토해야 한다.

```bash
read -s KOSIS_API_KEY
export KOSIS_API_KEY
python scripts/fetch_kosis_population.py \
  --month 202503 \
  --user-stats-id '발급받은_자료등록_ID' \
  --output data/raw/nationwide/resident_population/kosis-202503.csv
```

위 예시는 2025년 3월 한 달이다. 학습 검토에는 우선 2021년부터 2026년 6월까지의 분기말(3·6·9·12월)을 동일 조건으로 수집하고, 필요하면 다른 월을 보충한다. `--month`와 새 파일명을 바꿔 반복하며 2026년은 실제 확보한 월까지만 조회한다. 같은 달을 지역·연령별로 나누었다면 파일명에도 구분을 넣는다.

동일한 등록 조건으로 분기말을 연속 수집하는 예시다. 기존 파일이 있거나 API 응답에 문제가 있으면 중단한다. 먼저 위의 한 달 수집과 등록 범위를 검증한 뒤 사용한다.

```bash
read "KOSIS_USER_STATS_ID?자료등록 ID: "
for year in 2021 2022 2023 2024 2025 2026; do
  for month in 03 06 09 12; do
    if [[ "$year$month" > "202606" ]]; then break; fi
    if ! python scripts/fetch_kosis_population.py \
      --month "$year$month" \
      --user-stats-id "$KOSIS_USER_STATS_ID" \
      --output "data/raw/nationwide/resident_population/kosis-$year$month.csv"; then
      echo "수집을 중단했습니다: $year$month"
      break 2
    fi
  done
done
```

위 `read` 입력 형식은 macOS 기본 셸인 zsh 기준이다. 중간에 실패하면 원인을 확인하고 아직 생성되지 않은 월부터 다시 실행한다. 지역·연령을 분할한 경우 파일명이 충돌하지 않도록 각 등록 조건에 맞게 별도 이름을 지정한다.

CSV는 원본 응답의 필드와 값을 유지한 변환본이며, 옆의 `.csv.metadata.json`은 통계표·관측월·조회 시각·행 수·파일 해시를 기록한다. 등록 ID 원문은 저장하지 않고 해시만 기록한다. 표본 범위·누락·행정동 코드 체계와 경계 이력은 별도 검증이 필요하다. 통계표 ID와 응답이 맞아도 서울시 8자리 코드와 H10의 대응이 증명되지는 않는다. KOSIS의 과거 통계는 나중에 수정되었을 수 있으므로 취득일이나 최종 수정일을 과거 공개일로 간주하지 않는다. KOSIS 통계 활용 시 출처를 표시한다. [KOSIS 이용약관](https://kosis.kr/openapi/introduce/introduce_02List.do)

## 2. 기업마당 지원사업 공고

[기업마당 API 상세](https://www.bizinfo.go.kr/apiDetail.do?id=bizinfoApi)에서 사용 신청 및 인증키를 발급받는다. 직접 API와 각 공고의 CSV 변환·서비스 활용 조건을 제공처에 확인한 뒤 실행한다. 관련 공공데이터포털 안내는 공공누리 3유형으로 표기되어 있으나 이를 기업마당 직접 API에 그대로 적용한다고 단정하지 않는다.

```bash
read -s BIZINFO_API_KEY
export BIZINFO_API_KEY
python scripts/fetch_bizinfo_api.py \
  --confirm-source-terms \
  --output-dir data/raw/nationwide/policy/bizinfo-20261109
```

`--confirm-source-terms`는 이용조건을 확인한 경우에만 사용한다. 출력 폴더명은 수집일에 맞춰 변경한다. 원본 JSON, CSV, 메타데이터를 함께 저장한다. 선택적으로 `--hashtags 서울` 또는 `--hashtags 경기`를 지정할 수 있으나 해시태그가 모든 지역 대상 공고를 빠짐없이 식별한다고 가정하지 않는다.

기업마당 API는 최신 공고를 제공하므로 과거 정책의 전체 이력과 지역별 실제 집행액을 보장하지 않는다. 공고는 사업의 시행·집행이나 특정 상권의 수혜를 증명하지 않는다. 검색된 공고는 추후 사용자 안내 후보로 검토하고, 현재는 매출 예측 특징으로 자동 반영하지 않는다.

## 사용 전 확인

- 실제 호출은 아직 하지 않았다. 인증·응답 형태·자료 범위·이용 조건을 실제 수집에서 다시 확인한다.
- 지역 코드·지역 경계·관측 단위·실제 공개 시점을 검증하기 전까지 매출 자료와 자동 결합하거나 학습하지 않는다.
- 분석 기준일에 공개되지 않은 자료를 사용하지 않는다. 과거 스냅샷과 공개 이력이 확인되지 않으면 엄격한 과거 시점 검증은 불가능하다.
- 서울사랑상품권 데이터는 공공누리 4유형으로 표시돼 있어 별도 이용·가공 조건을 확인하기 전 수집·가공 대상으로 추가하지 않았다.

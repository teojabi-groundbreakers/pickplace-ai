# 경기도 AI 모델과 FE·BE 응답 계약 점검

확인일: 2026-11-09. 로컬 FE·BE 코드와 AI 코드를 **읽기 전용**으로 비교했다. 실제 서버 연결이나 FE·BE 통합 실행 결과는 아니다. 경기도 모델의 성능 개선이나 모델 파일 존재만으로 서비스 연동이 완료되었다고 볼 수 없다.

## 확인한 프로젝트

| 구분 | 로컬 프로젝트 | 근거 |
|---|---|---|
| FE | `pickplace-frontend` | `src/types/analysis.ts`, `src/lib/validation.ts`, `src/lib/api.ts`, `src/components/ResearchForecast.tsx` |
| BE | `pickplace-backend-develop` | `src/main/java/com/pickplace/backend/`, `docs/api/frontend-api-guide.md` |
| AI | `pickplace-ai` | `src/pickplace_ai/api.py`, `src/pickplace_ai/schemas.py`, `src/pickplace_ai/near_term_api.py`, `scripts/train_gyeonggi_research.py` |

## FE에서 요구하는 분석 응답

FE는 기본 설정의 `/api`와 `/analyses`를 결합하여 `POST /api/analyses`에 `regionCode`, `industryCode`를 전송한다. 정상 응답은 공통 `data/meta` 봉투가 아닌 **응답 본체 전체**를 `isAnalysis`로 검증하며, 응답의 요청 지역·업종이 원요청과 일치해야 한다. 근거: FE `src/lib/config.ts:2`, `src/lib/api.ts:201`, `src/lib/api.ts:208`, `src/lib/validation.ts:122`.

| FE 응답 항목 | FE의 현재 조건 | 경기도 AI와의 차이 |
|---|---|---|
| `id`, `request`, `regionName`, `industryName`, `period`, `analyzedAt`, `source` | 필수. 기간과 시각 형식도 검증 | AI 내부 추론 결과만으로 FE 전체 응답을 구성할 수 없다. BE의 요청·기준정보·응답 통합이 필요하다. |
| `scores.overall`, `growth`, `risk`, `fit` | 네 키가 필요하며 각각 `null` 가능. 값이 있으면 0~100의 `value`, `grade`, `description` 필요 | 경기도 연구 모델은 점수·등급·종합점수·성공확률을 학습하지 않았다. 매출 성장률을 점수로 임의 변환하지 않는다. |
| `factors` | 배열 필수. 각 항목은 `name`, `description`, 숫자 `contribution` | 경기도 학습 모델에 FE용 영향 요인 계약과 검증된 설명은 없다. 빈 배열은 형식상 가능하나 원인이나 기여도를 임의 생성하지 않는다. |
| `trends` | 배열 필수. 각 월의 `sales`, `population`, `stores`, `openings`, `closures`, `averageSales` 키는 각각 숫자 또는 `null`. 월은 오름차순 | 모델은 분기 성장률 연구용이며 월별 현황·인구·점포·개폐업 전체를 제공하지 않는다. 통계는 원천·단위가 검증된 별도 경로에서 제공한다. |
| `competition`, `map`, `summary`, `recommendations` | 배열·지도 객체가 필요하며 `map.center` 좌표와 비어 있지 않은 `summary` 필수. 지도 경계·장소 배열은 빈 배열 가능 | 경쟁·지도·안내문은 AI 모델이 공급하는 결과가 아니다. BE와 지도·통계 데이터의 계약이 필요하며 없는 좌표나 통계를 조작하지 않는다. |
| `researchForecast` | 필드 자체는 생략 또는 `null` 가능. 제공한다면 `experimental_research_only`, `method`, `basePeriod`, `dataAsOf`, `modelVersion`, 검증 여부, 1개 이상의 분기 예측 및 상태·사유·경고 필요 | 원천을 구분한 AI 응답과 BE의 camelCase 변환이 필요하다. `retrospectiveTest`는 선택·null 가능하며, 평가 후 재학습한 최종 모델에 기존 시험 점수를 그 모델의 독립 성능처럼 전달하지 않는다. |

관련 근거: FE `src/types/analysis.ts:27-96`, `src/lib/validation.ts:41-89`, `src/lib/validation.ts:122-186`.

### 서울 설명 하드코딩

`src/components/ResearchForecast.tsx:48`은 연구 결과의 설명을 **“서울시 행정동·업종별 추정매출”**로 고정한다. 경기도 카드매출 결과를 현재 `researchForecast`에 단순 주입하면 원천과 지역이 잘못 표시된다. FE에서 출처·지역별 설명을 지원하고 BE와 필드 계약을 합의하기 전까지 경기도 결과를 그 화면에 연결하지 않는다. 이번 점검에서는 FE를 수정하지 않았다.

## BE 실제 구현 상태

`src/main/java`의 컨트롤러에서 구현된 경로는 `/api/health`와 `/api/health/ready`, 서울 기준 `/api/districts`와 하위 행정동 목록, `/api/business-types`다. `POST /api/analyses`, AI 서버 호출 클라이언트와 FE 분석 응답을 합성하는 컨트롤러는 이 로컬 코드에서 발견되지 않았다. BE `docs/api/frontend-api-guide.md:7`, `:302-308`도 현황·추이·지도·AI 및 분석 API를 미구현으로 명시한다.

BE의 구현된 목록 응답은 `data`와 `meta`를 사용하는 공통 응답이다. FE의 현재 분석 요청은 응답 본체 전체를 분석 객체로 기대하므로, 향후 분석 API의 포장 형식은 별도 합의·어댑터가 필요하다. 목록 API를 그대로 분석 응답 계약으로 간주하지 않는다.

## 현재 AI 응답과 경기도 모델

- 기존 `src/pickplace_ai/api.py:211`의 `POST /predict`는 내부 `growth`, `risk`, `fit`, `warnings` 등을 반환한다. FE의 전체 분석 화면 DTO가 아니며 경기도 연구 모델의 별도 산출물을 읽는 로더도 아니다.
- `src/pickplace_ai/near_term_api.py:268`의 단기 연구 API는 서울 `seoul-adstrd` 체계를 검증한다. 경기도 코드 요청에 그대로 재사용할 수 없다.
- 경기도 모델은 `scripts/train_gyeonggi_research.py`의 연구용 학습·예측 함수와 별도 모델 파일로 존재한다. 작성 시점에 전용 FastAPI 엔드포인트, BE 호출, FE 연동은 구현·검증되지 않았다.
- 기존 공통 `Observation.sales_unit`은 `KRW`로 고정되어 있다(`src/pickplace_ai/schemas.py:21`). 경기도 원천 매출 단위와 월별 금액 합산 기준은 확인되지 않았으므로 임의로 원화로 표시하거나 기존 스키마에 맞춘 것으로 취급하지 않는다.
- FE·BE의 공개 행정동 H10 코드와 경기도 원천 코드의 동등성 및 시점별 경계는 미검증이다. 업종의 서울 원천 코드와 경기도 업종 코드 연계표도 확인되지 않았다. 같은 이름이나 자릿수만으로 변환하지 않는다.
- 경기도 자료는 과거 최초 공개 시점과 지역·업종 이력, 매출 단위·집계 정의가 미확인인 수정 스냅샷이다. 모델·단순 기준 평가 모두 후향 연구이며 운영 점수, 전향 정확도 또는 사업 성공 확률이 아니다. 미지원·결측은 `null`과 명시적인 상태·사유로 처리한다.

## 남은 작업

1. 제공기관에 매출 단위·월별 합산 가능성·공개 및 정정 이력·지역/업종 코드 적용 기준을 확인한다.
2. AI–BE 사이에 경기도 원천 코드·기준분기·성장률·데이터 취득일·모델 버전·연구 상태·경고·불가 사유를 포함하는 내부 계약을 합의한다. 경기도 전용 FastAPI 구현 여부와 모델 승인 범위는 별도 결정한다.
3. BE에서 공식적으로 검증된 코드 대응, 실제 자료 조회, AI 호출·오류 처리와 FE 분석 응답 통합을 구현한다. 기존 서울 목록이 전국 데이터를 제공한다고 가정하지 않는다.
4. FE와 BE가 경기도 데이터 출처 표시, 응답 포장 형식, 불가 상태와 각 화면 항목의 책임을 합의한다. FE의 서울 고정 설명을 일반화하기 전에는 경기 연구 결과를 해당 컴포넌트에 노출하지 않는다.
5. 통합 테스트에서 지원·미지원 지역과 업종, 결측·중복, 모델 장애, 원천 단위 미확인, 잘못된 응답과 부분 실패를 검증한다.


## 2026-11-09 이후 AI 측 추가 구현

위 점검 당시와 달리 AI 프로젝트에 서울·경기 공동 연구 모델과 `/research/unified/predict`가 추가됐다. 단 FE 전체 분석 응답을 제공하는 API는 아니고 BE 분석 API 및 호출 구현, FE의 서울 고정 문구와 코드 대응 문제는 해결되지 않았다. 모델이 제공하지 못하는 점수와 통계·지도 등은 null 및 사유를 반환한다. 실제 FE/BE 연결 검증은 수행하지 않았다. [통합 API 설명](UNIFIED_API.md) 참고.

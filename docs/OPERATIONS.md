# 실행·배포·운영

## 환경변수

| 이름 | 기본값 | 용도 |
|---|---|---|
| `PICKPLACE_MODEL_DIR` | `artifacts/current` | 운영자가 지정한 로컬 모델 디렉터리 |
| `PICKPLACE_API_KEY` | 미설정 | 설정 시 predict/model-info에 X-API-Key 요구 |
| `PICKPLACE_REQUIRE_API_KEY` | `false` | true인데 키가 없으면 시작 실패 |
| `PICKPLACE_ALLOW_SYNTHETIC` | `false` | 합성 요청·합성 모델 명시적 허용 |
| `PICKPLACE_ALLOW_EXPERIMENTAL` | `false` | 검증 전 모델 명시적 허용 |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `.env.example`에서 1 권장 | 다른 서버·DB와의 CPU 경쟁 제한 |

`.env.example`은 예시다. 일반 uvicorn 실행은 `.env`를 자동 로딩하지 않으므로 환경변수를 직접 전달하거나 별도 안전한 설정 방식을 사용한다. 실제 키를 Git·로그·문서에 넣지 않는다.

## 상태와 실패 처리

- `/health`의 status ok는 기본 API가 살아 있다는 뜻이다. `growth_model_ready`와 `growth_model_reason`을 함께 확인한다.
- 모델이 없어도 API는 기본진단을 수행한다. 필수 지표가 없는 지수는 null이다.
- 실제 데이터 요청이 실패했다고 합성 데이터나 합성 모델로 전환하지 않는다.
- 모델과 요청의 코드 체계·지역·업종·출처가 맞지 않으면 성장 예측을 제공하지 않는다.
- 모델의 학습·선택·평가에 사용한 정보가 요청의 `as_of`보다 미래이면 차단한다.
- 예측 대상 분기가 `as_of` 기준 이미 끝났으면 미래 예측으로 반환하지 않는다.
- 운영에서는 Spring Boot가 실제 분석일을 지정한다. 내부 API 호출자의 임의 과거 날짜를 현재 예측으로 표시하지 않는다.

## Docker

Dockerfile과 compose.yaml은 AI 서비스만 정의한다. 실제 Docker 빌드·컨테이너 실행 여부는 WORKLOG.md에 기록한다. 학습은 이미지 시작 과정에서 수행하지 않는다.

```bash
export PICKPLACE_API_KEY='운영 환경에서 별도로 설정한 값'
docker compose build
docker compose up -d
docker compose logs ai-api
```

서비스 이름과 네트워크는 compose.yaml을 기준으로 한다. 컨테이너 8000 포트는 host에 공개하지 않고 `pickplace-ai-internal` 네트워크의 메인 BE에서 호출한다. 별도 Compose 프로젝트의 Spring Boot는 이 네트워크에 명시적으로 참여해야 한다. 같은 Lightsail에 있다는 이유만으로 서로 다른 Compose 네트워크가 연결되지는 않는다.

```yaml
# Spring Boot 측 Compose에 추가할 네트워크 예시
networks:
  ai_internal:
    external: true
    name: pickplace-ai-internal
```

Spring Boot 서비스의 networks에도 ai_internal을 추가하고 내부 주소를 `http://ai-api:8000`으로 설정한다. 실제 구성과 이름 충돌 여부는 BE와 합의한다.

`artifacts/current`를 읽기 전용으로 마운트한다. 모델이 없거나 승인되지 않은 상태에서 up해도 성장 예측은 준비 완료로 표시하지 않는다. 합성·실험 플래그는 운영 compose에서 false를 유지한다.

## 모델 배포

1. 공식 데이터·정의·지역별 검증과 오류 한계를 확인한다.
2. 모델·전처리·메타데이터·평가 문서를 동일 버전으로 보관한다.
3. 팀 검토·승인 증거와 지원 범위를 남긴 뒤 운영 release_status를 관리한다. 학습 명령은 자동 승인하지 않는다.
4. 이전 모델을 보존하고 새 버전을 운영 경로에 배치한다.
5. 컨테이너를 재시작하고 health/model-info 및 대표 실제 요청을 확인한다.

메타데이터의 상태 문자열을 바꿨다는 사실만으로 실제 성능 검증이 끝난 것은 아니다. Python major/minor와 기록된 패키지 버전을 맞춘다. LightGBM 산출물은 해당 버전과 OpenMP 실행 환경도 준비해야 한다.

## 검증 명령

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
ruff check .
ruff format --check .
python -m pip check
```

실제 AWS·도커·Spring Boot 연동은 별도의 통합 검증이다. 로컬 TestClient 통과를 운영 배포 완료로 기록하지 않는다.

## Git 추적 범위

- 소스, 테스트, 문서, Docker 설정, pyproject.toml, requirements.lock과 비밀값 없는 .env.example을 추적한다.
- examples/의 JSON은 출처가 synthetic으로 표시된 참고 자료이므로 추적한다. 새로 생성하는 실행 요청·응답은 reports/ 등 제외 경로에 보관한다.
- .venv/·Python 캐시·패키지 메타데이터·개발 도구 캐시를 제외한다.
- data/·artifacts/·reports/·모델 파일·실험 산출물은 제외한다. 모델·대용량 원본은 별도 저장소로 전달한다.
- 실제 .env와 변형 파일, 인증서·개인키·로컬 자격증명 파일을 제외한다.
- 운영 키나 개인정보를 예제·테스트 fixture에 복사하지 않는다. .gitignore는 파일 내용 검사나 이미 커밋된 비밀정보의 삭제를 대신하지 않는다.
- 초기 main 등록 이후의 공동 개발은 팀 Git Flow와 리뷰 절차를 따른다. 원격 브랜치 보호 규칙은 실제 적용 여부를 별도 확인한다.

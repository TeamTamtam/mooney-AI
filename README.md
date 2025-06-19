# mooney-AI

**Mooney 프로젝트의 AI 서버 리포지토리입니다.**
이 서버는 사용자 맞춤형 소비 절약 챌린지를 자동 생성하기 위한 **시계열 예측 AI 기능**을 제공합니다. FastAPI 프레임워크와 Prophet, scikit-learn 등의 라이브러리를 활용해 소비 패턴 분석 및 과소비 예측 기능을 구현하였습니다.

## ✨ 프로젝트 개요

\*\*Mooney(무니)\*\*는 예산 내 소비에 어려움을 겪는 **Z세대**를 위한 AI 기반 절약 가계부 서비스입니다.
사용자가 스스로 설정한 예산 안에서 **지속 가능한 소비 습관**을 형성할 수 있도록 다음과 같은 기능을 제공합니다:

* 📊 **Prophet 기반 시계열 예측 모델**을 통해 **다음 주 과소비 예상 카테고리 자동 탐지**
* 🎯 지출 습관 개선을 유도하는 **맞춤형 절약 챌린지 생성**
* 💬 GPT-4o-mini 기반 챗봇 \*\*‘똑똑소비봇’\*\*으로 예산 내 소비 가능 여부 실시간 조언
* 🧩 소비 성공 시 **경험치, 캐릭터 해금, UI 변화 등 게이미피케이션 요소 제공**

무니는 단순한 기록형 가계부가 아닌, 사용자와 상호작용하며 소비 습관을 바꾸는 **AI 소비 파트너**입니다.

---
## 주요 기술 스택

* **FastAPI** – Python 기반의 비동기 웹 프레임워크
* **Prophet** – Facebook에서 개발한 시계열 예측 라이브러리
* **scikit-learn**, **scipy**, **pandas**, **numpy** – 데이터 전처리 및 ML 유틸리티
* **Docker** – 컨테이너 기반 배포 환경

## 사전 설치 항목

* Python 3.10
* Docker (선택 사항)

## 설치 및 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/TeamTamtam/mooney-AI.git
cd mooney-AI
```

### 2. 의존성 설치 (로컬 실행용)

```bash
pip install --no-cache-dir -r requirements.txt
```

### 3. 서버 실행

#### 개발 모드

```bash
uvicorn app.main:app --reload
```

#### 운영 모드 (멀티 프로세스)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Docker로 실행 (권장)

```bash
docker build -t mooney-ai .
docker run -p 8000:8000 mooney-ai
```

## API 엔드포인트

### 🔹 GET `/`

* 서버 상태 확인용 엔드포인트
* 응답:

  ```json
  { "message": "Time Series Prediction AI API is running." }
  ```

### 🔹 POST `/predict`

* 시계열 소비 예측 요청 처리
* 요청 본문은 최근 12주의 소비 내역을 담은 다음 형식을 따라야 합니다:

#### ✅ 요청 데이터 포맷

```json
{
  "data": [
    {
      "timestamp": "2024-03-04",
      "amount": 22000,
      "expense_category": "식비"
    },
    {
      "timestamp": "2024-03-11",
      "amount": 19800,
      "expense_category": "식비"
    }
    // ... 최대 12주 분량
  ]
}
```

* 필드 설명:

  * `timestamp`: 날짜 (ISO 8601 형식, 예: `"2024-08-12"`)
  * `amount`: 해당 주의 총 지출액 (정수, 단위: 원)
  * `expense_category`: 소비 항목명 (예: `"식비"`, `"교통"`, `"쇼핑"` 등)

#### 🔁 응답 예시

```json
{
  "message": "Prediction received",
  "data": {
    "predicted_amount": 21000,
    "target_week": "2024-08-19",
    "category": "식비"
  }
}
```

※ 실제 예측 로직은 `/app/routes/predict.py` 내부에 구현됩니다.

## 프로젝트 구조

```
mooney-AI/
├── app/
│   ├── main.py              # FastAPI 엔트리포인트
│   ├── routes/predict.py    # 예측 API 라우터
│   ├── models.py            # 요청 데이터 모델 정의
├── requirements.txt         # Python 패키지 목록
├── Dockerfile               # Docker 빌드 파일
```

## 참고 문서
* [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
* [Prophet 공식 문서 (Facebook / Meta)](https://facebook.github.io/prophet/docs/quick_start.html)
* [Docker 공식 문서](https://docs.docker.com/get-started/)



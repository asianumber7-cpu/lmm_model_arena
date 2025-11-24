# 1. Base Image: Python 3.11.14 (Slim 버전으로 용량 최적화)
FROM python:3.11.14-slim

# 2. 시스템 패키지 설치 & 한글 폰트 설정 (Matplotlib 깨짐 방지)
# git: HuggingFace 모델 로드 시 필요할 수 있음
# fonts-nanum: 리눅스 환경 한글 폰트
RUN apt-get update && apt-get install -y \
    git \
    curl \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 설치 (캐싱 활용을 위해 requirements.txt 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 포트 노출 (Streamlit 기본 포트)
EXPOSE 8501

# 7. 헬스체크 (컨테이너 상태 모니터링)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 8. 실행 명령어
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
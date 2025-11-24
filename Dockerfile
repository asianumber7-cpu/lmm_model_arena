# 1. Base Image
FROM python:3.11.14-slim

# 2. 시스템 패키지 설치 & 한글 폰트 설정
RUN apt-get update && apt-get install -y \
    git \
    curl \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. PyTorch 독립 설치
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision torchaudio

# 5. 나머지 의존성 설치
COPY requirements.txt .
# requirements.txt에서 이미 설치한 torch 관련 패키지가 있어도 pip가 알아서 건너뜁니다.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 6. 소스 코드 복사
COPY . .

# 7. 포트 노출
EXPOSE 8501

# 8. 헬스체크
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. 실행 명령어
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
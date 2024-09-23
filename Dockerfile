FROM python:3.10.13-slim-bookworm
WORKDIR /app
    
    
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    gnupg \
    git \
    g++ \
    unixodbc-dev \
    unixodbc

COPY . .
    
RUN pip install --no-cache-dir -r ./requirements.txt
    
CMD ["python", "-u", "./TelegramLangBot/main.py"]

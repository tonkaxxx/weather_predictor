FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 5050
CMD ["python", "assistant.py"]
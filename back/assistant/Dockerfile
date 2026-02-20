FROM python:3.9-alpine AS builder

RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.9-alpine

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 5050
CMD ["python", "assistant.py"]
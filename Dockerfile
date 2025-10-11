FROM python:3.9-slim

WORKDIR /app

RUN pip install requests python-dotenv pandas scikit-learn joblib flask numpy
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
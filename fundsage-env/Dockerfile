FROM python:3.10-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R appuser /app
USER appuser
EXPOSE 7860
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
ENV HF_TOKEN=""
CMD ["python", "main.py"]

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip --timeout 1000 install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "service.py"]

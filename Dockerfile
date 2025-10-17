FROM  python:3.13.5-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 6868

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "6868"]
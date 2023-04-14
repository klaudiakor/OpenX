FROM python:3.10

RUN mkdir /OpenX

WORKDIR /OpenX

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port=80"]


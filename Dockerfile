FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "7860"]


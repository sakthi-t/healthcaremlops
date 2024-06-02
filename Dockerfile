FROM python:3.11
# Create a virtual environment
RUN python3 -m venv /app/venv
# Activate the virtual environment (workaround for RUN)
WORKDIR /app/venv
RUN source bin/activate  # This line might need adjustment based on your venv layout
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]



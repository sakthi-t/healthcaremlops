FROM python:3.11
# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate the virtual environment using bash
WORKDIR /app/venv
RUN bash -c "source bin/activate"  # This activates the venv

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]








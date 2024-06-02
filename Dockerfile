FROM python:3.11
# Create a virtual environment
RUN python3 -m venv /app/venv

# Create a shell script to activate (optional: adjust path if needed)
RUN echo 'source /app/venv/bin/activate' > /app/activate.sh
RUN chmod +x /app/activate.sh

# Execute the shell script
WORKDIR /app/venv
RUN /app/activate.sh

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]



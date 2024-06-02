FROM python:3.11

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install DVC
RUN pip install dvc

# Copy the DVC configuration and pull data
COPY .dvc /app/.dvc
COPY .dvc/config /app/.dvc/config
COPY . /app
WORKDIR /app

# Pull the data from the remote storage
RUN dvc pull

# Expose the port and define the command to run your application
EXPOSE $PORT
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT:-8000}"]

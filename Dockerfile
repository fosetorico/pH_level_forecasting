# Use Python 3.10 slim-buster as the base image
FROM python:3.10-slim-buster

# Update and install necessary dependencies
RUN apt update -y && apt install -y awscli

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the application
CMD ["python3", "app.py"]

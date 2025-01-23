# Use Python 3.9 base image
FROM python:3.9-slim

# Set non-interactive mode to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev libssl-dev libpq-dev curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /var/task

# Copy requirements file first to leverage Docker caching
COPY requirements_linux.txt /var/task/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_linux.txt --target /var/task

# Copy the application files
COPY . /var/task/

# Expose the application port for local testing
EXPOSE 8000


CMD ["app.lambda_handler"]

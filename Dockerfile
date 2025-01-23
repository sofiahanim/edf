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

# Install AWS Lambda Runtime Interface Client
RUN pip install --no-cache-dir awslambdaric

# Copy the application files
COPY . /var/task/

# Expose the application port for local testing (optional)
EXPOSE 8000

# Set the Lambda runtime entry point
CMD ["python3", "-m", "awslambdaric", "app.lambda_handler"]

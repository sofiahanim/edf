# Use a compatible base image for AWS Lambda with Python 3.9 runtime
FROM public.ecr.aws/lambda/python:3.9

# Set non-interactive mode for apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies for the application
RUN yum update -y && \
    yum install -y gcc gcc-c++ make libtool autoconf automake bison gawk sudo \
    postgresql-devel jemalloc-devel && \
    yum clean all && rm -rf /var/cache/yum

# Set working directory
WORKDIR /var/task

# Copy application files
COPY app.py /var/task/
COPY templates /var/task/templates/
COPY requirements_linux.txt /var/task/
COPY data /var/task/data/
COPY static /var/task/static/
COPY cache /var/task/cache/

# Set permissions
RUN chmod -R 755 /var/task

# Upgrade pip to ensure compatibility with dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r /var/task/requirements_linux.txt

# Expose the application port for local testing (optional)
EXPOSE 8000

# Lambda runtime
CMD ["app.lambda_handler"]


# Command for local development
#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]

#NOTE

# Local Environment
#docker build --build-arg RUN_ENV=local -t electforecast-local .
#docker run -e RUN_ENV=local -p 8000:8000 electforecast-local

# AWS Lambda
#docker build --build-arg RUN_ENV=lambda -t electforecast-lambda .
#docker tag electforecast-lambda:latest <ECR_URI>/electforecast/web:latest
#docker push <ECR_URI>/electforecast/web:latest

#docker build -t electforecast-web .
#docker tag electforecast-web:latest 022499009488.dkr.ecr.us-east-1.amazonaws.com/electforecast/web:latest
#docker push 022499009488.dkr.ecr.us-east-1.amazonaws.com/electforecast/web:latest

# Testing

#curl -X GET http://localhost:8000/health

# Test Pagination

# /api/hourlydemand
# curl -X GET "http://localhost:8000/api/hourlydemand?start=0&length=10"
# curl -X GET "http://localhost:8000/api/hourlydemand?start=0&length=200"  # Should cap to 100

# /api/hourlyweather

# curl -X GET "http://localhost:8000/api/hourlyweather?start=0&length=10"
# curl -X GET "http://localhost:8000/api/hourlyweather?start=0&length=200"  # Should cap to 100


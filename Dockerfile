FROM amazonlinux:2

# Install Python 3.9 and dependencies
RUN yum update -y && \
    yum install -y python3.9 python3.9-pip gcc gcc-c++ libtool && \
    yum clean all && \
    rm -rf /var/cache/yum

# Set the working directory
WORKDIR /var/task

# Copy application files
COPY app.py /var/task/
COPY templates /var/task/templates/
COPY requirements_linux.txt /var/task/
COPY data /var/task/data/
COPY static /var/task/static/
COPY cache /var/task/cache/

# Install Python dependencies
RUN python3.9 -m pip install --no-cache-dir --upgrade pip && \
    python3.9 -m pip install --no-cache-dir -r /var/task/requirements_linux.txt

# Expose the application port for local testing (optional)
EXPOSE 8000

# Specify the Lambda runtime entry point
CMD ["app.lambda_handler"]

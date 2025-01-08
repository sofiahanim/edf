# Use a compatible base image for AWS Lambda with Python 3.11 runtime
FROM public.ecr.aws/lambda/python:3.11

# Set non-interactive mode for apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies for the application
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    make \
    libtool \
    autoconf \
    automake \
    bison \
    gawk \
    sudo \
    postgresql-devel \
    jemalloc-devel && \
    yum clean all && \
    rm -rf /var/cache/yum

# Set the Lambda working directory
WORKDIR /var/task

COPY app.py /var/task/
COPY templates /var/task/templates/
COPY requirements_linux.txt /var/task/
COPY data /var/task/data/
COPY static /var/task/static/

# Install Python dependencies
RUN pip install --no-cache-dir -r /var/task/requirements_linux.txt

# Set Lambda entrypoint
CMD ["app.lambda_handler"]
# Use a compatible base image for AWS Lambda with Python 3.9 runtime
FROM public.ecr.aws/lambda/python:3.9

# Set non-interactive mode for yum operations
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies and update glibc
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y gcc gcc-c++ libtool autoconf automake bison gawk sudo wget tar && \
    yum clean all && rm -rf /var/cache/yum

# Upgrade glibc to version 2.28 or later
RUN wget http://ftp.gnu.org/gnu/libc/glibc-2.28.tar.gz && \
    tar -xvzf glibc-2.28.tar.gz && \
    cd glibc-2.28 && \
    mkdir build && \
    cd build && \
    ../configure --prefix=/opt/glibc-2.28 && \
    make -j$(nproc) && \
    make install && \
    cd ../../ && \
    rm -rf glibc-2.28 glibc-2.28.tar.gz

# Add updated glibc to the library path
ENV LD_LIBRARY_PATH=/opt/glibc-2.28/lib:$LD_LIBRARY_PATH

# Set the working directory
WORKDIR /var/task

# Copy application files
COPY app.py /var/task/
COPY templates /var/task/templates/
COPY requirements_linux.txt /var/task/
COPY data /var/task/data/
COPY static /var/task/static/
COPY cache /var/task/cache/

# Set permissions for application files
RUN chmod -R 755 /var/task

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /var/task/requirements_linux.txt

# Expose the application port for local testing (optional)
EXPOSE 8000

# Specify the Lambda runtime entry point
CMD ["app.lambda_handler"]

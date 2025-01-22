FROM public.ecr.aws/lambda/python:3.9

# Set non-interactive mode for yum operations
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/opt/glibc-2.28/lib:$LD_LIBRARY_PATH

# Install dependencies and build tools in one step to reduce layers
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y gcc gcc-c++ libtool autoconf automake bison gawk sudo wget tar && \
    yum clean all && rm -rf /var/cache/yum && \
    mkdir -p /opt && \
    wget https://ftp.gnu.org/gnu/make/make-4.4.tar.gz && \
    tar -xvzf make-4.4.tar.gz && \
    cd make-4.4 && ./configure && make --jobs=$(nproc) && make install && \
    cd .. && rm -rf make-4.4 make-4.4.tar.gz && \
    ln -sf /usr/local/bin/make /usr/bin/make && \
    wget http://ftp.gnu.org/gnu/libc/glibc-2.28.tar.gz && \
    tar -xvzf glibc-2.28.tar.gz && \
    cd glibc-2.28 && mkdir build && cd build && ../configure --prefix=/opt/glibc-2.28 && \
    make --jobs=$(nproc) && make install && \
    cd ../../ && rm -rf glibc-2.28 glibc-2.28.tar.gz

# Set working directory
WORKDIR /var/task

# Copy application files
COPY . /var/task/

# Set permissions
RUN chmod -R 755 /var/task

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /var/task/requirements_linux.txt

# Expose port (for local testing)
EXPOSE 8000

# Lambda runtime entry point
CMD ["app.lambda_handler"]

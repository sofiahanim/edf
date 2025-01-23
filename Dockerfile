FROM public.ecr.aws/lambda/python:3.9

ENV DEBIAN_FRONTEND=noninteractive
ENV MAKEFLAGS="-j1"

# Install development tools and dependencies
RUN yum clean all && \
    yum makecache && \
    yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y gcc gcc-c++ libtool autoconf automake bison gawk sudo wget tar && \
    yum clean all && rm -rf /var/cache/yum

# Install an updated version of GNU Make
RUN wget https://ftp.gnu.org/gnu/make/make-4.4.tar.gz && \
    tar -xvzf make-4.4.tar.gz && \
    cd make-4.4 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf make-4.4 make-4.4.tar.gz

# Link the new make binary
RUN ln -sf /usr/local/bin/make /usr/bin/make

# Install and build glibc
RUN mkdir -p /opt && \
    wget http://ftp.gnu.org/gnu/libc/glibc-2.28.tar.gz && \
    tar -xvzf glibc-2.28.tar.gz && \
    cd glibc-2.28 && \
    mkdir build && \
    cd build && \
    ../configure --prefix=/opt/glibc-2.28 && \
    make && \  # Removed parallel jobs
    make install && \
    cd ../../ && \
    rm -rf glibc-2.28 glibc-2.28.tar.gz

# Set the library path
ENV LD_LIBRARY_PATH=/opt/glibc-2.28/lib:$LD_LIBRARY_PATH

# Set the working directory
WORKDIR /var/task

# Copy application files
COPY . /var/task/

# Set permissions for application files
RUN chmod -R 755 /var/task

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /var/task/requirements_linux.txt

# Expose the application port for local testing (optional)
EXPOSE 8000

# Specify the Lambda runtime entry point
CMD ["app.lambda_handler"]

FROM catthehacker/ubuntu:act-20.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    gcc \
    g++ \
    make \
    libtool \
    autoconf \
    automake \
    bison  # Install bison

# Install glibc 2.34
RUN wget http://ftp.gnu.org/gnu/libc/glibc-2.34.tar.gz && \
    tar -xvzf glibc-2.34.tar.gz && \
    cd glibc-2.34 && \
    mkdir build && cd build && ../configure --prefix=/usr && \
    make -j$(nproc) && make install && \
    cd ../.. && rm -rf glibc-2.34 glibc-2.34.tar.gz

FROM catthehacker/ubuntu:act-20.04

# Install required system dependencies
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
    bison \
    gawk \
    sudo \
    libpq-dev \
    libjemalloc-dev 

# Clear the apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

# Install glibc 2.34
RUN wget http://ftp.gnu.org/gnu/libc/glibc-2.34.tar.gz && \
    tar -xvzf glibc-2.34.tar.gz && \
    cd glibc-2.34 && \
    mkdir -p build && cd build && \
    ../configure --prefix=/usr && \
    make VERBOSE=1 -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf glibc-2.34 glibc-2.34.tar.gz

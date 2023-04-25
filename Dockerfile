FROM akitenkrad/python.cpu.arm64:latest

COPY requirements.txt /tmp
RUN apt update -y && \
    apt upgrade -y && \
    apt install -y hugo && \
    pip install -r /tmp/requirements.txt



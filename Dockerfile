FROM akitenkrad/python.cpu:latest

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y hugo

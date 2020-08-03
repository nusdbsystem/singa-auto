FROM nvidia/cuda:9.0-base-ubuntu16.04

WORKDIR /

COPY sanodemonitor /usr/local/bin

CMD ["sanodemonitor"]
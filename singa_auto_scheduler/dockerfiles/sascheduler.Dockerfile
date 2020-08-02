FROM debian:stretch-slim

WORKDIR /

COPY sascheduler /usr/local/bin

CMD ["sascheduler"]
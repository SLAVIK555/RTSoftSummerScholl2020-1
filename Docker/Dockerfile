FROM python:3.6.7-alpine3.6

LABEL maintainer="Slava Bogomolov <slavabogomoloff@yandex.ru>"

LABEL description="This example Dockerfile install Python and copy README."

#It's for python
RUN apk update \
    && apk add --virtual build-dependencies \
        build-base \
        gcc \
        wget \
        git \
    && apk add \
        bash

RUN apk add --update python-dev python

WORKDIR /home/slava/RTSoftSummerSchool2020/Docker

COPY README /

#RUN apk add --update nginx && \
#rm -rf /var/cache/apk/* && \
#mkdir -p /tmp/nginx/

#COPY files/nginx.conf /etc/nginx/nginx.conf
#COPY files/default.conf /etc/nginx/conf.d/default.conf
#ADD files/html.tar.gz /usr/share/nginx/

EXPOSE 80/tcp

#ENTRYPOINT ["nginx"]

#CMD ["-g", "daemon off;"]

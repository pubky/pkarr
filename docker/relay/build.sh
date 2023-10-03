#!/bin/sh
cd js
yarn install
docker build . -f ../docker/relay/Dockerfile -t nuhvi/pkarr
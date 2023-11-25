#!/bin/bash

# apt 패키지 업데이트
apt update

# 필요한 패키지 설치
apt-get install -y g++
apt install -y software-properties-common

# Python 3.8 PPA 추가
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
ln -sF /usr/share/zoneinfo/Asia/Seoul/etc/localtime
DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
dpkg-reconfigure --frontend noninteractive tzdata

# Python 3.8 설치
apt install -y python3.8
6
69
apt install -y python3.8-dev
apt-get install -y python3.8-distutils

# Python 3.9 설치
apt install -y python3.9
apt install -y python3.9-dev
apt-get install -y python3.9-distutils

# Python 3.10 설치
apt install -y python3.10
apt install -y python3.10-dev
apt-get install -y python3.10-distutils

# Python 3.11 설치
apt install -y python3.11
apt install -y python3.11-dev
apt-get install -y python3.11-distutils

# Git 설치
apt install -y git

# vim 설치
apt-get install -y vim

# 가상환경 생성
apt install -y virtualenv
virtualenv nltkenv8 --python=python3.8
virtualenv nltkenv9 --python=python3.9
virtualenv nltkenv10 --python=python3.10
virtualenv nltkenv11 --python=python3.11

# nltk 설치
source nltkenv8/bin/activate
git config --global http.sslVerify false
pip install git+https://github.com/cwnu-airlab/NLTKo
deactivate

source nltkenv9/bin/activate
pip install git+https://github.com/cwnu-airlab/NLTKo
deactivate

source nltkenv10/bin/activate
pip install git+https://github.com/cwnu-airlab/NLTKo
deactivate

source nltkenv11/bin/activate
pip install git+https://github.com/cwnu-airlab/NLTKo
deactivate

echo "python3.8"
source nltkenv8/bin/activate
python test.py
deactivate

echo "python3.9"
source nltkenv9/bin/activate
python test.py
deactivate

echo "python3.10"
source nltkenv10/bin/activate
python test.py
deactivate

echo "python3.11"
source nltkenv11/bin/activate
python test.py
deactivate

# 스크립트 종료
exit 0

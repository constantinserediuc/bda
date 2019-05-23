#!/bin/bash

yes | sudo apt-get install python3-pip

yes | sudo apt-get install wget
wget https://raw.githubusercontent.com/constantinserediuc/bda/master/req.txt
wget https://raw.githubusercontent.com/constantinserediuc/bda/master/keras.json
pip3 install -r req.txt
rm /home/serediucctin/.keras/keras.json
cp keras.json /home/serediucctin/.keras/

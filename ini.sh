#!/bin/bash

yes | sudo apt-get install python3-pip

yes | sudo apt-get install wget
wget https://raw.githubusercontent.com/constantinserediuc/bda/master/req.txt
pip3 install -r req.txt
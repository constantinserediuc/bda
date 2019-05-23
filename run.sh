#!/bin/bash
hadoop fs -put articles hdfs://bda-m/user/serediucctin/articles
python3 extractor.py articles
python word_dict.py words 2000
python word_dict.py indexes 2000
python txt_to_index.py words word_dict
python nn.py txt_as_idx word_dict


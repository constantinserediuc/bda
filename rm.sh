#!/bin/bash
hadoop fs -rm -R hdfs://bda-m/user/serediucctin/text
hadoop fs -rm -R  hdfs://bda-m/user/serediucctin/words
hadoop fs -rm -R  hdfs://bda-m/user/serediucctin/indexes
hadoop fs -rm -R  hdfs://bda-m/user/serediucctin/word_dict
hadoop fs -rm -R  hdfs://bda-m/user/serediucctin/txt_to_idx

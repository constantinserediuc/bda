from __future__ import print_function
from os import environ

environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

import os
import sys
import json
import logging
from operator import index
from ast import literal_eval as make_dict

from pyspark.sql import SparkSession

rec = 0
def autoinc(a, b):
    global rec
    rec += 1
    return rec
    
def tokenize(a):
    global py_dict
    a = a.strip("\(\)\[\]\,\",\'").replace("\'", '').replace("\"", '').replace(',', '').split(' ')
    # print(len(a))
    # print(a[:10])
    # print(list(py_dict.keys())[:10])
    tokens = [py_dict[elem] for elem in a if elem in py_dict]
    # print(len(tokens))
    tokens.insert(0, '0')
    tokens.append('-1')
    return tokens
    
def add_to_dict(a):
    py_dict = {}
    for x in a:
        # x = line.strip("\(\)\[\]\,\",\'\n").replace("\'", '').replace("\"", '').replace(',', '').split(' ')
        py_dict[x[0]] = x[1]
    return py_dict

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <texts file> <dict file>", file=sys.stderr)
        sys.exit(-1)
        
    logger = logging.getLogger('pyspark')
    if os.path.exists("txt_as_idx"):
        os.system("rm -rf "+"txt_as_idx")

    spark = SparkSession\
        .builder\
        .appName("WordDict")\
        .getOrCreate()

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    py_dict = spark.read.text(sys.argv[2]).rdd.map(lambda x:x['value']).collect()
    py_dict = make_dict(py_dict[0])

    counts = lines.map(lambda x: x.split('], ['))
    counts = counts.map(lambda x: [tokenize(x[0]), tokenize(x[1])])
    counts.saveAsTextFile("txt_as_idx")

    spark.stop()
from __future__ import print_function
from os import environ

# environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'

import os
import sys
import json
import logging
from operator import index

from pyspark.sql import SparkSession

rec = 0
def autoinc(a, b):
    global rec
    rec += 1
    return rec
    
def tokenize(a):
    global py_dict
    a = a.strip("\(\)\[\]\,\",\'").replace("\'", '').replace("\"", '').replace(',', '').split(' ')
    tokens = [py_dict[elem] for elem in a if elem in py_dict]
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
    # dicts = json.loads()
    with open('new_dict.json') as data_file:
        py_dict = json.load(data_file)
    # logger.warn(len(dicts))
    # py_dict = add_to_dict(dicts)
    
    # logger.warn(py_dict)
    counts = lines.map(lambda x: x.split('], ['))
    counts = counts.map(lambda x: [tokenize(x[0]), tokenize(x[1])])
    counts.saveAsTextFile("txt_as_idx")

    spark.stop()
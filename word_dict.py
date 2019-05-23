from __future__ import print_function
from os import environ

environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

import sys
import operator
from operator import add
import random
import logging
import os
import json

from pyspark.sql import SparkSession


rec = 0
def autoinc(a, b):
    global rec
    rec += 1
    return rec
    
def select(a, b):
    return min(int(a), int(b))
    # return choices.pop()
    
def add_to_dict(a):
    py_dict = {}
    for line in a:
        x = line.strip("[[\(\)\,\",\'\n").replace("\'", '').replace("\"", '').replace(',', '').split(' ')
        py_dict[x[0]] = int(x[1])
    return py_dict

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <num_words>", file=sys.stderr)
        sys.exit(-1)
    logger = logging.getLogger('pyspark')
    if sys.argv[1] == "words":
        if os.path.exists("indexes"):
            os.system("rm -rf "+"indexes")
        spark = SparkSession\
            .builder\
            .appName("WordDict")\
            .getOrCreate()
        lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
        counts = lines.flatMap(lambda x: x.strip("\[\]\(\)\,\",\'").replace("\'", '').replace("\"", '').replace(',', '').split(' ')) \
                      .map(lambda x: (x, 1)) \
                      .reduceByKey(add)
        counts.saveAsTextFile("indexes")
        spark.stop()
    else:
        import os
        items = os.listdir(sys.argv[1])

        newlist = []
        for name in items:
            if name.startswith("part"):
                dicts = open(os.path.join(sys.argv[1], name), 'r').readlines()
                py_dict = add_to_dict(dicts)
        logger.warn(len(py_dict))
        sorted_x = sorted(py_dict.items(), key=operator.itemgetter(1), reverse=True)[:int(sys.argv[2])]
        indexes = {"_START": 0, "_STOP": -1}
        print(len(sorted_x))
        for i in range(len(sorted_x)):
            indexes[sorted_x[i][0]] = i+1
            
        with open('new_dict.json', 'w') as outfile:
            json.dump(indexes, outfile)

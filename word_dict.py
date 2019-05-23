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
from ast import literal_eval as make_tuple

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
    spark = SparkSession \
        .builder \
        .appName("WordDict") \
        .getOrCreate()
    logger = logging.getLogger('pyspark')
    if sys.argv[1] == "words":
        if os.path.exists("indexes"):
            os.system("rm -rf " + "indexes")

        lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
        counts = lines.flatMap(
            lambda x: x.strip("\[\]\(\)\,\",\'").replace("\'", '').replace("\"", '').replace(',', '').split(' ')) \
            .filter(lambda x: len(x) > 1) \
            .map(lambda x: (x, 1)) \
            .reduceByKey(add)
        counts.saveAsTextFile("indexes")
        spark.stop()
    else:
        indexes = spark.sparkContext.wholeTextFiles(sys.argv[1]) \
            .map(lambda x: x[1]) \
            .collect()
        pair_word_count = [j for i in indexes for j in i.split('\n')]
        pair_word_count_as_tuple = []
        for i in pair_word_count:
            try:
                pair_word_count_as_tuple.append(make_tuple(i))
            except:
                continue
        logger.warn('words dict length ' + str(len(pair_word_count_as_tuple)))
        sorted_x = sorted(pair_word_count_as_tuple, key=lambda x: x[1], reverse=True)[:int(sys.argv[2])]
        indexes = {"_START": 0, "_STOP": -1}
        for i in range(len(sorted_x)):
            indexes[sorted_x[i][0]] = i + 1
        rdd = spark.sparkContext.parallelize([json.dumps(indexes)],1)

        rdd.saveAsTextFile("word_dict")

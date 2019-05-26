from os import environ

environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.11:0.5.0 pyspark-shell'
environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
import sys
from pyspark.sql import SparkSession
from bs4 import BeautifulSoup # pip install beautifulsoup4
import nltk
nltk.download()
import re
import os
from ast import literal_eval as make_tuple


def word_tokenize(x):
    abstract_lower = x[0].lower()
    content_lower = x[1].lower()
    return (nltk.word_tokenize(abstract_lower), nltk.word_tokenize(content_lower))


def remove_unnecessary_chars(x):
    x = make_tuple(x.asDict()['value'])
    abstract = x[0].replace("\\n", '')
    content = x[1].replace("\\n", '')
    abstract = re.sub("[^a-zA-Z]+", " ", abstract)
    content = re.sub("[^a-zA-Z]+", " ", content)
    return (abstract, content)


def filter_stopwords(x):
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    abstract = x[0]
    content = x[1]
    stops = set(stopwords.words("english"))
    abstract = [i for i in abstract if i not in stops]
    content = [i for i in content if i not in stops]
    return (abstract, content)


def extract(text):
    soup = BeautifulSoup(text[1], 'xml')
    articles = soup.find_all('article')
    out = []
    for a in articles:
        abstract = a.find('abstract')
        content = a.find('body')
        if abstract is None or content is None:
            continue
        out.append((abstract.text, content.text))
    return out


class PubMedInformationExtractor(object):
    def __init__(self):
        self.spark = SparkSession \
            .builder \
            .master("local") \
            .appName("sumary") \
            .getOrCreate()

    def extract_abstract_and_contents(self, directory_path):
        text = self.spark.sparkContext.wholeTextFiles(directory_path)
        text = text.flatMap(extract)
        text.saveAsTextFile("text")

    def extract_words(self):
        extracted_text = self.spark.read.text('text/')
        words = extracted_text.rdd.map(remove_unnecessary_chars) \
            .map(word_tokenize) \
            .map(filter_stopwords)
        words.saveAsTextFile('words')

if os.path.exists("text"):
    os.system("rm -rf "+"text")
if os.path.exists("words"):
    os.system("rm -rf "+"words")

extractor = PubMedInformationExtractor()
extractor.extract_abstract_and_contents(sys.argv[1])
extractor.extract_words()
# hadoop fs -mkdir  hdfs://bda-m/user/serediucctin
# git clone https://github.com/constantinserediuc/bda.git
# hadoop fs -put articles hdfs://bda-m/user/serediucctin/articles
# python3 extractor.py articles
#  python3 word_dict.py words 2000
# 2. python3 word_dict.py indexes 2000
# 3. python3 txt_to_index.py words word_dict
# 4. python3 nn.py txt_as_idx word_dict
# $HOME/.keras/keras.json

# hadoop fs -rm -R hdfs://bda-m/user/serediucctin/text
# hadoop fs -rm -R  hdfs://bda-m/user/serediucctin/words
# hadoop fs -rm -R  hdfs://bda-m/user/serediucctin/indexes

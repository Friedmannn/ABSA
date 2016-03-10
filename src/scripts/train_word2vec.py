import sys
sys.path.append("../../libs/sentence2vec/")
from word2vec import Word2Vec, LineSentence

import logging
import os

input_file = "../../data/laptop.unlabeled.txt"

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

model = Word2Vec(LineSentence(input_file), size=100, window=5,\
        sg=0, min_count=5, workers=8)
model.save("../../models/laptop.word2vec.model")
program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)

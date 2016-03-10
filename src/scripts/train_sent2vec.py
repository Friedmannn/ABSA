import sys
sys.path.append("../")
from dataset import load_dataset
sys.path.append("../../libs/sentence2vec/")
from word2vec import Sent2Vec, LineSentence


import logging
import os

def sent_iter(reviews, linesentence):
    for review in reviews:
        for sent in review.sentences:
            sent_str = ' '.join(sent.words)
            yield sent_str

    for sent in linesentence:
        yield sent


train_file = "../../data/ABSA-15_Laptops_Train_Data.xml"
test_file = "../../data/ABSA15_Laptops_Test.xml"

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

train_reviews = load_dataset(train_file)
#train_sents = sent_iter(train_reviews)
test_reviews = load_dataset(test_file)
#test_sents = sent_iter(test_reviews)

unlabeled_sents = LineSentence("../../data/laptop.unlabeled.txt")

model_1 = Sent2Vec(sent_iter(train_reviews, unlabeled_sents),\
        model_file="../../models/laptop.word2vec.model")

model_1.save_sent2vec_format("../../models/laptop.sent2vec.model")

model_2 = Sent2Vec(sent_iter(train_reviews+test_reviews, unlabeled_sents),
        model_file="../../models/laptop.word2vec.model")
model_2.save_sent2vec_format("../../models/laptop_with_test.sentenc2vec.model")

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)

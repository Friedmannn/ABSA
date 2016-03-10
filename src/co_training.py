from dataset import load_dataset, get_all_categories, dict2list,\
        build_vocab, list2dict
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from gensim.models.doc2vec import TaggedLineDocument, Doc2Vec
import use_bow, use_sent2vec


TRAIN_FILE = "../data/ABSA-15_Laptops_Train_Data.xml"
TEST_FILE = "../data/ABSA15_Laptops_Test.xml"
UNLABELED_FILE = "../data/laptop.unlabeled.txt"
DOC2VEC_MODEL = "../models/laptop.doc2vec.model"


def main():
    # load data set
    training_reviews = load_dataset(TRAIN_FILE)
    
    # build vocab
    vocab = build_vocab(training_reviews, TOPN=1000)
    vocab_index = list2dict(vocab)

    cate_index = get_all_categories(training_reviews)
    cates = dict2list(cate_index)
    n_cates = len(cates)

    bow_X = use_bow.get_X(training_reviews, vocab_index)
    train_labels =  use_bow.get_labels(training_reviews, cate_index)

    # load doc2vec model
    doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL)
    doc2vec_X = use_sent2vec.get_X(training_reviews, doc2vec_model)

    # transform to mono-label form
    M = len(bow_X)
    BOWX = []
    DENSEX = []
    Y = []
    for i in range(M):
        if not train_labels[i]:
            Y.append(n_cates)  # category index from 0 to n_cates-1, n_cates is for None-label
            BOWX.append(bow_X[i])
            DENSEX.append(
        else:
            for y in train_labels[i]:
                Y.append(y)
                BOWX.append(list(bow_X[i]))
                DENSEX.append(list(doc2vec_X[i]))

    # train a naive bayes model on BoW feature
    nb_model = MultinomialNB()
    nb_model.fit(X, np.array(Y))

    # train a svm model on doc2vec feature
    prob = svm_problem(Y, X)
    param = svm_parameter("-s 0 -t 2 -b 1")
    svm_model = svm_train(prob, param)

    #TODO 

    

if __name__ == "__main__":
    main()

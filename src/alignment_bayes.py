from collections import Counter
from dataset import load_dataset, get_all_categories, dict2list,\
        build_vocab, list2dict
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy.sparse import csr_matrix
from use_bow import get_X, get_labels, get_all_categories
from math import log, exp

PRODUCTS = ("Laptops", )
#PRODUCTS = ("Laptops", "Restaurants")


def train_pola_clf(reviews, vocab_index, cate_index):
    X = []
    Y = []
    for review in reviews:
        for sent in review.sentences:
            bow = [0] * len(vocab_index)
            for w in sent:
                if w in vocab_index:
                    i = vocab_index[w]
                    bow[i] = 1
            for opinion in sent.opinions:
                boc = [0] * len(cate_index) # boc: bag of categories
                boc[cate_index[opinion.category]] = 1 # one hot
                x = bow + boc
                X.append(x)
                Y.append(opinion.polarity)

    clf_model = MultinomialNB()
    clf_model.fit(X, np.array(Y))

    return clf_model


def predict_proba(sent, align_model, prior):
    '''
    sent: a list of words
    align_model: a dict, which's key is a pair of category and word
    prior: prior probability of each category, a dict, which's key is category
    '''
    #proba = dict(prior)
    proba = dict()
    for cate in prior:
        proba[cate] = 0
        for w in sent:
            if (cate, w) in align_model:
                proba[cate] += log(1 - align_model[(cate, w)])
            else:
                pass # p *= 1 - 0

    for cate in proba:
        proba[cate] = 1 - exp(proba[cate])
        #print proba[cate]

    norm_factor = 0
    for cate in proba:
        norm_factor += proba[cate]
    for cate in proba:
        if norm_factor != 0:
            proba[cate] /= norm_factor
        #print proba[cate]

    return proba


def predict(sentence, align_model, prior, lev2_model, vocab_index, cate_index, threshold=0.2):
    categories_predict = []
    probs = predict_proba(sentence, align_model, prior)
    for cate in probs:
        if probs[cate] >= threshold:
            categories_predict.append(cate_index[cate])

    bow = [0] * len(vocab_index)
    for w in sentence:
        if w in vocab_index:
            i = vocab_index[w]
            bow[i] = 1
    X = []
    for cate in categories_predict:
        boc = [0] * len(cate_index)
        boc[cate] = 1
        X.append(bow + boc)
    if len(X) > 0:
        P = lev2_model.predict(X)
    else:
        P = []
    pairs_predict = []
    for i in range(len(categories_predict)):
            pairs_predict.append((categories_predict[i], P[i]))

    return pairs_predict


def load_align_model(model_file):
    model = dict()
    with open(model_file) as f:
        for line in f:
            cate, word, p = line.split()
            p = float(p)
            model[(cate, word)] = p

    return model


def get_prior(reviews):
    counter = Counter()
    norm_factor = 0.0
    for review in reviews:
        for sent in review.sentences:
            for opi in sent.opinions:
                counter[opi.category] += 1
                norm_factor += 1

    for cate in counter:
        counter[cate] /= norm_factor

    return counter


def main(product):
    TRAIN_FILE = "../data/ABSA-15_{}_Train_Data.xml".format(product)
    TEST_FILE = "../data/ABSA15_{}_Test.xml".format(product)

    # load data set
    training_reviews = load_dataset(TRAIN_FILE)
    testing_reviews = load_dataset(TEST_FILE)

    # build vocab
    vocab = build_vocab(training_reviews, TOPN=1000)
    vocab_index = list2dict(vocab)

    cate_index = get_all_categories(training_reviews)
    cates = dict2list(cate_index)
    n_cates = len(cates)

    print "Loading alignment model"
    align_model = load_align_model("s2t64.actual.ti.final")

    print "Get prior"
    prior = get_prior(training_reviews)

    print "Training level 2 model..."
    lev2_model = train_pola_clf(training_reviews, vocab_index, cate_index)

    print "Predicting..."
    results = []
    for review in testing_reviews:
        for sent in review.sentences:
            pairs_predict = predict(sent, align_model, prior, lev2_model, vocab_index, cate_index)
            results.append(pairs_predict)

    print "Evaluation"
    opinions = []
    for review in testing_reviews:
        for sent in review.sentences:
            #opis = [(cate_index[opi.category], opi.polarity) for opi in sent.opinions]
            opis = []
            for opi in sent.opinions:
                if opi.category in cate_index:
                    opis.append((cate_index[opi.category], opi.polarity))
            opinions.append(opis)

    TP1 = 0.0
    FP1 = 0.0
    FN1 = 0.0
    for i in range(len(opinions)):
        o = set([pair[0] for pair in results[i]])
        g = set([pair[0] for pair in opinions[i]])
        TP1 += len(o & g)
        FP1 += len(o - g)
        FN1 += len(g - o)
    
    p = TP1 / (TP1 + FP1)
    r = TP1 / (TP1 + FN1)
    if p + r == 0:
        f = 0
    else:
        f = 2. * p * r / (p + r)

    print p, r, f
    
    TP2 = 0.0
    FP2 = 0.0
    FN2 = 0.0
    for i in range(len(opinions)):
        o = set(results[i])
        g = set(opinions[i])
        TP1 += len(o & g)
        FP1 += len(o - g)
        FN1 += len(g - o)
    
    p = TP1 / (TP1 + FP1)
    r = TP1 / (TP1 + FN1)
    if p + r == 0:
        f = 0
    else:
        f = 2. * p * r / (p + r)

    print p, r, f


def test():
    align_model = load_align_model("s2t64.actual.ti.final")
    counter = Counter()
    words = set()
    for (cate, word) in align_model:
        counter[cate] += align_model[(cate, word)]
        words.add(word)

    #print counter
    print len(words)
    print words


if __name__ == "__main__":
    #for pro in PRODUCTS:
    #    print pro
    #    main(pro)
    test()


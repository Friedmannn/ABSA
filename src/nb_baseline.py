from dataset import load_dataset, get_all_categories, dict2list,\
        build_vocab, list2dict, unwrap
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
from scipy.sparse import csr_matrix
from use_bow import get_all_categories
from collections import Counter
from cate_feats import extract_entattri_X, concatenate, get_entity_attribute, extract_entattri, extract_estimated_entattri, extract_estimated_entattri_X
from use_alignment import load_align_model


PRODUCTS = ("Laptops", "Restaurants")
PRED_THRESHOLD = 0.2
DESC_THRESHOLD = 0.6


def get_bow_X(sentences, vocab_index):
    X = []
    dim = len(vocab_index)
    for sent in sentences:
        x = [0] * dim
        for w in sent:
            if w in vocab_index:
                i = vocab_index[w]
                x[i] = 1
        X.append(x)
    return X


def get_cate_labels(sentences, cate_index):
    labels = []
    for sent in sentences:
        label_bag = [cate_index[opi.category] for opi in sent.opinions if opi.category in cate_index]
        labels.append(label_bag)

    return labels


def train_level1_clf(sentences, vocab_index, cate_index, align_model):
    n_cates = len(cate_index)
    X1 = get_bow_X(sentences, vocab_index)
    X2 = extract_estimated_entattri_X(sentences, align_model, threshold=DESC_THRESHOLD)
    train_X = concatenate(X1, X2)

    train_labels = get_cate_labels(sentences, cate_index)
    # transtform to mono-label problem
    M = len(train_X)
    X = []
    Y = []
    for i in range(M):
        if not train_labels[i]:
            Y.append(n_cates)  # category index from 0 to n_cates-1, n_cates is for None-label
            X.append(train_X[i])
        else:
            for y in train_labels[i]:
                Y.append(y)
                X.append(list(train_X[i]))

    clf_model = MultinomialNB()
    clf_model.fit(X, np.array(Y))

    return clf_model


def train_pola_clf(sentences, vocab_index, cate_index):
    X = []
    Y = []
    for sent in sentences:
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


def predict_cate(sentence, lev1_model, vocab_index, cate_index, feat_names, align_model):
    bow = [0] * len(vocab_index)
    for w in sentence:
        if w in vocab_index:
            i = vocab_index[w]
            bow[i] = 1

    entattri_x = extract_estimated_entattri(sentence, feat_names, align_model, DESC_THRESHOLD)
    x = bow + entattri_x
    predict_proba = lev1_model.predict_proba([x])
    categories_predict = []
    for i in range(len(predict_proba[0])-1):
        if predict_proba[0][i] >= PRED_THRESHOLD:
            categories_predict.append(i)
    return categories_predict


def predict(sentence, lev1_model, lev2_model, vocab_index, cate_index, feat_names, align_model):
    bow = [0] * len(vocab_index)
    categories_predict = predict_cate(sentence, lev1_model, vocab_index, cate_index, feat_names, align_model)

    pairs_predict = []
    X = []
    for cate in categories_predict:
        boc = [0] * len(cate_index)
        boc[cate] = 1
        X.append(bow + boc)
        #X.append(x + boc)
    if len(X) > 0:
        P = lev2_model.predict(X)
    else:
        P = []
    for i in range(len(categories_predict)):
            pairs_predict.append((categories_predict[i], P[i]))

    return pairs_predict


def main(product, pred_threshold, desc_threshold):
    global PRED_THRESHOLD, DESC_THRESHOLD
    PRED_THRESHOLD = pred_threshold
    DESC_THRESHOLD = desc_threshold

    TRAIN_FILE = "../data/ABSA-15_{}_Train_Data.xml".format(product)
    TEST_FILE = "../data/ABSA15_{}_Test.xml".format(product)

    # load data set
    training_reviews = load_dataset(TRAIN_FILE)
    training_sentences = unwrap(training_reviews)
    testing_reviews = load_dataset(TEST_FILE)
    testing_sentences = unwrap(testing_reviews)

    # build vocab
    vocab = build_vocab(training_sentences, TOPN=1000)
    feature_names, align_model = load_align_model(product + ".t2s64.actual.ti.final")
    vocab_index = list2dict(list(vocab))

    cate_index = get_all_categories(training_sentences)
    cates = dict2list(cate_index)
    n_cates = len(cates)
    

    print "Training level 1 model..."
    lev1_model = train_level1_clf(training_sentences, vocab_index, cate_index, align_model)
    print "Training level 2 model..."
    lev2_model = train_pola_clf(training_sentences, vocab_index, cate_index)

    print "Predicting..."
    (feat_names, entattri_indexes) = get_entity_attribute(training_sentences)
    results = []
    for sent in testing_sentences:
        pairs_predict = predict(sent, lev1_model, lev2_model, vocab_index, cate_index, feat_names, align_model)
        results.append(pairs_predict)

    print "Evaluation"
    opinions = []
    for sent in testing_sentences:
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


if __name__ == "__main__":
    print "Laptops"
    main("Laptops", 0.2, 0.6)
    main("Restaurants", 0.25, 0.4)



from dataset import load_dataset, get_all_categories, dict2list,\
        build_vocab, list2dict
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy.sparse import csr_matrix

PRODUCTS = ("Laptops", "Restaurants")

def get_X(reviews, vocab_index):
    X = []
    dim = len(vocab_index)
    for review in reviews:
        for sent in review.sentences:
            x = [0] * dim
            for w in sent:
                if w in vocab_index:
                    i = vocab_index[w]
                    x[i] = 1
            X.append(x)
    return X


def get_labels(reviews, cate_index):
    labels = []
    for review in reviews:
        for sent in review.sentences:
            label_bag = [cate_index[opi.category] for opi in sent.opinions if opi.category in cate_index]
            labels.append(label_bag)

    return labels


def predict(X, nb_model, threshold=0.2):
    output = []
    predict_proba = nb_model.predict_proba(X)
    for i in range(len(predict_proba)):
        l = []
        for j in range(len(predict_proba[0])-1):
            if predict_proba[i][j] >= threshold:
                l.append(j)

        output.append(l)

    return output


def microF1(output, ground_truth):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(len(output)):
        o = set(output[i])
        g = set(ground_truth[i])
        TP += len(o & g)
        FP += len(o - g)
        FN += len(g - o)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    if p + r == 0:
        f = 0
    else:
        f = 2. * p * r / (p + r)
    return p, r, f


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

    train_X = get_X(training_reviews, vocab_index)
    test_X = get_X(testing_reviews, vocab_index)

    train_labels = get_labels(training_reviews, cate_index)
    test_labels = get_labels(testing_reviews, cate_index)

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

    # predict
    output = predict(test_X, clf_model, threshold=0.2)

    # evaluation
    p, r, f = microF1(output, test_labels)

    # output
    out_dir = "../data/bow_nb/"
    out_file = out_dir + "laptop.txt"
    with open(out_file, 'w') as out:
        out.write("Precision:\t{}\nRecall:\t{}\nF1:\t{}\n".format(p, r, f))
        print("{}\n{}\n{}".format(p, r, f))


if __name__ == "__main__":
    for pro in PRODUCTS:
        print pro
        main(pro)
        print "=" * 20
    

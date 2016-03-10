from math import log, exp
from sklearn.naive_bayes import GaussianNB
from dataset import load_plain
from evaluation import microF1
import sys
sys.path.append("../libs/libsvm-3.18/python/")
from svmutil import *


TRAINING_X_FILE = "../data/ALIGN_X.txt"
TESTING_X_FILE = "../data/ALIGN_Y.txt"


def load_align_model(model_file):
    model = dict()
    feature_names = set()
    with open(model_file) as f:
        for line in f:
            word, target, p = line.split()
            if target.startswith('E-') or target.startswith('A-'):
                target = target[2:]
                model[(word, target)] = float(p)
                feature_names.add(target)
    return list(feature_names), model


def extract_feature(sentence, feature_names, align_model):
    N = len(feature_names)
    instance = [0.] * N
    for i in range(len(feature_names)):
        f = feature_names[i]
        p = 0.
        for word in sentence:
            if (word, f) in align_model:
                if align_model[(word, f)] == 1.0:
                    instance[i] = 1.
                else:
                    p += log(1.0-align_model[(word, f)])
        p = 1 - exp(p)
        if instance[i] != 1.:
            instance[i] = p

    return instance


def train_level1_clf(sentences, opinions, cate_index, feature_names, align_model):
    if TRAINING_X_FILE:
        training_x_file = open(TRAINING_X_FILE, 'w')
    else:
        training_x_file = None
    n_cates = len(cate_index)
    X = []
    Y = []
    M = len(sentences)
    for i in range(M):
        instance = extract_feature(sentences[i], feature_names, align_model)
        if training_x_file:
            training_x_file.write(str(instance) + '\n')
        instance_opinions = opinions[i]
        for opi in instance_opinions:
            X.append(list(instance))
            Y.append(cate_index[opi.category])

    #clf_model = GaussianNB()
    #clf_model.fit(X, Y)
    prob = svm_problem(Y, X)
    param = svm_parameter("-s 0 -t 2 -b 1")
    clf_model = svm_train(prob, param)

    return clf_model


def predict_cate(sentence, lev1_model, feature_names, align_model, threshold=0.2):
    x = extract_feature(sentence, feature_names, align_model)
    #predict_proba = lev1_model.predict_proba([x])
    p_label, p_acc, p_val = svm_predict([0], [x], lev1_model, '-b 1')
    predict_proba = p_val
    print predict_proba
    labels_predict = []
    for i in range(len(predict_proba[0])):
        if predict_proba[0][i] >= threshold:
            labels_predict.append(i)

    return labels_predict


def get_cate_index(opinions):
    cates = set()
    for line_opinions in opinions:
        for opinion in line_opinions:
            cates.add(opinion.category)

    cates = list(cates)
    cate_index = dict()
    for i in range(len(cates)):
        cate_index[cates[i]] = i
    return cate_index

def get_categories(opinions, cate_index):
    categories = []
    for line_opinions in opinions:
        line_categories = []
        for opinion in line_opinions:
            if opinion.category in cate_index:
                line_categories.append(cate_index[opinion.category])
            else:
                print "DEBUG: Unseen category " + opinion.category
        categories.append(line_categories)

    return categories


def main():
    data_path = "../data/eng_senti/"
    training_english_file = data_path + "Laptops.english.txt"
    training_sentinese_file = data_path + "Laptops.sentinese"
    testing_english_file = data_path + "Laptops.test.english.txt"
    testing_sentinese_file = data_path + "Laptops.test.sentinese"

    feature_names, align_model = load_align_model("t2s64.actual.ti.final")

    training_sentences, training_opinions = load_plain(training_english_file, training_sentinese_file)

    cate_index = get_cate_index(training_opinions)

    lev1_model = train_level1_clf(training_sentences, training_opinions, cate_index, feature_names, align_model)

    outputs = []
    testing_sentences, testing_opinions = load_plain(testing_english_file, testing_sentinese_file)
    for sentence in testing_sentences:
        labels_predict = predict_cate(sentence, lev1_model, feature_names, align_model)
        outputs.append(labels_predict)

    ground_truth = get_categories(testing_opinions, cate_index)

    p, r, f = microF1(outputs, ground_truth)
    print p, r, f


if __name__ == "__main__":
    main()

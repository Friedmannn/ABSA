from dataset import load_dataset, get_all_categories, dict2list
from gensim.models.doc2vec import Doc2Vec
from baseline import get_Y, microF1
import sys
sys.path.append("../../libs/libsvm-3.18/python/")
from svmutil import *


TRAIN_FILE = "../data/ABSA-15_Laptops_Train_Data.xml"
TEST_FILE = "../data/ABSA15_Laptops_Test.xml"
DOC2VEC_MODEL = "../models/laptop.doc2vec.model"


def get_X(reviews, doc2vec_model):
    X = []
    for review in reviews:
        for sent in review.sentences:
            x = doc2vec_model.infer_vector(sent.words).tolist()
            X.append(x)
    return X


def get_labels(reviews, cate_index):
    labels = []
    for review in reviews:
        for sent in review.sentences:
            label_bag = [cate_index[opi.category] for opi in sent.opinions if opi.category in cate_index]
            labels.append(label_bag)

    return labels

def main():
    #load data set
    training_reviews = load_dataset(TRAIN_FILE)
    testing_reviews = load_dataset(TEST_FILE)

    #load doc2vec model
    doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL)

    cate_index = get_all_categories(training_reviews)
    cates = dict2list(cate_index)
    n_cates = len(cates)

    train_X = get_X(training_reviews, doc2vec_model)
    test_X = get_X(testing_reviews, doc2vec_model)

    train_labels = get_labels(training_reviews, cate_index)
    test_labels = get_labels(testing_reviews, cate_index)

    labelwise_acc = []
    labelwise_output = []

    for cate in range(n_cates):
        # train a bonary model
        train_Y = get_Y(train_labels, cate)
        prob = svm_problem(train_Y, train_X)
        param = svm_parameter("-s 0 -t 2 -b 1")
        m = svm_train(prob, param)

        # test
        test_Y = get_Y(test_labels, cate)
        p_label, p_acc, p_val = svm_predict(test_Y, test_X, m, '-b 1')

        labelwise_acc.append(p_acc)
        labelwise_output.append(p_label)

    # evaluation
    p, r, f = microF1(labelwise_output, test_labels)

    # output
    out_dir = "../data/use_doc2vec/"
    out_file = out_dir + "laptop.txt"
    labelwise_acc = [(cates[i], labelwise_acc[i][0]) for i in range(n_cates)]
    labelwise_acc = sorted(labelwise_acc, key=lambda x:x[1])
    with open(out_file, 'w') as out:
        out.write("Precision:\t{}\nRecall:\t{}\nF1:\t{}\n".format(p, r, f))
        print("{}\n{}\n{}".format(p, r, f))
        for cate_i in range(n_cates):
            out.write("{}:\t{}\n".format(labelwise_acc[cate_i][0], labelwise_acc[cate_i][1]))


if __name__ == "__main__":
    main()

    

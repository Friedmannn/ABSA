import sys
sys.path.append("../libs/libsvm-3.18/python/")
sys.path.append("../libs/sentence2vec/")
from word2vec import Word2Vec

from svmutil import *
from dataset import load_dataset
#from feature_extract import extract_unigram
#from feature_extract import extract_labels
from optparse import OptionParser


DATA_PATH = "../../data/"
CORPUS = {"laptop": ("ABSA15_LaptopsTrain/ABSA-15_Laptops_Train_Data.xml", "ABSA15_Laptops_Test.xml"),\
        "restaurant": ("ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml", "ABSA15_Restaurants_Test.xml")}


def bag2vec(bag):
    n_instance = len(bag)
    dim = len(bag[0])
    x = [0.0] * dim
    for j in range(dim):
        for i in range(n_instance):
            if bag[i][j] > 0:
                x[j] = 1.0
                break
    return x


def get_Y(labels, cate):
    n = len(labels)
    Y = [-1] * n
    for i in range(n):
        if cate in labels[i]:
            Y[i] = 1
    return Y


def word2vec_feat(reviews, model):
    examples = []
    for review in reviews:
        example = None
        count = 0.
        for sent in review.sentences:
            for w in sent:
                if w not in model:
                    #print w + " NOT IN MODEL"
                    continue
                if count == 0:
                    example = model[w]
                    count += 1.
                else:
                    example += model[w]
                    count += 1.
        example /= count
        examples.append(example.tolist())

    return examples


def merge_features(X1, X2):
    n = len(X1)
    if n != len(X2):
        print("ERROR!")
        exit(-1)

    X = []
    for i in range(n):
        X.append(X1[i] + X2[i])

    return X


def microF1(output, ground_truth):
    '''
    output: N x M matrix, where N is the number of all categories,
            and M is the number of bags
    ground_truth: M bags of labels
    '''
    N = len(output)
    M = len(output[0])
    gtruth = []
    for i in range(N):
        gtruth.append([-1] * M)
    for bag_i in range(M):
        for l in ground_truth[bag_i]:
            gtruth[l][bag_i] = 1

    TP = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(N):
        for j in range(M):
            if output[i][j] == 1:
                if gtruth[i][j] == 1:
                    TP += 1.
                else:
                    FP += 1.
            elif gtruth[i][j] == 1:
                FN += 1.

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    if p + r == 0:
        f = 0
    else:
        f = 2. * p * r / (p + r)
    return p, r, f


def main():
    optparser = OptionParser()
    optparser.add_option("-p", "--pro", dest="product")
    (options, args) = optparser.parse_args()

    (train_file, test_file) = CORPUS[options.product]
    train_reviews = load_dataset(DATA_PATH + train_file)
    test_reviews = load_dataset(DATA_PATH + test_file)

    n_cates, cate_index = get_categories(train_reviews + test_reviews)
    vocab_size = 1000
    vocab_index = get_vocab(train_reviews, vocab_size)

    train_bags = [extract_unigram(vocab_index, vocab_size, review)\
            for review in train_reviews]
    train_X = [bag2vec(bag) for bag in train_bags]
    train_labels = [extract_labels(cate_index, review)\
            for review in train_reviews]

    test_bags = [extract_unigram(vocab_index, vocab_size, review)\
            for review in test_reviews]
    test_X = [bag2vec(bag) for bag in test_bags]
    test_labels = [extract_labels(cate_index, review)\
            for review in test_reviews]


    # add word2vec feature
    w2v_model_file = "../../models/laptop.word2vec.model"
    w2v_model = Word2Vec.load(w2v_model_file)
    train_X2 = word2vec_feat(train_reviews, w2v_model)
    train_X = merge_features(train_X, train_X2)
    test_X2 = word2vec_feat(test_reviews, w2v_model)
    test_X = merge_features(test_X, test_X2)


    labelwise_acc = []
    labelwise_output = []
    for cate in range(n_cates):
        # train a binary svm model
        train_Y = get_Y(train_labels, cate)
        prob = svm_problem(train_Y, train_X)
        #param = svm_parameter("-s 0 -t 0 -b 1")
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
    out_dir = "results/rbf/"
    out_dir = "results/"
    out_file = out_dir + options.product + ".txt"
    cates = list(cate_index.items())
    cates = sorted(cates, key=lambda x:x[1])
    labelwise_acc = [(cates[i][0], labelwise_acc[i][0]) for i in range(n_cates)]
    labelwise_acc = sorted(labelwise_acc, key=lambda x:x[1])
    with open(out_file, 'w') as out:
        out.write("Precision:\t{}\nRecall:\t{}\nF1:\t{}\n".format(p, r, f))
        print("{}\n{}\n{}".format(p, r, f))
        for cate_i in range(n_cates):
            out.write("{}:\t{}\n".format(labelwise_acc[cate_i][0], labelwise_acc[cate_i][1]))


if __name__ == "__main__":
    main()


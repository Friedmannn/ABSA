from random import shuffle
import numpy as np
import logging
from nb_baseline import predict_cate, train_level1_clf
from dataset import *
from use_alignment import load_align_model
from cate_feats import *
from evaluation import microF1


def cross_valiadation(sentences, predict_thres, descret_thres,\
        align_model, n_fold=10):
    shuffle(sentences)
    M = len(sentences)
    fold_size = M/n_fold
    f_score_sum = 0.
    for i in range(n_fold):
        testing_sentences = sentences[i*fold_size: (i+1)*fold_size]
        training_sentences = sentences[:i*fold_size] + sentences[(i+1)*fold_size:]
        f_score = experiment_once(training_sentences, testing_sentences, predict_thres, descret_thres, align_model)
        f_score_sum += f_score

    return f_score_sum/n_fold


def experiment_once(training_sentences, testing_sentences, predict_thres, descret_thres, align_model):
    vocab = build_vocab(training_sentences, 1000)
    vocab_index = list2dict(list(vocab))
    cate_index = get_all_categories(training_sentences)
    cates = dict2list(cate_index)
    n_cates = len(cates)

    #print "Training"
    clf_model = train_level1_clf(training_sentences, vocab_index, cate_index, align_model, threshold=descret_thres)
    #print "Testing"
    (feat_names, entattri_indexes) = get_entity_attribute(training_sentences)
    results = []
    for sent in testing_sentences:
        predicted_cates = predict_cate(sent, clf_model, vocab_index, cate_index, feat_names, align_model, threshold=predict_thres, descret_thres=descret_thres)
        results.append(predicted_cates)

    # Evaluation
    ground_truth = []
    for sent in testing_sentences:
        labels = set()
        for opi in sent.opinions:
            if opi.category in cate_index:
                labels.add(cate_index[opi.category])
        ground_truth.append(labels)

    p,r,f = microF1(results, ground_truth)

    return f


def grid_search(sentences, align_model):
    predict_thres_range, pred_step = ((0.1, 0.4), 0.02)
    descret_thres_range, desc_step  = ((0.4, 1.0), 0.02)
    logging.basicConfig(filename="grid_search.log", level=logging.DEBUG)

    highest_result = 0

    for predict_thres in np.arange(predict_thres_range[0], predict_thres_range[1], pred_step):
        for descret_thres in np.arange(descret_thres_range[0], descret_thres_range[1], desc_step):
            average_f = cross_valiadation(sentences, predict_thres, descret_thres, align_model)
            logging.info("{}, {}, {}".format(predict_thres, descret_thres, average_f))
            if average_f > highest_result:
                highest_result = average_f
                best_pred_thres = predict_thres
                best_desc_thes = descret_thres

    print "Best Predicting Threshold: {}\n Best Descreting Threshold: {}\n Best F-score:{}".format(best_pred_thres, best_desc_thes, highest_result)


def main(product):
    TRAIN_FILE = "../data/ABSA-15_{}_Train_Data.xml".format(product)
    training_reviews = load_dataset(TRAIN_FILE)
    training_sentences = unwrap(training_reviews)
    feat_names, align_model = load_align_model("t2s64.actual.ti.final")

    grid_search(training_sentences, align_model)


if __name__ == "__main__":
    main("Laptops")

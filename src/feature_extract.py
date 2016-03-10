import sys
sys.path.append("..")
sys.path.append("../../libs/sentence2vec/")
from word2vec import Word2Vec, LineSentence
from feat4miml import get_categories
from feat4miml import get_vocab
from dataset import load_dataset
from optparse import OptionParser
from save_as_mimlmix_format import *
import save_as_mimlmix_format


def extract_unigram(vocab_index, vocab_size, review):
    bag = []
    for sentence in review.sentences:
        instance = [0] * vocab_size
        for w in sentence:
            if w in vocab_index:
                index = vocab_index[w]
                instance[index] = 1
        bag.append(instance)
    return bag


def extract_labels(cate_index, review):
    bag = set()
    for sentence in review.sentences:
        for opinion in sentence.opinions:
            label = cate_index[opinion.category]
            bag.add(label)

    return list(bag)


def word2vec_feat(reviews):
    w2v_model_file = "../../models/laptop.word2vec.model"
    w2v_model = Word2Vec.load(w2v_model_file)
    bags = []
    for review in reviews:
        bag = []
        for sent in review.sentences:
            instance = None
            count = 0.
            for w in sent:
                if w not in w2v_model:
                    continue
                if count == 0:
                    instance = w2v_model[w]
                    count += 1.
                else:
                    instance += w2v_model[w]
                    count += 1.
            instance /= count
            bag.append(instance.tolist())
        bags.append(bag)

    save_sparse_feature(corpus_name="laptop", view_name="word2vec", features=bags)
    save_view_info(view_name="word2vec", dim=100, data_format="sparse", view_type="continuous")


def main():
    optparser = OptionParser()
    optparser.add_option("--train", dest="train_file",
            help="training file name")
    optparser.add_option("--test", dest="test_file",
            help="testing file")
    optparser.add_option('--pro', dest='product')

    (options, args) = optparser.parse_args()
    save_as_mimlmix_format.PATH = "./{}/data/".format(options.product)

    train_reviews = load_dataset(options.train_file)
    test_reviews = load_dataset(options.test_file)

    n_cates, cate_index = get_categories(train_reviews + test_reviews)
    vocab_size = 1000
    vocab_index = get_vocab(train_reviews, vocab_size)

    train_bags = [extract_unigram(vocab_index, vocab_size, review)\
            for review in train_reviews]
    train_labels = [extract_labels(cate_index, review)\
            for review in train_reviews]

    test_bags = [extract_unigram(vocab_index, vocab_size, review)\
            for review in test_reviews]
    test_labels = [extract_labels(cate_index, review)\
            for review in test_reviews]

    save_label_id(cate_index)
    save_view_info(view_name="ngram", dim=vocab_size,\
            data_format="sparse", view_type="discrete")
    features = train_bags + test_bags
    save_sparse_feature(corpus_name=options.product, view_name="ngram", features=features)
    labels = train_labels + test_labels
    save_label(options.product, labels)
    save_partition(len(train_labels), len(test_labels))

    #word2vec
    word2vec_feat(train_reviews+test_reviews)

    print("Done")


if __name__ == "__main__":
    main()

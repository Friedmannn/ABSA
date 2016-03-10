from MixedNaiveBayes.nb import Feature, NaiveBayesClassifier
from MixedNaiveBayes import distributions
from dataset import load_plain
from evaluation import microF1


class UnigramExtractor:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, sentence):
        for word in self.vocab:
            val = 1 if word in sentence else 0
            feature = Feature(name=word, distribution=distributions.Multinomial, value=val)
            yield feature


def build_vocab(sentences, TOPN=1000):
    from collections import Counter
    from nltk.corpus import stopwords
    counter = Counter()
    stw = stopwords.words("english")
    for sent in sentences:
        for w in sent:
            if w not in stw:
                counter[w] += 1

    vocab = counter.most_common(TOPN)
    vocab = [item[0] for item in vocab]
    return vocab


def predict_cate(sentence, nb_model, threshold=0.2):
    predicted_labels = []
    max_p = 0
    for cate in nb_model.labelCounts:
        p = nb_model.probability(sentence, cate)
        max_p = max(max_p, p)
        if p >= threshold:
            predicted_labels.append(cate)
    print max_p
    #probs = nb_model.probabilities(sentence)
    '''
    print probs.most_common(10)
    for cate in probs:
        if probs[cate] >= threshold:
            predicted_labels.append(cate)
    '''

    return predicted_labels


def main():
    data_path = "../data/eng_senti/"
    training_english_file = data_path + "Laptops.english.txt"
    training_sentinese_file = data_path + "Laptops.sentinese"
    testing_english_file = data_path + "Laptops.test.english.txt"
    testing_sentinese_file = data_path + "Laptops.test.sentinese"

    training_sentences, training_opinions = load_plain(training_english_file, training_sentinese_file)
    vocab  = build_vocab(training_sentences)

    featurizer = UnigramExtractor(vocab)
    nb_model = NaiveBayesClassifier(featurizer)

    unfold_sentences = []
    unfold_labels = []
    identical_cates = set()
    for index, line_opinions in enumerate(training_opinions):
        for opinion in line_opinions:
            unfold_labels.append(opinion.category)
            unfold_sentences.append(training_sentences[index])
            identical_cates.add(opinion.category)

    nb_model.train(unfold_sentences, unfold_labels)

    # Testing
    testing_sentences, testing_opinions = load_plain(testing_english_file, testing_sentinese_file)
    #predicted_labels = [predict_cate(sent, nb_model, identical_cates)\
    #        for sent in testing_sentences]
    predicted_labels = []
    for index, sent in enumerate(testing_sentences):
        predicted_labels.append(predict_cate(sent, nb_model, identical_cates))
    ground_truth = [set([opinion.category for opinion in line_opinions])\
            for line_opinions in testing_opinions]

    print predicted_labels

    p, r, f = microF1(predicted_labels, ground_truth)
    print p, r, f
    

if __name__ == '__main__':
    main()

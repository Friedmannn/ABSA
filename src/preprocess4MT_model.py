from dataset import load_dataset
from collections import Counter

PRODUCTS = ("Laptops", "Restaurants")
UNK_FREQUENCY = 0


def merge_opinions(opinions):
    '''
    Merge Opinions which share the same Entity and Polarity
    '''
    old = opinions
    opinions = []
    while len(old) > 0:
        current = old[0]
        old = old[1:]
        current_entity, current_attri = current.category.split('#')
        flag = False
        for opi in opinions:
            opi_entity, opi_attri = opi.category.split('#')
            if opi_entity == current_entity and opi.polarity == current.polarity:
                opi.category = opi_entity + '#' + opi_attri + ' ' + current_attri
                flag = True
                break
        if not flag:
            opinions.append(current)

    return opinions


def normalize(opinion):
    entity, attri = opinion.category.split('#')
    if entity == "LAPTOP" or entity == "RESTAURANT":
        entity = "OVERALL"
    entity = "E-" + entity
    attri = "A-" + attri
    opinion.category = '#'.join((entity, attri))


def do_preprocess(product, test=False):
    if not test:
        FILE = "../data/ABSA-15_{}_Train_Data.xml".format(product)
    else:
        FILE = "../data/ABSA15_{}_Test.xml".format(product)

    reviews = load_dataset(FILE)

    # Replace words with frequency lower than UNK_FREQUENCY with <UNK>
    word_freq_counter = Counter()
    for review in reviews:
        for sentence in review.sentences:
            for w in sentence:
                word_freq_counter[w] += 1
    for review in reviews:
        for sentence in review.sentences:
            sentence.words = map(lambda w: w if word_freq_counter[w] >= UNK_FREQUENCY else '<UNK>', sentence.words)
    
    english_corpus = []
    sentinese_corpus = []

    for review in reviews:
        for sentence in review.sentences:
            if sentence.opinions:
                english_corpus.append(" ".join(sentence.words))
                #english_corpus.append(sentence.raw_text)
                sentinese_sentence = ""
                for opi in sentence.opinions:
                    normalize(opi)
                #sentence.opinions = merge_opinions(sentence.opinions)
                for opi in sentence.opinions:
                    entity, attri = opi.category.split('#')
                    if opi.polarity > 0:
                        polarity = "POSITIVE"
                    elif opi.polarity < 0:
                        polarity = "NEGATIVE"
                    else:
                        polarity = "NEUTRAL"
                    sentinese_sentence += "{} {} {} ; ".\
                            format(entity, attri, polarity)
                    #sentinese_sentence += "{} ".\
                    #        format(opi.category)
                sentinese_corpus.append(sentinese_sentence)

    return english_corpus, sentinese_corpus


def list2file(li, filename):
    with open(filename, 'w') as outfile:
        for sent in li:
            #print sent
            outfile.write(sent.encode('utf8') + "\n")


def save_as_dev(src, target, filename):
    n = len(src)
    tuples = ['\n'.join((src[i], "notree", target[i], '='*8)) for i in range(n)]
    list2file(tuples, filename)


def main(product, test):
    english_corpus, sentinese_corpus = do_preprocess(product, test)
    if not test:
        list2file(english_corpus, "../data/eng_senti/{}.english".format(product))
        list2file(sentinese_corpus, "../data/eng_senti/{}.sentinese".format(product))
        save_as_dev(english_corpus, sentinese_corpus, "../data/eng_senti/{}.dev".format(product))
    else:
        list2file(english_corpus, "../data/eng_senti/{}.test.english".format(product))
        list2file(sentinese_corpus, "../data/eng_senti/{}.test.sentinese".format(product))
        save_as_dev(english_corpus, sentinese_corpus, "../data/eng_senti/{}.test.pll".format(product))

    print "Done"


if __name__ == "__main__":
    for pro in PRODUCTS:
        main(pro, test=True)
        main(pro, test=False)


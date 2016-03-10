from nltk import word_tokenize
import xml.etree.ElementTree as ET
from collections import Counter


class Review:
    '''
    define the data structure of a review
    '''
    def __init__(self, id):
        self.id = id
        self.sentences = []


class Sentence:
    '''
    define the data structure of a sentence
    '''
    def __init__(self, id):
        self.id = id
        self.raw_text = ""
        self.words = []
        self.opinions = []
        self.clauses = []


    def __iter__(self):
        for w in self.words:
            yield w


class Opinion:
    '''
    define the data structure of an opinion
    '''
    def __init__(self, target='', category='', polarity=0, _from=0, to=0):
        pass
        # target - str
        # category - str
        # polarity - +1, 0, -1
        # _from - int
        # to - int
        self.target = target
        self.category = category
        self.polarity = polarity
        self._from = _from
        self.to = to


def pola_atoi(polarity):
    if polarity == 'positive':
        return +1
    if polarity == 'negative':
        return -1
    if polarity == 'neutral':
        return 0


def load_dataset(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    reviews = []
    for review_node in root.iter('Review'):
        review = Review(review_node.get('rid'))
        for sentence_node in review_node.iter('sentence'):
            sentence = Sentence(sentence_node.get('id'))
            raw_text = sentence_node.find('text').text
            words = word_tokenize(raw_text)
            words = [w.lower() for w in words]
            sentence.words = words
            sentence.raw_text = raw_text

            # clauses
            for clause_node in sentence_node.iter('clause'):
                sentence.clauses.append(clause_node.text)

            # opinions
            for opi in sentence_node.iter('Opinion'):
                opinion = Opinion()
                opinion.target = opi.get('target')
                opinion.category = opi.get('category')
                opinion.polarity = pola_atoi(opi.get('polarity'))
                if opinion.target:
                    opinion._from = int(opi.get('from'))
                    opinion.to = int(opi.get('to'))

                sentence.opinions.append(opinion)
            review.sentences.append(sentence)

        reviews.append(review)

    return reviews

def unwrap(reviews):
    '''
    return a list of sentences
    '''
    sentences = []
    for rv in reviews:
        sentences += rv.sentences
    return sentences


def get_all_categories(sentences):
    i = 0
    cate_index = {}
    for sent in sentences:
        for opinion in sent.opinions:
            if opinion.category not in cate_index:
                cate_index[opinion.category] = i
                i += 1

    return cate_index


def build_vocab(sentences, TOPN=1000):
    from nltk.corpus import stopwords
    stw = stopwords.words("english")
    counter = Counter()
    for sent in sentences:
        for w in sent:
            if w not in stw:
                counter[w] += 1
    
    vocab = counter.most_common(TOPN)
    vocab = [item[0] for item in vocab]
    
    return vocab


def dict2list(dic):
    '''
    return a list of keys in a dict,
    ordered by the values in this dict
    '''
    items = dic.items()
    items = sorted(items, key=lambda x:x[1])
    items = [item[0] for item in items]
    return items


def list2dict(li):
    '''
    Reverse function of dict2list
    '''
    d = {}
    for i in range(len(li)):
        d[li[i]] = i
    return d


def load_plain(english_file, sentinese_file):
    sentences = []
    opinions = []

    with open(english_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line.split())

    with open(sentinese_file) as f:
        for line in f:
            line = line.strip()
            segments = line.split(';')
            line_opinions = []
            for segment in segments:
                if not segment:
                    continue
                entity, attribute, polarity = segment.split()
                entity = entity[2:]
                attribute = attribute[2:]
                polarity = pola_atoi(polarity.lower())
                line_opinions.append(Opinion(category='#'.join([entity, attribute]), polarity=polarity))
            opinions.append(line_opinions)

    return sentences, opinions


def test_get_all_categories():
    import sys
    reviews = load_dataset(sys.argv[1])
    cate_index = get_all_categories(reviews)
    print cate_index
    print dict2list(cate_index)


def test():
    import sys
    reviews = load_dataset(sys.argv[1])
    with open(sys.argv[1][:-4] + '.linesentence.txt', 'w') as out:
        for rv in reviews:
            for sent in rv.sentences:
                raw_str = ' '.join(sent.words)
                out.write(raw_str + '\n')

def test_clause():
    import sys
    reviews = load_dataset(sys.argv[1])
    for rv in reviews:
        for sent in rv.sentences:
            for clause in sent.clauses:
                print(clause)


def count_cate():
    import sys
    from collections import Counter
    reviews = load_dataset(sys.argv[1])
    counter = Counter()
    for rv in reviews:
        for sent in rv.sentences:
            for opinion in sent.opinions:
                counter[opinion.category] += 1

    for item in counter.most_common():
        print item


def test_load_plain():
    from sys import argv
    sentences, opinions = load_plain(argv[1], argv[2])
    print len(sentences) == len(opinions)
    #print sentences
    #print opinions
    for line_opinions in opinions:
        for opinion in line_opinions:
            print opinion.category, opinion.polarity, 
        print

if __name__ == '__main__':
    #test()
    #test_get_all_categories()
    #test_load_plain()
    count_cate()


def parse_sentinese(sentinese):
    default_ent = "E-OVERALL"
    default_attr = "A-GENERAL"

    clauses = sentinese.split(';')
    polarity_set = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}

    for clause in clauses:
        entities = set()
        attrs = set()
        polarity = 0
        for word in clause.split():
            if word.startswith("E-"):
                entities.add(word)
            elif word.startswith("A-"):
                attrs.add(word)
            elif word in polarity_set:
                polarity += polarity_set[word]
            else:
                pass

        if polarity != 0:
            if len(entities) > 0 and len(attrs) == 0:
                attrs.add(default_attr)
            elif len(attrs) > 0 and len(entities) == 0:
                entities.add(default_ent)

        if polarity > 0:
            polarity = 1
        elif polarity < 0:
            polarity = -1
        # transfer [0, 1, -1] to  ["NEUTRAL", "POSITIVE", "NEGATIVE"]
        polarity = ["NEUTRAL", "POSITIVE", "NEGATIVE"][polarity]

        for ent in entities:
            for attr in attrs:
                opinion = (ent, attr, polarity)
                yield opinion


def read_set(filename):
    s = set()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                s.add(line)

    return s


def f1_score(ground_truth, predict):
    N = len(ground_truth)
    true_positive = 0.
    false_positive = 0.
    false_negative = 0.
    for i in range(N):
        true_positive += len(ground_truth[i] & predict[i])
        false_positive += len(predict[i] - ground_truth[i])
        false_negative += len(ground_truth[i] - predict[i])

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluation(gtruth_file, predict_file, product="Laptops"):
    ground_truth = []
    predict = []
    with open(gtruth_file) as f:
        for line in f:
            ground_truth.append(set(parse_sentinese(line)))
    with open(predict_file) as f:
        for line in f:
            predict.append(set(parse_sentinese(line)))

    print f1_score(ground_truth, predict)

    ground_truth_cate = []
    for truth in ground_truth:
        ground_truth_cate.append(set([(item[0], item[1]) for item in truth]))
    predict_cate = []
    for pred in predict:
        predict_cate.append(set([(item[0], item[1]) for item in pred]))
    print f1_score(ground_truth_cate, predict_cate)


def main():
    from sys import argv
    evaluation(gtruth_file=argv[1], predict_file=argv[2])


def test():
    import sys

    with open(sys.argv[1]) as infile:
        outfile = open(sys.argv[2], 'w')
        for line in infile:
            for opinion in parse_sentinese("Laptops", entity_set, attr_set, line):
                outfile.write(str(opinion) + '; ')
            outfile.write('\n')


#test()
if __name__ == "__main__":
    main()



import copy
import math
import collections import Counter

class NaiveBayesClassifier(object):

    def __init__(self):
        self.priors = Counter()
        self.distributions = None
        self.categorys = None
        self.cate_index = None


    def train(self, X, labels, dist_types):
        M = float(len(labels))
        N = len(dist_types)

        self.categorys = sorted(list(set(labels)))
        n_cates = len(self.categorys)
        cate_index = {}
        self.cate_index = cate_index
        for index, cate in enumerate(self.categorys):
            cate_index[cate] = index

        # Initialize distributions
        line_dists = []
        for feature in dist_types:
            dist = dist_types()
            line_dists.append(dist)

        for cate in self.categorys:
            self.distributions.append(copy.deepcopy(line_dists))
        
        # Estimate distributions and count labels
        label_count = Counter()
        for i, label in enumerate(labels):
            label_count[label] += 1
            index = cate_index[label]
            for j, dist in enumerate(self.distributions[index]):
                dist.add_point(X[i][j])

        for line in self.distributions:
            for dist in line:
                dist.param_estimate()

        for cate in label_count:
            self.priors[cate] = math.log(label_count[cate]/M)


    def probabilities(self, x):
        labelWeights = copy.deepcopy(self.priors)

        for #TODO



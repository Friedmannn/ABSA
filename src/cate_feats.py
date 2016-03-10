import use_alignment


def get_entity_attribute(sentences):
    feat_names = set()
    for sent in sentences:
        for opinion in sent.opinions:
            entity, attribute = opinion.category.split('#')
            feat_names.add(entity)
            feat_names.add(attribute)

    feat_names = list(feat_names)
    entattri_indexes = dict()
    for index, feat in enumerate(feat_names):
        entattri_indexes[feat] = index

    return feat_names, entattri_indexes


def extract_entattri(sentence, entattri_indexes):
    N = len(entattri_indexes)
    x = [0.0] * N
    for opinion in sentence.opinions:
        entity, attribute = opinion.category.split('#')
        try:
            x[entattri_indexes[entity]] = 1.
        except KeyError:
            print "INFO: Unseen entity {}".format(entity)
        try:
            x[entattri_indexes[attribute]] = 1.
        except KeyError:
            print "INFO: Unseen attribute {}".format(attribute)

    return x


def extract_estimated_entattri(sentence, feat_names, align_model, threshold):
    continuous_x = use_alignment.extract_feature(sentence, feat_names, align_model)
    x = [0 if i < threshold else 1 for i in continuous_x]
    return x


def extract_entattri_X(sentences):
    feat_names, entattri_indexes = get_entity_attribute(sentences)
    #X = []
    #for rv in reviews:
    #    for sent in rv.sentences:
    #        X.append(extract_entattri(sent, entattri_indexes))
    X = [extract_estimated_entattri(sent, entattri_indexes) for sent in sentences]
    return X


def extract_estimated_entattri_X(sentences, align_model, threshold):
    feat_names, entattri_indexes = get_entity_attribute(sentences)
    #X = []
    #for rv in reviews:
        #for sent in rv.sentences:
            #X.append(extract_estimated_entattri(sent, feat_names, align_model))
    X = [extract_estimated_entattri(sent, feat_names, align_model, threshold) for sent in sentences]
    return X


def concatenate(X1, X2):
    X = []
    for index, x1 in enumerate(X1):
        x2 = X2[index]
        X.append(x1 + x2)
    return X


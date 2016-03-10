
def microF1(output, ground_truth):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(len(output)):
        o = set(output[i])
        g = set(ground_truth[i])
        TP += len(o & g)
        FP += len(o - g)
        FN += len(g - o)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    if p + r == 0:
        f = 0
    else:
        f = 2. * p * r / (p + r)
    return p, r, f


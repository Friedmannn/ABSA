from gensim.models.doc2vec import TaggedLineDocument, Doc2Vec

print "Load data"
sentences = TaggedLineDocument("../../data/laptop.unlabeled.txt")

#model = Doc2Vec(alpha=0.025, min_alpha=0.025, workers=4)  
model = Doc2Vec(workers=4)  
model.build_vocab(sentences)

print "Start training"
model.train(sentences)
#for epoch in range(10):
#    model.train(sentences)
#    model.alpha -= 0.002  # decrease the learning rate
#    model.min_alpha = model.alpha  # fix the learning rate, no decay

print model[99]

model.save("../../models/laptop.doc2vec.model")

print "Done"

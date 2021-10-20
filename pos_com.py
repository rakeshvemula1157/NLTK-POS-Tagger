import nltk

from nltk.corpus import indian
data = indian.tagged_sents()
test_data = indian.sents()

data_m = []
test_data_m = []
with open('IIIT_data') as f:
    fh = f.readlines()
    lis = []
    lis1 = []
    for i in fh:
        if (i == '\n'):
            data_m.append(lis)
            test_data_m.append(lis1)
            lis = []
            lis1 = []
        else:
            temp = (i.strip()).split('\t')
            lis.append(tuple((temp[0],temp[1])))
            lis1.append(temp[0])
print(len(test_data_m))


tot_data = data_m
test_data = test_data_m
train_size = int(len(tot_data)/10)*9

import random
import numpy as np
training_data = []
testing_data = []
gold = []
data_batch = 5

samples = []
for i in range(data_batch):
    train_num = random.sample(range(len(tot_data)), train_size)
    train_data_temp = [tot_data[i] for i in train_num]
    test_num = [i for i in range(len(tot_data)) if i not in train_num]
    test_data_temp = [test_data[i] for i in test_num]
    gold_data_temp = [tot_data[i] for i in test_num]
    training_data.append(train_data_temp)
    testing_data.append(test_data_temp)
    gold.append(gold_data_temp)



comb_results = np.zeros((5,4))
ind_results = np.zeros((5,4))
for ki in range(data_batch):
    from nltk import TnT
    from nltk.tag import hmm
    from nltk.tag.perceptron import PerceptronTagger
    from nltk.tag import CRFTagger

    perc_tagger = PerceptronTagger(load=False)
    tnt_tagger = TnT()
    crf_tagger = CRFTagger()


    tnt_tagger.train(training_data[ki])
    hmm_tagger = nltk.HiddenMarkovModelTagger.train(training_data[ki])
    perc_tagger.train(training_data[ki])
    crf_tagger.train(training_data[ki],'model.crf.tagger')

    # t.tagdata(test_data[800:])

    perc_pred = []
    hmm_pred = []

    for i in testing_data[ki]:
        perc_pred.append(perc_tagger.tag(i))
        hmm_pred.append(hmm_tagger.tag(i))
    crf_pred = crf_tagger.tag_sents(testing_data[ki])
    tnt_pred = tnt_tagger.tagdata(testing_data[ki])
    pred = {'p':perc_pred,'h':hmm_pred,'c':crf_pred,'t':tnt_pred}
    def most_frequent(List): 
        return max(set(List), key = List.count) 
    import itertools
    def picker(tag_seq,i,j):
        tags = []
        for k in tag_seq:
            tags.append(pred[k][i][j][1])
        return tags,most_frequent(tags)

    s='phct'
    seq_comb = list(itertools.combinations(s,3))

    def evaluate_seq(pred_seq,gold):
        total = count = 0
        for i in range(len(gold)):
            for j in range(len(gold[i])):
                tg, v = picker(pred_seq,i,j)
                total+=1
                if(v == gold[i][j][1]):
                    count+=1
        return (count/total)
    ind=0
    for seq in seq_comb:
        comb_results[ki][ind] = evaluate_seq(seq,gold[ki])
        ind+=1

    def evaluate(pred,gold):
        total = count = 0
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                total+=1
                if(pred[i][j][1]==gold[i][j][1]):
                    count+=1
        return (count/total)

    ind_results[ki][0] = evaluate(crf_pred,gold[ki])
    ind_results[ki][1] = evaluate(tnt_pred,gold[ki])
    ind_results[ki][2] = evaluate(hmm_pred,gold[ki])
    ind_results[ki][3] = evaluate(perc_pred,gold[ki])

print("Results of Combinations: \n'perceptron', 'hmm', 'crf':",sum(comb_results[:,0])/data_batch, "\n'perceptron', 'hmm', 'tnt':",sum(comb_results[:,1])/data_batch,"\n'perceptron', 'crf', 'tnt':",sum(comb_results[:,2])/data_batch ,"\n'hmm', 'crf', 'tnt':",sum(comb_results[:,3])/data_batch)

print("Results of Individual Taggers: \n'crf':",sum(ind_results[:,0])/data_batch, "\n'tnt':",sum(ind_results[:,1])/data_batch,"\n'hmm':",sum(ind_results[:,2])/data_batch ,"\n'perceptron':",sum(ind_results[:,3])/data_batch)

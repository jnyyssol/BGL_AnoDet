import numpy as np
import os
import DL_models as dl
from NGram import NGram 
from pandas import DataFrame


def y_list_with_ano_pos(n = 5, pos = 0):
    full_y = np.load("bgl_y_e_seq_"+str(n)+".npy", allow_pickle=True)

    if n==1:
        full_y = 1-full_y
        return list(full_y)

    y = list()

    #dynamically checking whether the position we are predicting is an anomaly 
    for seq in full_y:
        if len(seq) > pos:
            if seq[pos] == '-':
                no_ano = 1
            else:
                no_ano = 0
        else:
            no_ano = 0
        y.append(no_ano)

    return y


def give_accuracy_measures(model,model_type, train_selection, ks = [1], ps = [0]):
    bgl_y = y_list_with_ano_pos(n = model._ngrams_, pos = model.predfromlast)
    result_list = list()

    for p in ps:
        for k in ks:
            ngram_preds2 = list()
            if model_type=="ngram":
                valuesl = list()
                for seq in model.normal_test:
                    preds, values = NGram.give_preds(model, seq)
                    ngram_preds2.append(round(np.mean(preds), 0)) #rounding required if many preds, but bgl only has 1
                    valuesl.append(values)
                print(DataFrame(valuesl).value_counts(normalize=True))
            else:   #for dl
                tenth = int(len(model.normal_test)/10)
                lstm_preds_all = list()
                for i in range(0,10):
                    pos = i*tenth
                    lstm_preds_t= dl.give_DL_preds(model,model.normal_test[pos:pos+tenth], top_k = k, top_p=p)[0]
                    lstm_preds_all.extend(lstm_preds_t) 
                ngram_preds2 = lstm_preds_all
            

            nancount = 0
            onecount = 0
            zerocount = 0
            checkcount = 0
            for i,v in enumerate(ngram_preds2):
                if v == 1:
                    onecount += 1
                if v == 0:
                    zerocount += 1
                if np.isnan(v):
                    nancount += 1
                    ngram_preds2[i] = 0
            #helpful prints:
            #print("one ", onecount, ", zero ", zerocount, ", nan ", nancount)
            #print(np.array(ngram_preds2))
            #print(np.mean(np.array(ngram_preds2)))

            pn_tn = 0 #predict normal, true normal
            pa_ta = 0 
            pa_tn = 0
            pn_ta = 0

            #the list of predictions ngram_preds2 has already train seqs taken out
            #so we need to do compare only matching ids for ano labels in bgl_y
            test_ids = [x for x in range(len(ngram_preds2)) if x not in train_selection]
            print(len(train_selection), " removed from train")
            print(len(ngram_preds2),len(bgl_y))

            for i in range(len(test_ids)):
                if ngram_preds2[i] == 1 and bgl_y[test_ids[i]] == 1:
                    pn_tn += 1
                if ngram_preds2[i] == 0 and bgl_y[test_ids[i]] == 0:
                    pa_ta += 1
                if ngram_preds2[i] == 0 and bgl_y[test_ids[i]] == 1:
                    pa_tn += 1
                if ngram_preds2[i] == 1 and bgl_y[test_ids[i]] == 0:
                    pn_ta += 1
                checkcount += 1
                if checkcount % 100000 == 0:
                    print(checkcount)
            if pa_ta+pa_tn > 0:
                prec = pa_ta/(pa_ta+pa_tn)
            else:
                prec = 0
            rec = pa_ta/(pa_ta+pn_ta)
            if prec+rec > 0:
                f1 = 2*(prec*rec/(prec+rec))
            else:
                f1 = 0
            with open('results_metrics.txt', 'a') as f:
                f.write("True negative, no ano correctly predicted: "+ str(pn_tn)+"\n")
                f.write("True positive, ano correctly predicted: "+ str(pa_ta)+"\n")
                f.write("False positive, predicted ano when there is none: "+ str(pa_tn)+"\n")
                f.write("False negative, predicted no ano when there is: "+ str(pn_ta)+"\n")
                f.write("Recall: "+ str(rec)+"\n")
                f.write("Precision: "+ str(prec)+"\n")
                f.write("F1 (old): "+str(pa_ta/(pa_ta+0.5*(pa_tn+pn_ta)))+"\n\n")
                f.write("F1 (new): "+str(f1)+"\n\n")
            
            #recall, precision, f1
            temp_list = [rec,prec,f1, k, p]
            result_list.append(temp_list)
            
    return result_list
                



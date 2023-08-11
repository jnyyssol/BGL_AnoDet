

from collections import Counter
from collections import defaultdict 
import numpy as np
import time
from pandas import DataFrame
#More clo



class NGram:

    _start_ ="SoS" #Start of Sequence used in padding the sequence
    _end_ = "EoS" #End of Sequence used in padding the sequence

    #Jesse: I want to store normal_test in the class to unify the structure with the DL_models class. It's used in the accuracy_metrics.py functions.
    #I added now an empty array as the default value, so you don't need to put it if you create the test set dynamically.
    def __init__(self, train_data, normal_test=[], n=5, predictword = "next", predfromlast = 0, padding=1,top_k=[1],top_p=[1],kp=0):
        self._ngrams_ = n
        self.n_gram_counter = Counter()
        self.n_gram_counter_1 = Counter()
        self.predictword = predictword
        self.predfromlast = predfromlast
        self.padding = padding
        self.n1_gram_dict = defaultdict() # to keep mappings of possible following events e1 e2 -> e1 e2 e3, e1 e2 e4, 
        self.n1_gram_winner = dict() #What is the event n following n-1 gram, i.e. the prediction ? 
        self.prepad =  [self._start_]*(self._ngrams_-1)
        self.postpad = [self._end_]
        self.top_k = top_k[0]
        self.top_p = top_p[0]
        self.normal_test = normal_test
        self.kp=kp
        if kp > 0:
            self.create_ngram_model_kp(train_data)
        else:
            self.create_ngram_model (train_data)


    def create_ngram_model(self, train_data):
        #global n_gram_counter, n_gram_counter_1

        start_s = time.time()
        ngrams = list()
        ngrams_1 = list()
        for seq in train_data:
            seqs, seqs_1 = self.slice_to_ngrams(seq)
            ngrams.extend(seqs)
            ngrams_1.extend(seqs_1)
        self.n_gram_counter += Counter (ngrams)
        self.n_gram_counter_1 += Counter (ngrams_1)

        end_s = time.time()
        print("Done slicigin", end_s - start_s)
        start_s = time.time()

        for idx, s in enumerate(ngrams):
            #dictionary for faster access from n-1 grams to n-grams, e.g. from  [e1 e2 e3] -> [e1 e2 e3 e4]; [e1 e2 e3] -> [e1 e2 e3 e5] etc...
            self.n1_gram_dict.setdefault(ngrams_1[idx],[]).append(s)
            #precompute the most likely sequence following n-1gram. Needed to keep prediction times fast
            if (ngrams_1[idx] in self.n1_gram_winner): #is there existing winner 
                n_gram = self.n1_gram_winner[ngrams_1[idx]]
                if (self.n_gram_counter[n_gram] < self.n_gram_counter[s]): #there is but we are bigger replace
                    self.n1_gram_winner[ngrams_1[idx]] = s
            else: 
                self.n1_gram_winner[ngrams_1[idx]] = s #no n-1-gram key or winner add a new one...
        end_s = time.time()
        print("Done modeling", end_s - start_s)

    
    def create_ngram_model_kp(self, train_data):
        #global n_gram_counter, n_gram_counter_1

        start_s = time.time()
        ngrams = list()
        ngrams_1 = list()
        for seq in train_data:
            seqs, seqs_1 = self.slice_to_ngrams(seq)
            ngrams.extend(seqs)
            ngrams_1.extend(seqs_1)
        self.n_gram_counter += Counter (ngrams)
        self.n_gram_counter_1 += Counter (ngrams_1)

        end_s = time.time()
        print("Done slicigin", end_s - start_s)
        start_s = time.time()

        testcount = 0
        for idx, s in enumerate(ngrams):
            #dictionary for faster access from n-1 grams to n-grams, e.g. from  [e1 e2 e3] -> [e1 e2 e3 e4]; [e1 e2 e3] -> [e1 e2 e3 e5] etc...
            self.n1_gram_dict.setdefault(ngrams_1[idx],[]).append(s)
            #precompute the most likely sequence following n-1gram. Needed to keep prediction times fast
            if ngrams_1[idx] in self.n1_gram_winner:
                candidates = self.n1_gram_winner[ngrams_1[idx]]
                if s not in candidates:
                    candidates.append(s)
            else:
                self.n1_gram_winner[ngrams_1[idx]] = [s]

        if self.kp==1:
            # keep only top k candidates
            for gram in self.n1_gram_winner:
                candidates = self.n1_gram_winner[gram]
                if len(candidates) > self.top_k:
                    sorted_candidates = sorted(candidates, key=lambda x: self.n_gram_counter[x], reverse=True)
                    self.n1_gram_winner[gram] = sorted_candidates[:self.top_k]
        elif self.kp==2:
            # keep only top p candidates based on cumulative probability
            for gram in self.n1_gram_winner:
                candidates = self.n1_gram_winner[gram]
                sorted_candidates = sorted(candidates, key=lambda x: self.n_gram_counter[x], reverse=True)
                cum_probs = np.cumsum([self.n_gram_counter[x] for x in sorted_candidates]) / np.sum([self.n_gram_counter[x] for x in sorted_candidates])

                top_p_candidates = [sorted_candidates[i] for i in range(len(sorted_candidates)) if cum_probs[i] <= self.top_p]
                self.n1_gram_winner[gram] = top_p_candidates
                
        #print(cum_probs[0], top_p_candidates)

        end_s = time.time()
        print("Done modeling", end_s - start_s)

    #Produce required n-grams. E.g. With sequence [e1 ... e5] and _ngrams_=3 we produce [e1 e2 e3], [e2 e3 e4], and [e3 e4 5] 
    def slice_to_ngrams (self, seq):
        #Add SoS and EoS
        #with n-gram 3 it is SoS SoS E1 E2 E3 EoS
        #No need to pad more than one EoS as the final event to be predicted is EoS
        if (self.padding):
            if self.predictword=="next":
                seq = [*self.prepad, *seq, *self.postpad] # this is slightly faster than the below ones. 
                #seq = self.prepad +seq+self.postpad
                #seq = [self._start_]*(self._ngrams_-1) +seq+[self._end_]
            #Calculate appropriate amount of padding based on the predict position
            elif self.predictword=="masked":
                seq = [self._start_]*(self._ngrams_-1-self.predfromlast) +seq+[self._end_]*(1+self.predfromlast)

        ngrams = list()
        ngrams_1 = list()
        for i in range(self._ngrams_, len(seq)+1):#len +1 because [0:i] leaves out the last element 
            ngram_s = seq[i-self._ngrams_:i]
            # convert into a line
            line = ' '.join(ngram_s)
            # store
            ngrams.append(line)
            ngram_s_1= seq[i-self._ngrams_:i-1] #if i=13, takes from indexes 8 to 11
            #masked word test
            if(self.predictword == "masked"):
                temp = seq[i-self._ngrams_:i-(self.predfromlast+1)]
                temp.extend(seq[i-self._ngrams_+(self._ngrams_-self.predfromlast):i])
                ngram_s_1 = temp
            line2 = ' '.join(ngram_s_1)
            # store
            ngrams_1.append(line2)
        return ngrams, ngrams_1


    # Return two anomaly scores as in the paper
    # Ano score per line (i.e. given the previous lines how probable is this line). 
    # And n of occurences per line seen in the past
    def give_ano_score (self, seq):
        seq_shingle, seq_shingle_1 = self.slice_to_ngrams(seq)
        scores = list()
        for s in seq_shingle:
            scores.append(self.n_gram_counter [s])
        scores_1 = list()
        for s in seq_shingle_1:
            scores_1.append(self.n_gram_counter_1 [s])

        #Remove 0s from n1 gram list to get rid of division by zero. 
        # If n-1 gram is zero following n-gram must be zero as well so it does not effect the results
        scores_1 = [1 if i ==0 else i for i in scores_1]
        #Convert n-gram freq counts to probs of n-gram given n-gram-minus-1
        scores_prop = np.divide(np.array(scores), np.array(scores_1))
        scores_abs = np.array(scores)
        return (scores_prop, scores_abs)



    def give_preds_l (self,seq):
        seq_shingle, seq_shingle_1 = self.slice_to_ngrams(seq)
        #   print(seq_shingle)
        correct_preds = list()
        pred_values = list()
        for s in seq_shingle:
            to_be_matched_s =  s.rpartition(' ')[0]
            #print("to be matched " + to_be_matched_s)
            if (not (self.match(self, to_be_matched_s, s,correct_preds, pred_values))):
                correct_preds.append(0)
                pred_values.append(np.NaN) #we have no prediction add Nan
                #print("no key")
        return correct_preds, pred_values

    def match(self, to_be_matched_s, s, correct_preds, pred_values):
        #print ("Called ngram with n ",self._ngrams_)
        if (to_be_matched_s in self.n1_gram_dict):
            winner = self.n1_gram_winner[to_be_matched_s]
            if (winner == s):
                correct_preds.append(1) #Correct
            else: 
                correct_preds.append(0) #incorrect
                #print("incorrec predic")
                #value of predicted event, i.e. "A0010"
            if(self.predictword == "masked" ):
                pred_values.append(splitted[self._ngrams_-1-self.predfromlast]) #add what was predicted
            if(self.predictword == "next" ):
                pred_values.append(s.rpartition(' ')[2])
                #print("correct")
            return True
        else:
            if hasattr(self, 'next_NGram'):
                #print ("calling NGram",self.next_NGram._ngrams_ )
                s_short = s.strip().split(maxsplit=self._ngrams_)
                s_short = " ".join(s_short[self._ngrams_ - self.next_NGram._ngrams_ : ])
                to_be_matched_s_short =  s_short.rpartition(' ')[0]
                return self.next_NGram.match(to_be_matched_s_short, s_short, correct_preds, pred_values)
            else: 
                False    

    def give_preds (self,seq):
        seq_shingle, seq_shingle_1 = self.slice_to_ngrams(seq)
        #   print(seq_shingle)
        correct_preds = list()
        pred_values = list()
        for s in seq_shingle:
            to_be_matched_s =  s.rpartition(' ')[0]
            if (self.predictword == "masked" ):
                splitted = s.split(' ')
                asplit = splitted[:(self._ngrams_-self.predfromlast-1)]
                asplit.extend(splitted[(self._ngrams_-self.predfromlast):]) 
                to_be_matched_s = " ".join(asplit)
            #print("to be matched " + to_be_matched_s)
            if (to_be_matched_s in self.n1_gram_dict):
                winner = self.n1_gram_winner[to_be_matched_s]
                if not isinstance(winner, (list)): #Turn single value to a list, so we can use same code regardless of number of winners
                    winner = [winner]
                if (s in winner):
                    correct_preds.append(1) #Correct
                else: 
                    correct_preds.append(0) #incorrect
                    #print("incorrec predic")
                    #value of predicted event, i.e. "A0010"
                if(self.predictword == "masked" ):
                    pred_values.append(splitted[self._ngrams_-1-self.predfromlast]) #add what was predicted
                if(self.predictword == "next" ):
                    pred_values.append(s.rpartition(' ')[2])
                    #print("correct")
            else:
                correct_preds.append(0)
                pred_values.append(np.NaN) #we have no prediction add Nan
                #print("no key")
        return correct_preds, pred_values


    def ngram_prediction(normal_test, ngram):
    #ngram prediction-------------------------------------------
        #ngram test with loop
        ngram_preds = list()
        ngram_preds2 = list()
        ngram_predvalues = list()
        start_s = time.time()
        for normal_s in normal_test:
            preds, values = ngram.give_preds(normal_s)
            ngram_preds.append(preds)
            ngram_preds2.extend(preds)
            ngram_predvalues.extend(values)
            #print(".")
        end_s = time.time()
        #print("prediction time ngram with ngrams:", _ngrams_, "done in", end_s - start_s)
        #sheet_form_print("N-Gram", "I-Time", end_s - start_s)
        #ngram investigate
        ngram_preds_means = list()
        for preds in ngram_preds:
            ngram_mean = np.mean(preds)
            ngram_preds_means.append(ngram_mean)
            #print (np.mean(lstm_mean))

        valuedf = DataFrame(ngram_predvalues)
        #sheet_form_print("N-Gram", "Accuracy (Mom)", np.mean(ngram_preds_means))
        #sheet_form_print("N-Gram", "Accuracy (Moa)", np.mean(ngram_preds2))
        #print("Most frequent event: "+ valuedf.value_counts(normalize=True).index[0][0] +", "+ str(round(valuedf.value_counts(normalize=True)[0], 3)*100) + "%")
        print("Correct Mean of means", np.mean(ngram_preds_means))
        print("Correct Mean of all", np.mean(ngram_preds2))
        missing = valuedf.isna().sum().sum() 
        total = len(ngram_predvalues)
        print("Incorrect due to missing",missing ,"percentage ", missing/total )

        return np.mean(ngram_preds2)
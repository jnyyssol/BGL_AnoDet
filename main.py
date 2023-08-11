from attr import dataclass
import numpy as np
import os
import load_functions as lf
import DL_models as dl
from NGram import NGram 
import accuracy_metrics as am
import time

#Adjust accordingly
os.chdir("/home/ubuntu/Development/jesse/BGL_AnoDet/data")

#Prints sheet format. Example (tabs adjusted for readability):
#Data	Method	Epoch   predict word  Window (n)	HW	    Split	Source	        Measure	    Measurement
#HDFS	CNN	    10      masked        5		        GPU-PC  50-50	M-23-02-2022	Accuracy    0.847
def sheet_form_print(model,model_type,data,split, measure, measurement): 
    if not os.path.isdir("results"):
        os.makedirs("results")
    source = "TEST" #Identify who did the test and also appears in the filename
    hw = "GPU-PC" #GPU-PC is CSC
    #make data source text uniform
    datastr = data.capitalize()
    if(data == "hdfs" or data == "bgl"):
        datastr = data.upper()
    #if masked, add position of predictionS
    predstr = model.predictword
    if(predstr=="masked"):
        predstr = "masked_"+str(model.predfromlast)
    if model_type == "ngram":
        epoch = ""
    else:
        epoch = str(model.model_epoch)
    sheetstr = datastr+"\t"+model_type+"\t"+epoch+"\t"+predstr+"\t"+str(model._ngrams_)+"\t"+hw+"\t"+str(split)+"-"+str(100-int(split))+"\t"+source+"\t"+measure+"\t"+str(round(measurement,3))
    print(sheetstr)
    #save results to file
    with open('results/results_'+source+'.txt', 'a') as f:
        f.write(sheetstr+"\n")

#Configurations
class Configurations:
    def __init__(self, models, datasets, seqlen, splits, maskrange,epoch=10):
        self.datasets = datasets
        self.seqlen = seqlen
        self.splits = splits
        self.maskrange = maskrange
        self.models = models
        self.epoch = epoch

def multiruns(conf, bgl_accmetrics=False, top_k=[1], top_p=[0]):
    model_list = list()
    for data in conf.datasets:  
        for _ngrams_ in conf.seqlen:
            if data == "bgl":
                padding = 0
                #lf.create_splits(["bgl"], _ngrams_) 
                #bgl needs new splits when the n changes because several files based on seq length
                #however, commented out because we use the new function for accuracy metrics
            else:
                padding = 1
            for model_type in conf.models:
                for predfromlast in conf.maskrange: 
                    if predfromlast < _ngrams_:
                        for split in conf.splits:
                            predword = "masked"
                            if predfromlast == 0:
                                predword = "next"
                            if bgl_accmetrics:
                                s = lf.load_and_split_BGL(_ngrams_,split)
                            else:
                                s = lf.split_normal(split, data, _ngrams_)                                
                            if model_type == "ngram":
                                start_s = time.time()
                                if top_k[0]==0:
                                    kp = 2
                                if top_p[0]==0:
                                    kp = 1
                                m = NGram(s[0],s[1], _ngrams_,predword,predfromlast,padding,top_k,top_p,kp)
                                ttime = time.time() - start_s
                                pred = NGram.ngram_prediction(s[1], m)
                                
                            else:
                                start_s = time.time()
                                m = dl.DL_models(s[0], s[1], model_type,_ngrams_, predword,predfromlast,conf.epoch,padding)
                                ttime = time.time() - start_s
                                #pred = dl.DL_preds_mean(m, data) 
                            if bgl_accmetrics:
                                result_list = am.give_accuracy_measures(m, model_type, s[2], top_k,top_p) #for whole dataset
                                for line in result_list:
                                    sheet_form_print(m, model_type, data, split, "Precision_k="+str(line[3])+"_p="+str(line[4]), line[1])
                                    sheet_form_print(m, model_type, data, split, "Recall_k="+str(line[3])+"_p="+str(line[4]), line[0])
                                    sheet_form_print(m, model_type, data, split, "F1_k="+str(line[3])+"_p="+str(line[4]), line[2])
                            #sheet_form_print(m, model_type, data, split, "Accuracy", pred) #for normal test
                            sheet_form_print(m, model_type, data, split, "T-Time", ttime)
                            model_list.append(m)

### Note! Using multiple k and p values doesn't work yet for ngram, because it was implemented 
###       for prediction in DL models, but for ngram they need to be applied in model creation
###       Top k/p is determined by whichever is not [0] in the multiruns parameter.

#Conf parameters: models, datasets, window size, split, mask position, epoch
#Example: ["cnn", "ngram"], ["bgl","pro","hadoop","hdfs"], [10], [50], [1], 10

conftest = Configurations(["ngram"], ["bgl"], [2,5,10,15,20,50], [50], [0],5)
multiruns(conftest, True, [400], [0]) 
#In multiruns after conf: first if bgl_accmetrics used, second for top_k, third for top_p


#--- PRESETS --- 

#Different k-values with CNN model (window 10, mask pos 1)
conf1 = Configurations(["cnn"], ["bgl"], [10], [25], [1],25)
multiruns(conf1, True, [1,3,5,10,25,50,100,200,300,400,500,600,700,800], [0])

#Same as above except for p-values
plist=list()
for x in range(0,10):
    y=1-1/(pow(2,x))
    plist.append(y)
conf2 = Configurations(["cnn"], ["bgl"], [10], [25], [1],25)
multiruns(conf2, True, [0], plist)

#Various window sizes and mask positions
conf3 = Configurations(["cnn"], ["bgl"], [2,5,10,15,20,50], [25], [0,1],25)
multiruns(conf3, True, [200], [0])


#Loop the same test for more results
for i in range(50):
    conf4 = Configurations(["transformer"], ["bgl"], [5], [25], [1],1)
    multiruns(conf4, True, [200], [0])


#Other models
conf5 = Configurations(["transformer", "lstm"], ["bgl"], [5], [50], [1],50)
multiruns(conf4, True, [200], [0])

#If necessary for some reason create splits! You can pass a parameter but by default these are created: [10,25,50,75,90]
#lf.create_splits()


#single run example DL
#data = "hadoop"
#split = 50
#s = lf.split_normal(split, data) #create a split for normal_train s[0] and normal_test s[1]

#model_type = "transformer"
#m = dl.DL_models(s[0], s[1], model_type,10, "next",0, model_epoch=3)
#pred = dl.DL_preds_mean(m, data) #check splitting the predictions for memory
#sheet_form_print(m, model_type, data, split, "Accuracy", pred)
#To get arrays of predictions and correct values, run dl.give_DL_preds(m)

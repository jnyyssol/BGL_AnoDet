
import numpy as np
import os


def load_bgl_data(len):
    bgl_x = np.load("bgl_x_seq_"+str(len)+".npy", allow_pickle=True)
    bgl_y = np.load("bgl_y_seq_"+str(len)+".npy", allow_pickle=True)

    bgl_y = bgl_y == 1

    bgl_x_normal = bgl_x[~bgl_y]
    abnormal_test = bgl_x[bgl_y]
    normal_data = bgl_x_normal

    return normal_data, abnormal_test

def split_normal(split, data, n):    
    normal_train = np.loadtxt('splits/split_'+str(split)+'_'+data+'_train.txt') #load split
    if data == "bgl":
        funcstr = "load_bgl_data("+str(n)+")" 
        normal_data,abnormal_test = eval(funcstr)
    else:
        funcstr = "load_" + data + "_data()" 
        normal_data,abnormal_test = eval(funcstr)
    normal_train = np.array(normal_train, dtype=bool)
    normal_test = normal_data[~normal_train]
    normal_train = normal_data[normal_train]
    return normal_train, normal_test

def create_splits(datasets = ["pro","hdfs", "bgl", "hadoop"], n=5):
    portions = [10,25,50,75,90]
    if not os.path.isdir("splits"):
        os.makedirs("splits")

    for dataset in datasets:
        if dataset == "bgl":
            funcstr = "load_bgl_data("+str(n)+")" 
            normal_data,abnormal_test = eval(funcstr)
        else:
            funcstr = "load_" + dataset + "_data()" 
            normal_data,abnormal_test = eval(funcstr)
        
        for portion in portions:
            train_i = np.random.choice(normal_data.shape[0], int(normal_data.shape[0]*(portion/100)), replace=False)
            normal_train = np.isin(range(normal_data.shape[0]), train_i)
            namestr = "splits/split_"+str(portion)+"_"+dataset+"_train.txt"
            np.savetxt(namestr, normal_train, fmt='%d')

def load_and_split_BGL(n, portion):
    #Specifically this splits the data into a training set without anomalies and assigns everything else in the test_set.
    #This requires its own function, because we can't retroactively match the ano labels after doing random splitting on normal data

    bgl_x = np.load("bgl_x_seq_"+str(n)+".npy", allow_pickle=True)
    bgl_y = np.load("bgl_y_seq_"+str(n)+".npy", allow_pickle=True)

    normal_ids = list()
    for i,e in enumerate(bgl_y):
        if e == 0:
            normal_ids.append(i)


    train_selection = np.random.choice(normal_ids, int(len(normal_ids)*(portion/100)), replace=False)
    normal_train = np.isin(range(len(bgl_y)), train_selection)

    test_data = bgl_x[~normal_train]
    normal_train = bgl_x[normal_train]

    return normal_train, test_data, train_selection
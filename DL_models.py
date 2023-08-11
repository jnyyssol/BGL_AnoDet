from collections import Counter
from collections import defaultdict 
import numpy as np
import time
import random
import pickle
import math
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from keras.layers import Dropout
from tensorflow import keras
import load_functions as lf
from NGram import NGram 
import os

class DL_models:

    _start_ ="SoS" #Start of Sequence used in padding the sequence
    _end_ = "EoS" #End of Sequence used in padding the sequence

    def __init__(self, normal_train, normal_test, model_type, n=5, predictword = "next", predfromlast = 0, model_epoch = 10, padding = 1):
        self._ngrams_ = n
        self.n_gram_counter = Counter()
        self.n_gram_counter_1 = Counter()
        self.predictword = predictword
        self.predfromlast = predfromlast
        self.model_epoch = model_epoch
        self.normal_test = normal_test
        self.model_type = model_type
        self.padding = padding
        self.n1_gram_dict = defaultdict() # to keep mappings of possible following events e1 e2 -> e1 e2 e3, e1 e2 e4, 
        self.n1_gram_winner = dict() #What is the event n following n-1 gram, i.e. the prediction ? 
        self.prepad =  [self._start_]*(self._ngrams_-1)
        self.postpad = [self._end_]
        lstm_ngrams, lstm_ngrams_num, lstm_vocab_size, self.lstm_tokenizer = sequences_to_dl_ngrams(self, normal_train)
        self.model = create_DL_model(self, lstm_ngrams_num, lstm_vocab_size, model_epoch=model_epoch)
        #self.model = create_transformer_model(lstm_ngrams_num, lstm_vocab_size)


# We need to change events e1 e2 e3 to numbers for the DL model so they are mapped here, e.g. e1 -> 137, e2 -> 342 
def sequences_to_dl_ngrams (self, normal_train):
    ngrams = list() #ngrams= []
    for seq in normal_train:
        t_ngrams, t_ngrams_1 = NGram.slice_to_ngrams(self, seq)
        ngrams.extend(t_ngrams)
    tokenizer = Tokenizer(oov_token=1)    
    tokenizer.fit_on_texts(ngrams)
    ngrams_num = tokenizer.texts_to_sequences(ngrams)
    vocab_size = len(tokenizer.word_index) + 1
    return ngrams, ngrams_num, vocab_size, tokenizer


def create_DL_model(self, ngrams, vocab_size, share_of_data=1, model_epoch=10):
    #If we want to use less than 100% of data select samples. I am not sure this is ever used
    if (share_of_data < 1):
        select = int(len(ngrams) * share_of_data)
        ngrams = random.sample(ngrams, select)
    random.shuffle(ngrams)
    # How many dimensions will be used to represent each event. 
    # With words one would use higher values here, e.g. 200-400
    # Higher values did not improve accuracy but did reduce perfomance. Even 50 might be too much
    dimensions_to_represent_event = 50

    #For early stopping, not used at the moment
    callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, min_delta=0.001)
    
    opt = keras.optimizers.Adam() #learning_rate=0.01
    if (self.model_type != "transformer"):
        model = Sequential()
        model.add(Embedding(vocab_size, dimensions_to_represent_event, input_length=self._ngrams_-1))
        #model.add(Dropout(0.2, input_shape=(100,)))
        
    # We will use a two LSTM hidden layers with 100 memory cells each. 
    # More memory cells and a deeper network may achieve better results.
    print("creating " + self.model_type + " model")
    if(self.model_type == "lstm"):
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(vocab_size, activation='softmax'))
    if(self.model_type == "cnn"):
        model.add(Conv1D(filters=20 , kernel_size = self._ngrams_-1, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(vocab_size, activation='softmax'))
    if(self.model_type == "transformer"):
        model = build_transformer_model(
            input_shape=(self._ngrams_-1,1),
            head_size=64,
            num_heads=12,
            ff_dim=2048,
            num_transformer_blocks=1,
            mlp_units=[2],
            vocab_size=vocab_size,
            mlp_dropout=0.0,
            dropout=0.0
        )
    
    #model.add(Flatten())
    #model.add(Dropout(0.1)) #options to consider

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #print(model.summary())
    start_s = time.time()

    #Loop might be needed as Office PC would crash in the to_categorixal with Profilence data set as it out of memory. 
    #in CSC we don't need to loop so very high loop_variable
    loop_variable = 50000000
    for x in range(0, len(ngrams), loop_variable):
        print(f'loop with x= {x}. / {len(ngrams)}')
        ngrams0 = np.array(ngrams[x:x+loop_variable])
        if(self.predictword == "next"):
            X, y = ngrams0[:,:-1], ngrams0[:,-1]
        #masked word test
        if(self.predictword == "masked"):
            X = ngrams0[:,:(-1-self.predfromlast)]
            if(self.predfromlast > 0):
                X = np.append(X, ngrams0[:,-self.predfromlast:],axis=1)
            y = ngrams0[:,(-1-self.predfromlast)]

        y = to_categorical(y, num_classes=vocab_size)
        batch_size = 1024
        if(self.model_type == "transformer"):
            X = X.reshape(X.shape[0], X.shape[1],1)
            batch_size = 512
        #Modify batch_size and epoch to influence the training time and resulting accuracy. 
        history = model.fit(X, y, validation_split=0.05, batch_size=batch_size, epochs=model_epoch, shuffle=True).history #callbacks=[callback],
    
    print("training took", time.time() - start_s)
    return model


#LSTM predictions with multiple sequences packed in numpy array 
def give_DL_preds(self,seq,top_k=1,top_p=0, b_size=4096):
    if (self.model_type == "transformer"):
        b_size = 512
    seq_shingle = list()
    #check if this is an array of sequences
    start_s = time.time()
    if (isinstance(seq, np.ndarray)):
        for s in seq:
            temp_seq_shingle, temp_seq_shingle_1 = NGram.slice_to_ngrams(self, s)
            seq_shingle.extend(temp_seq_shingle)
    else: #if not numpy array then as 
        seq_shingle, seq_shingle_1 = NGram.slice_to_ngrams(self, s)
    end_s = time.time()
    print("Shingle creation took", end_s - start_s)
    start_s = time.time()
    seq_shingle_num = self.lstm_tokenizer.texts_to_sequences(seq_shingle) #do this before slice to n-grams
    end_s = time.time()
    print("lstm_tokenizer took", end_s - start_s)
    seq_shingle_num_np = np.array(seq_shingle_num)

    if(self.predictword == "next"):
        seq_shingle_num_1 = seq_shingle_num_np[:,:-1]
        seq_shingle_truth = seq_shingle_num_np[:,-1]
    if (self.predictword == "masked"):
        seq_shingle_num_1 = seq_shingle_num_np[:,:(-1-self.predfromlast)]
        if (self.predfromlast > 0):
            seq_shingle_num_1 = np.append(seq_shingle_num_1, seq_shingle_num_np[:,-self.predfromlast:], axis=1)
        seq_shingle_truth = seq_shingle_num_np[:,(-1-self.predfromlast)]

    start_s = time.time()
    predicted_sec = self.model.predict(seq_shingle_num_1,verbose=0, batch_size=b_size)
    end_s = time.time()
    print("prediction took", end_s - start_s)
    predicted_events_sorted = np.argsort(-1*predicted_sec, axis=1)
    correct_preds = list()
    
    for i in range(len(predicted_events_sorted)):
        iter_e = 0
        cumu_p = 0
        correct = False
        while cumu_p <= top_p or iter_e < top_k:
            cumu_p += predicted_sec[i][predicted_events_sorted[i][iter_e]]
            if seq_shingle_truth[i] == predicted_events_sorted[i][iter_e]:
                correct = True
                break
            iter_e += 1
        #Helpful for debugging:
        #if correct == False:
        #    print("Failed to predict " + str(seq_shingle_truth[i]))
        #    print("Checked " + str(iter_e) + " events with cumu p " + str(cumu_p))
        correct_preds.append(correct)

    print(np.mean(np.array(correct_preds)))

    predicted_events = list()
    for i in range(len(correct_preds)):
        predicted_events.append(predicted_events_sorted[i])
    #correct_preds = seq_shingle_truth == predicted_events
    return np.array(correct_preds), np.array(predicted_events)

def DL_preds_mean(self,data, b_size=4096):
    lstm_preds_all = list()
    if (data!="pro" and data!="bgl"):
        lstm_preds_all = give_DL_preds(self,self.normal_test, b_size=b_size)[0]
    else :#Cannot do all pro data in one pass runs out of memory at 15gigs. Split to five calls
        tenth = int(len(self.normal_test)/10)
        for i in range(0,10):
            pos = i*tenth
            lstm_preds_t = give_DL_preds(self,self.normal_test[pos:pos+tenth],b_size=b_size)[0]
            lstm_preds_all.extend(lstm_preds_t)  
    return np.mean(lstm_preds_all)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    vocab_size,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:08:35 2019

@author: Karan Vijay Singh
"""
#import sys
import ast
import csv
import gensim
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
#from keras.regularizers import l2
#import matplotlib.pyplot as plt

MAX_SEQUENCE_LENGTH = 19
EMBEDDING_DIM = 300
BATCH_SIZE = 512
EPOCHS = 50

def convert_textreview_to_list(pos_file, neg_file): 
        """reads the review files and then converts each review from string to list and then 
        join to form a sentence and combines positive and negative reviews together, 
        add labels and returns dataframe

        Parameters:
            pos_file (string): path to the postive review file
            neg_file (string): path to the negative review file
            colname (string): colname can be train, val, test

        Returns:
            lists:Returning value

       """

        pos_review_list = []
        pos_label = []
        with open(pos_file,"r") as f:
            reader = csv.reader(f,delimiter="\n")
            for line in reader:
                pos_review_list.append(' '.join(ast.literal_eval(line[0])))
                
        pos_label = [1]*len(pos_review_list)    
        
        neg_review_list = []
        neg_label = []
        with open(neg_file,"r") as f:
            reader = csv.reader(f,delimiter="\n")
            for line in reader:
                neg_review_list.append(' '.join(ast.literal_eval(line[0])))
        neg_label = [0]*len(neg_review_list) 
        
        return pos_review_list,pos_label,neg_review_list,neg_label
      
def get_embeddings(model,word_index):
    """reads the word2vec trained model and then creates an embedding index (dict) from the vocab of word2vec
    model i.e. key = word and value = embedding and then create an embedding matrix to be fed to embedding 
    layer of the neural network. if embedding for the corresponding word is not available then use 0 embedding 
    otherwise put the embedding corresponding to the word
        Parameters:
            model (model): trained word2vec model
            word_index (dict): dict that has key as word and correspoding index as number

        Returns:
            matrix: embedding matrix

       """
    model = gensim.models.Word2Vec.load(model)
    vocab = model.wv.vocab
    embeddings_index = {}
    for word in vocab.keys():
        vector = model.wv[word]
        embeddings_index[word] = np.asarray(vector, dtype='float32')
        
    print('Found %s word vectors.' % len(embeddings_index))   
    
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model( embedding_matrix, X_train, y_train, X_val,  y_val,
                activation_func = None, l2_reg = None, drop_val = None):
    print('Build model...')
    model = Sequential()
    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, # eg [34,45,54,23,...]
                            trainable=False, name='word_embedding_layer')
    model.add(embedding_layer)
    model.add(Flatten())
    if l2_reg is not None:
        model.add(Dense(512, kernel_regularizer=regularizers.l2(l2_reg), name='dense_layer')) #dropout=0.2,activation='relu'
    else:
        model.add(Dense(512, name='dense_layer'))
    if drop_val is not None:
        model.add(Dropout(rate=drop_val, name='dropout_1')) # Can try varying dropout rates
    if activation_func is not None:
        model.add(Activation(activation=activation_func, name='activation_1'))
    
        model.add(Dense(2, activation='softmax', name='output_layer'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='auto', patience = 2, verbose=1)
    #print(model.summary())
    
    print('Train...')
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=(X_val, y_val), verbose=2, callbacks=[es])
    
    return history, model
    

def print_accuracy(dict_p, technique):
    for k,v in dict_p.items():
        print("The accuracy for {} {} is {}.".format( k,technique,v))

if __name__ == "__main__":
    input_file_training_pos = "train_pos.csv" 
    input_file_training_neg = "train_neg.csv" 
    input_file_validation_pos = "val_pos.csv"
    input_file_validation_neg = "val_neg.csv"
    input_file_test_pos = "test_pos.csv"
    input_file_test_neg = "test_neg.csv"
    word2vec_model = "new_model1.model"
    
    train_pos,train_pos_label, train_neg, train_neg_label = convert_textreview_to_list(input_file_training_pos,input_file_training_neg)
    val_pos,val_pos_label, val_neg, val_neg_label = convert_textreview_to_list(input_file_validation_pos,input_file_validation_neg)
    test_pos,test_pos_label, test_neg, test_neg_label = convert_textreview_to_list(input_file_test_pos,input_file_test_neg)
    
    texts = train_pos + train_neg + val_pos + val_neg + test_pos + test_neg
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    
    train_reviews = train_pos + train_neg
    train_sequences = tokenizer.texts_to_sequences(train_reviews)
    y_train = train_pos_label + train_neg_label
    y_train = to_categorical(np.asarray(y_train))
    

    #train_neg_sequences = tokenizer.texts_to_sequences(train_neg)
    val_reviews = val_pos + val_neg
    val_sequences = tokenizer.texts_to_sequences(val_reviews)
    val_labels = val_pos_label + val_neg_label
    y_val = to_categorical(np.asarray(val_labels))
    
    #val_neg_sequences = tokenizer.texts_to_sequences(val_neg)
    test_reviews = test_pos + test_neg
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    test_labels = test_pos_label + test_neg_label
    y_test = to_categorical(np.asarray(test_labels))
    #test_neg_sequences = tokenizer.texts_to_sequences(test_neg)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    X_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_val = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    embedding_matrix = get_embeddings(word2vec_model,word_index) 
    
    print(embedding_matrix.shape)
    act_funs = ['relu','tanh','sigmoid']
    val_act_dict = dict()
    for activation_func in act_funs:
        history, model = create_model(embedding_matrix, X_train, y_train, X_val, y_val, activation_func)
        score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=2)
        val_act_dict[activation_func] = history.history['val_acc'][len(history.history['val_acc']) - 2 -1]
        
    activation_func = max(val_act_dict, key=val_act_dict.get)
    
    l2_reg_values = [0.01,0.001,0.0001]
    l2_reg_dict = dict()
    for l2_reg in l2_reg_values:    
        history, model = create_model(embedding_matrix, X_train, y_train, X_val, y_val, activation_func, l2_reg)
        score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=2)
        l2_reg_dict[l2_reg] = history.history['val_acc'][len(history.history['val_acc']) - 2 -1]
        
    l2_reg = max(l2_reg_dict, key=l2_reg_dict.get)
    
    drop_val_list = [0.2,0.3,0.4]
    drop_acc_dict = dict()
    for drop_val in drop_val_list:
        history, model = create_model(embedding_matrix, X_train, y_train, X_val, y_val, activation_func, l2_reg, drop_val)
        score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=2)
        drop_acc_dict[drop_val] = history.history['val_acc'][len(history.history['val_acc']) - 2 -1]
        
    drop_val = max(drop_acc_dict, key=drop_acc_dict.get)
    
    print_accuracy(val_act_dict, "Activation") 
    print_accuracy(l2_reg_dict, "L2 Regularisation")
    print_accuracy(drop_acc_dict, "Dropout")
    #using best model to get test set accuracy
    history, model = create_model(embedding_matrix, X_train, y_train, X_val, y_val, activation_func, l2_reg, drop_val)
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)
    
    print('Test accuracy for the best tuned model with activation:{}, l2 regularisation:{}, dropout rate:{} is {} :'.format(activation_func,l2_reg,drop_val, acc))
    
    
    
    
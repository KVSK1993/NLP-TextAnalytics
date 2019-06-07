# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 23:44:17 2019

@author: Karan Vijay Singh
"""

import sys
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv

def convert_textreview_to_dataframe(pos_file, neg_file, colname): 
        """reads the review files and then converts each review from string to list and then 
        join to form a sentence and combines positive and negative reviews together, 
        add labels and returns dataframe

        Parameters:
            pos_file (string): path to the postive review file
            neg_file (string): path to the negative review file
            colname (string): colname can be train, val, test

        Returns:
            dataframe:Returning value

       """

        pos_review_list = []
        with open(pos_file,"r") as f:
            reader = csv.reader(f,delimiter="\n")
            for line in reader:
                pos_review_list.append(line[0])
            
        neg_review_list = []
        with open(neg_file,"r") as f:
            reader = csv.reader(f,delimiter="\n")
            for line in reader:
                neg_review_list.append(line[0])
        
        df_pos = pd.DataFrame(data ={colname:pos_review_list})
        df_neg = pd.DataFrame(data ={colname:neg_review_list})
        
        df_pos[colname+"String"] = df_pos[colname].apply(lambda x: ' '.join(review for review in ast.literal_eval(x)))
        df_neg[colname+"String"] = df_neg[colname].apply(lambda x: ' '.join(review for review in ast.literal_eval(x)))
        df_pos['label'] =  1
        df_neg['label'] =  0
        df = df_pos.append(df_neg)
        return df[[colname+"String",'label']].sample(frac=1).reset_index(drop=True)
    
if __name__ == "__main__":
    input_file_training_pos = sys.argv[1] 
    input_file_training_neg = sys.argv[2] 
    input_file_validation_pos = sys.argv[3]
    input_file_validation_neg = sys.argv[4]
    input_file_test_pos = sys.argv[5]
    input_file_test_neg = sys.argv[6]
    
    df_train = convert_textreview_to_dataframe(input_file_training_pos,input_file_training_neg,"train")
    df_val = convert_textreview_to_dataframe(input_file_validation_pos,input_file_validation_neg,"val")
    df_test = convert_textreview_to_dataframe(input_file_test_pos,input_file_test_neg,"test")
   
    y_val = df_val['label']
    y_test = df_test['label']
    gram_dict = {(1,1):"unigram",(2,2):"bigram",(1,2):"uni+bigram"}
    for ngram in gram_dict.keys():
        # create the transform
        vectorizer = CountVectorizer(ngram_range = ngram)
        # tokenize and build vocab
        vectorizer.fit(df_train['trainString'])
        
        # encode document
        X = vectorizer.transform(df_train['trainString'])
        y = df_train['label']
        
        X_val = vectorizer.transform(df_val['valString'])
        
        prev_val_acc = 0
        tuned_alpha = .1
        alpha_range = np.arange(0.1,1.1,0.1)
        for alpha_val in alpha_range:#[0.5,0.8,1,5,10]:
            
            clf = MultinomialNB(alpha=alpha_val)
            clf.fit(X, y)
            
            y_val_pred = clf.predict(X_val)
            curr_val_acc = accuracy_score(y_val, y_val_pred)
            #print("val_acc for {} with alpha {} is = {}".format(gram_dict[ngram],alpha_val,curr_val_acc))
            if curr_val_acc > prev_val_acc:
                tuned_alpha = alpha_val
            prev_val_acc = curr_val_acc
            
        #fitting the model    
        clf1 = MultinomialNB(alpha=tuned_alpha)
        clf1.fit(X, y)
        
        X_test = vectorizer.transform(df_test['testString'])
        y_test_pred = clf1.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        print("Test accuracy for {} with tuned alpha {} is = {}".format(gram_dict[ngram],tuned_alpha,test_acc))
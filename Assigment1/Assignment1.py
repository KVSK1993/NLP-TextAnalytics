# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:58:11 2019
Assignment 1
@author: Karan Vijay Singh
Command to run script
python Assignment1.py -p "path to data files"
"""

import argparse
import re
import pandas as pd
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to review text files")
#ap.add_argument("-n", "--pos", required=True, help="path to pos.txt")

args = vars(ap.parse_args())

#open positive.txt
open_pos_data = open(args['path']+'\\pos.txt').read()

#open negative.txt
open_neg_data = open(args['path']+'\\neg.txt').read()

text,stopword_removed_text, label = [], [], []

#open stopwordlist pickle file
with open('stopwords.pickle', 'rb') as handle:
    stopwordsList = pickle.load(handle)
    
#to remove empty strings/reviews.  
positive_data = [review for review in open_pos_data.split("\n") if review]
negative_data = [review for review in open_pos_data.split("\n") if review]

#removing punctuations and stopwords from sentences
for line in (positive_data):
    processed_line = re.sub(r'[!"#$%&()*+\/:;<=>@[\\\]^`{|}~\t\n]', '', line)
    label.append("1")
    text.append(processed_line)
    processed_without_stop_line = ' '.join([word for word in processed_line.split() if word not in stopwordsList])
    stopword_removed_text.append(processed_without_stop_line)

for line in (negative_data):
    processed_line = re.sub(r'[!"#$%&()*+\/:;<=>@[\\\]^`{|}~\t\n]', '', line)
    label.append("0")
    text.append(processed_line)
    processed_without_stop_line = ' '.join([word for word in processed_line.split() if word not in stopwordsList])
    stopword_removed_text.append(processed_without_stop_line)

# create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['text'] = text
trainDF['stopword_removed_text'] = stopword_removed_text
trainDF['label'] = label

#for downloading the stop words from gensim library
'''import pickle
import gensim
stopwordList = gensim.parsing.preprocessing.STOPWORDS
with open('stopwords.pickle', 'wb') as handle:
    pickle.dump(list(stopwordList), handle, protocol=pickle.HIGHEST_PROTOCOL)'''
    

#Shuffling the dataframe/dataset
trainDF = trainDF.sample(frac=1).reset_index(drop=True)

#splitting the dataset into train, validation and test
train, validate, test = np.split(trainDF, [int(.8*len(trainDF)), int(.9*len(trainDF))])

def save_to_csv(cols,df,filename):
    df[cols].to_csv(filename, index = False)
    
cols_without_stopwords = ['stopword_removed_text','label']
cols_with_stopwords = ['text','label']
#saving with stopwords
save_to_csv(cols_with_stopwords,train,"trainDF_with_stopwords.csv")
save_to_csv(cols_with_stopwords,validate,"valDF_with_stopwords.csv")
save_to_csv(cols_with_stopwords,test,"testDF_with_stopwords.csv")

#saving with stopwords removed
save_to_csv(cols_without_stopwords,train,"trainDF_without_stopwords.csv")
save_to_csv(cols_without_stopwords,validate,"valDF_without_stopwords.csv")
save_to_csv(cols_without_stopwords,test,"testDF_without_stopwords.csv")


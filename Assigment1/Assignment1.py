# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:58:11 2019
Assignment 1
@author: Karan Vijay Singh
Command to run script
python Assignment1.py -p "path to data files"
"""

import re
import numpy as np
import sys
import gensim
import pandas as pd


if __name__ == "__main__":
    input_path = sys.argv[1]

    stopwordList = list(gensim.parsing.preprocessing.STOPWORDS)
    
    #open positive.txt
    open_pos_data = open(input_path).read()
    open_pos_data = open_pos_data.lower()
    text,stopword_removed_text = [], []
    
    #to remove empty strings/reviews.  
    positive_data = [review for review in open_pos_data.split("\n") if review]
    #negative_data = [review for review in open_pos_data.split("\n") if review]
    
    #removing punctuations and stopwords from sentences
    for line in (positive_data):
        processed_line = re.sub(r'[!"#$%&()*+\/:;<=>@[\\\]^`{|}~\t\n]', '', line)
        processed_line = re.sub(r'[\']'," ' ",   processed_line) #remove_Apostophe    
        processed_line = re.sub(r'[\.]',' . ',   processed_line) #replace , with ' , '
        processed_line = re.sub(r'[\,]',' , ',   processed_line) #replace . with ' . '
        processed_line = re.sub(r'[-]',' - ',   processed_line) #replace . with ' . '
        #label.append("1")
        text.append(processed_line.split())
        processed_without_stop_line = [word for word in processed_line.split() if word not in stopwordList]
        stopword_removed_text.append(processed_without_stop_line)
    
    
    
    #splitting the dataset into train, validation and test
    train_list, val_list, test_list = np.split(text, [int(.8*len(text)), int(.9*len(text))])
    train_list_no_stopword, val_list_no_stopword, test_list_no_stopword = np.split(stopword_removed_text, [int(.8*len(stopword_removed_text)), int(.9*len(stopword_removed_text))])
    
    
    df = pd.DataFrame(data ={"train":train_list})
    df.to_csv("train.csv", index = False)
    
    df = pd.DataFrame(data ={"val":val_list})
    df.to_csv("val.csv", index = False)
    
    df = pd.DataFrame(data ={"test":test_list})
    df.to_csv("test.csv", index = False)
    
    df = pd.DataFrame(data ={"train":train_list_no_stopword})
    df.to_csv("train_no_stopword.csv", index = False)
    
    df = pd.DataFrame(data ={"val":val_list_no_stopword})
    df.to_csv("val_no_stopword.csv", index = False)
    
    df = pd.DataFrame(data ={"test":test_list_no_stopword})
    df.to_csv("test_no_stopword.csv", index = False)
    
    

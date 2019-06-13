# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:30:48 2019

@author: Jasdeep
"""
import gensim
import sys
import ast

class MySentences(object):
 
    def __iter__(self):
        for i in range(1,7):
            for line in open(sys.argv[i]):
                yield ast.literal_eval(line)

if __name__ == "__main__":
    input_file_training_pos = sys.argv[1] 
    input_file_training_neg = sys.argv[2] 
    input_file_validation_pos = sys.argv[3]
    input_file_validation_neg = sys.argv[4]
    input_file_test_pos = sys.argv[5]
    input_file_test_neg = sys.argv[6]
    
    sentences = MySentences() # a memory-friendly iterator
    new_model = gensim.models.Word2Vec(sentences, min_count =2,iter = 5,size = 300 )
    
    print("The top 20 words similar to good are : ")
    print(new_model.wv.most_similar(positive = "good", topn = 20)) 
    
    print("The top 20 words similar to bad are : ")
    print(new_model.wv.most_similar(positive = "bad", topn = 20))
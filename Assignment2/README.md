## Assignment 2  
**Student ID :** `20745105`  
  
  
Classification accuracy is as follows:
Stopwords removed| text features | Accuracy (test set)
--- | --- | ---
yes | unigrams | 71.97%
yes | bigrams | 72.22%
yes | unigrams+bigrams | 74.60%
no | unigrams | 72.30%
no | bigrams | 75.17%
no | unigrams+bigrams | 76.03%


  
### Answers ###  
  
  
  
**Q (a).** Which condition performed better: with or without stopwords? Write a brief  paragraph (5-6 sentences) discussing why you think there is a difference in  performance.  
  
**Ans** The corpus with stopwords performed better than the corpus without stopwords. The difference in performance can be attributed to the fact that reviews with stopwords are able to capture the context of the review better than those without stopwords and hence the model is able to capture the context and performs better. For example, in some reviews, there were words like "never", "didnt", which got removed as stopwords and the presence of these words in the review would have helped model make better predictions.
  
**Q(b).** Which condition performed better: unigrams, bigrams or unigrams+bigrams?  Briefly (in 5-6 sentences) discuss why you think there is a difference?  
  
**Ans** Unigrams+bigrams performed the best followed by bigrams and then unigrams in both the cases as can be seen from the accuracy table. This is because bigrams+unigrams are better able to capture the context and relation between the words in a review than bigram or unigram alone. Because of which, model is able to learn the context and performs better. The bigrams performance is better than unigrams as bigrams are able to capture more meaning and context than unigrams alone.
  
  
**P.S.**  
Have used gensim library for tokenising and removing the stopwords from the reviews and have used CountVectorizer.
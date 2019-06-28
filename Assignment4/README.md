## Assignment 4 
 
**Student ID :** `20745105`  
  
Report your classification accuracy results in a table with three different activation functions in the hidden layer (ReLU, sigmoid and tanh). What effect do activation functions have on your results? What effect does addition of L2-norm regularization have on the results? What effect does dropout have on the results? Explain your intuitions briefly (up to 10 sentences).


Classification accuracy is as follows:
Activation Function| L2 Regularisation | Dropout | Accuracy (val set)
--- | --- | --- | ---
relu | - | - | 0.7352
tanh | - | - | 0.7236
sigmoid | - | - | 0.7306
relu | 0.01 | - | 0.6917
relu | 0.001 | - | 0.7080
relu | 0.0001 | - | 0.7264
relu | 0.0001 | 0.2 | 0.7328
relu | 0.0001 | 0.3 | 0.7315
relu | 0.0001 | 0.4 | 0.7339

Test accuracy for the best tuned model with 
activation:sigmoid, 
l2 regularisation:0.0001, 
dropout rate:0.4
is 0.7436.
  
### Answers ###  
  Effect of activation functions:
Activation functions introduce non-linearity in the network (non-linear complex functional mappings between the inputs and response variable). The model with Relu activation function gives the best results on the validation set in our case. This can be attributed to the fact that sigmoid and tanh functions suffer from vanishing gradient problems and the major benefits of ReLUs are sparsity and a reduced likelihood of vanishing gradient. 
We know, relu is _R(x) = max(0,x) i.e if x < 0 , R(x) = 0 and if x >= 0 , R(x) = x_.  So when x>0, gradient has a constant value which results in faster learning unlike in sigmoid, the gradient becomes increasingly small as the absolute value of x increases.
The other benefit sparsity arises when x≤0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoids on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations.[1]

  Effect of L2 Regularization on the model :
In general, Regularization is a technique to prevent overfitting in the models i.e to make complex models simpler. L2 regularization tends to push/shrink  the weights of the model towards zero. Higher the value of L2 regulariser, more will be the shrinkage of weights and hence can lead to underfitting. 
  Lower the value of L2 regulariser, higher the weights, and hence complex model and can lead to overfitting. So, we need to tune the value of l2 norm so that model reaches the sweet spot in the bias-variance trade-off i.e. neither underfitting nor overfitting. In our model, 0.0001 value of l2 norm gives better validation accuracy than 0.001. So, with the addition of L2 regulariser in the hidden layer results in further improvement in the accuracy.
  
  Effect of dropout :
It is a simple and effective regularization method in which we probabilistically drop out nodes in the network which helps in improving the generalization performance of the model as it prevents neurons from co-adapting too much. As we can see in our case, Dropout  combined with l2 regularization yields a further improvement in the model. A 0.4 dropout rate in the hidden layer yields best results in our case and this has resulted in improving the accuracy of the overall model by 2%.


References
[1] [https://www.cnblogs.com/casperwin/p/6235485.html](https://www.cnblogs.com/casperwin/p/6235485.html)
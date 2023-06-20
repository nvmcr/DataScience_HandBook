# Deep Learning
Deep learning algorithms are inspired by the structure and function of the human brain, allowing machines to learn from vast amounts of data, recognize patterns, and make accurate predictions. From image and speech recognition to natural language processing and autonomous vehicles, deep learning is everywhere these days. I will try to cover the most talked about topics in this chapter.
# Loss Functions
As discussed vastly in [Machine Learning chapter](https://github.com/nvmcr/DataScience_HandBook/tree/main/Machine_Learning), a loss function is something that tells us how well our model is working. In deep learning, we will most look into classification cases. Multiclass SVM (also called hinge loss) and Softmax Loss.
## Multiclass SVM
It is the same loss function used in SVMs for multi-class classification. Say we have a classifier and scores vector is, $s=f(x_i, W), then the loss is given as:

$$ L_i = \Sigma_{j\neq y_i} max(0, s_j-s_{y_i}+1) $$

Here $y_i$ is the correct label. The main aim of this loss function is to have the highest score in $s_{y_i}$. If it doesn't have the highest scores, loss is given as the difference between an actual class score and sum of all other classes scores.

Ex: Say we have a *cat* image and our scores vector looks like this:

|Class|Score|
|---|--|
|cat|3.2|
|car|5.1|
|frog|-1.7|

The loss is calculated as $max(0, 5.1-3.2+1) + max(0, -1.7-3.2+1) = 2.9$ Loss is calculated for the classes other than the actual class.

We generally added regularization like L1 or L2 to the loss function.

## Softmax Loss
Scores given by hinge loss don't mean much. Those are just some arbitrary values. But softmax loss given scores as probabilities for better interpretation. 

![](Images/softmax.png)

Same as before we have scores like 3.2 for cat. This is passed through an exponential function and then is normalized by the sum of the exponential value of all classes. This exponential and normalizing is called softmax function, $P(Y=k|X=x_i) = \frac{e^{s_k}}{\Sigma_j e^{s_j}}$. The final loss is given by $L_i=-logP(Y=y_i|X=x_i)$ which is nothing but doing cross-entropy loss for a softmax function. 

In general, people call this loss as cross-entropy loss too. But it is actually cross-entropy loss for a softmax function.

# References
1. [Deep Learning by Ranjay Krishna and Aditya Kusupati](https://courses.cs.washington.edu/courses/cse493g1/23sp/schedule/)
2. [Machine Learning CSE 446 UW](https://courses.cs.washington.edu/courses/cse446/22au/)

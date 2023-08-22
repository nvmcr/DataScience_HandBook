<details>
<summary>Table of Contents</summary>

## Table of contents
There will be a lot of content in this markdown file. Please use the github's interactive navigation. (Too lazy to write/generate TOC)
![toc](Images/toc.gif)

</details>

# Introduction
## What is Machine Learning
Fancy way
> A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 
—Tom Mitchell, 1997

I would say its just making a machine learn using data so that it can apply its learning on unseen data.
## Types of Machine Learning<sup>1</sup>
### Based on human supervision
1. Supervised Learning 

If the training data has labels, it is supervised.

2. Unsupervised Learning 

Machine trained without labelled data.

3. Semisupervised Learning 

Only a few samples of the data are labeled. The machine learns to label the unlabelled samples. An example is a photo-hosting service like Google Photos.  Once
you upload all your family photos to the service, it automatically recognizes that the
same person A shows up in photos 1, 5, and 11, while another person B shows up in
photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all
the system needs is for you to tell it who these people are. Just one label per person,
and it is able to name everyone in every photo, which is useful for searching photos.

4. Reinforcement Learning 

The learning system called an agent
in this context, can observe the environment, select and perform actions, and get
rewards in return (or penalties in the form of negative rewards). It
must then learn by itself what is the best strategy, called a policy, to get the most
reward over time. A policy defines what action the agent should choose when it is in a
given situation.
### Batch vs Online Learning
In batch learning, the system must be trained
using all the available data. This will generally take a lot of time and computing
resources, so it is typically done offline. First, the system is trained, and then it is
launched into production and runs without learning anymore; it just applies what it
has learned. This is called offline learning.

In online learning, you train the system incrementally by feeding it data instances
sequentially, either individually or in small groups called mini-batches. Each learning
step is fast and cheap, so the system can learn about new data on the fly, as it arrives. One important parameter of online learning systems is how fast they should adapt to
changing data: this is called the learning rate. If you set a high learning rate, then your
system will rapidly adapt to new data, but it will also tend to quickly forget the old
data (you don’t want a spam filter to flag only the latest kinds of spam it was shown).
Conversely, if you set a low learning rate, the system will have more inertia; that is, it
will learn more slowly, but it will also be less sensitive to noise in the new data or to
sequences of nonrepresentative data points (outliers).

### Generative vs Discriminative Models
A Discriminative model ‌models the decision boundary between the classes (conditional probability distribution p(y|x)). A Generative Model ‌explicitly models the actual distribution of each class (joint probability distribution p(x,y)). In final both of them is predicting the conditional probability P(Output | Features). But Both models learn different probabilities. In Math, 

Training classifiers involve estimating f: X -> Y, or P(Y|X)

Generative classifiers 
* Assume some functional form for P(Y), P(X|Y)
* Estimate parameters of P(X|Y), P(Y) directly from training data
* Use Bayes rule to calculate P(Y |X)
* Naïve Bayes, 
Bayesian networks, 
Markov random fields, AutoEncoders, GANs.

Discriminative Classifiers.

* Assume some functional form for P(Y|X)
* Estimate parameters of P(Y|X) directly from training data
* ‌Logistic regression, SVMs, ‌CNNs, RNNs, Nearest neighbors.

### Parametric vs Non-parametric Models
A parameter is something that is estimated from the training data and changes (learned) while training a model. They can be weights, coefficients, support vectors etc.

A parametric model summarizes data with a set of **fixed-size** parameters (independent on the number of instances of training). Parametric machine learning algorithms are which optimizes the function to a known form. 
For example, we already assume the function is linear in linear regression. If y = b0 + b1\*x, we fixed parameters to b0 and b1, which are learned while training. Examples include: Logistic Regression, linear SVM (w<sup>T</sup>x + b = 0), Linear Discriminant Analysis, Perceptron, Naive Bayes, Simple Neural Networks.

Nonparametric models are those which do not make specific assumptions about the type of the mapping function. They are prepared to choose any functional form from the training data, by not making assumptions. The word nonparametric does not mean that the value lacks parameters existing in it, but rather that the parameters are adjustable and can change. 
For example, in k-nearest neighbors' algorithm we make predictions for a new data instance based on the most similar training patterns k. The only assumption it makes about the data set is that the training patterns that are the most similar are most likely to have a similar result. Examples include: k-Nearest Neighbors, Decision Trees, SVMs.
## Maximum Likelihood Estimation
In one line, MLE is a method that determines values of the parameters of a model such that they maximise the likelihood of observed data given a probability distribution. In simpler terms, say we have a random sample of data from some gaussian distribution. We need to find out which gaussian curve (need to find mean and variance) is most likely responsible for creating our data sample. We use MLE to find the parameters mean and variance.

Say we have a random sample from a sequence of coin trails (H,H,T,T,..) from a binomial distribution with k heads out of n flips. Let us assume there is a parameter $\theta$ which is probability of getting heads. W.K.T, the probaility distribution of the data with a fixed unknown parameter $\theta$ is represented by, 

$$ P(D|\theta) = \theta^k\*(1-\theta)^{(n-k)} $$

MLE helps us find $\theta$ which maximises the probability of obtaining the data that we saw.

$$ \hat{\theta}\_{MLE} = \arg\max_{\theta} P(D|\theta) $$

In the above eqn, arg max means the value that returns the maximum value of a function. (Say we have a function, f(x) = x + 10 where x is in range \[1,5]. Max of the fn will be f(5) = 5 + 10 = 15. But arg max would be 5 beacuse that is the value that returned max value of function.)

We will consider log likelihood as it is easy for calculation. Log function is a monotonically increasing function which means the arg max would be same for with log or without log.

$$ \hat{\theta}\_{MLE} = \arg\max_{\theta} \log P(D|\theta) = \arg\max_{\theta} \log \theta^k*(1-\theta)^{(n-k)} $$

We know that derivative is zero at maxima and minima. So we need to find at which $\frac{d}{d\theta} \log P(D|\theta) = 0$. Calculating, we will get $\theta = \frac{k}{n}$.

## Linear Regression
It is a parametric model where we assume our data is linear i.e our output y(house price) is a linear function of feature x(sq.ft). There might be d number of different features like sq.ft, no.of rooms, etc and we represent number of samples in training data with n. As we assume the data is linear, w.k.t y = mx where m is our parameter which is our slope. We represent the parameter/weight with w. So for each sample in our data, y = wx. The error in our model is represented with e. A loss/cost/objective function is used to know this error. We optimize this loss function to get the least error. 

![Linear Regression](Images/lr.png)

Linear Regression has a closed form solution i.e we have a formula to get to a solution without any iterative approaches of trail and error like gradient descent. We can find this solution by taking arg min of loss function because we need to find the weight which will minimize the loss function. Commonly we use Least Squares as the loss function.

![Closed Form Solution](Images/cfs.png)

We can take derivative to find the minimum and we will obtain:

$$ \hat{w}\_{LS} = (X^TX)^{-1}X^Ty $$

> Least squares is preferred over absolute values because, LS is differentiable which is a necessity for gradient descent approaches. But more importantly, least squares closed form solution is equal to MLE closed form.

But data is not always linear, we then use polynomial regression to fit the data.

![Polynomial Regression](Images/pr.png)

The more the degree of the polynomial the better the fit but the more is the issue of overfitting.
# Core Concepts
## Bias-Variance Tradeoff
Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. The more bias means more error as the model(usually very simple models) couldn't properly fit to the data which is called underfitting. Variance is the variability of the model prediction and it tells about the spread of the data. Model with high variance (usually a complex model) pays a lot of attention to training data and fits almost exactly. When we use model with high variance on unseen data, the model couldn't generalize/adapt to the new data has it is too rigid and fixed with training data. This is called over-fitting. We desire low bias and low variance for our model to generalize well.

To summarize, if the model is simple(less complex), there will be more bias but generally less variance and if the model is too complex, there will be less bias but more variance. To choose the best model, there is a tradeoff between bias and variance to decrease the overall error given by 

$$ Total Error = Bias^2 + Variance + Irreducible Error $$

One popular way of talking about model complexity is **Occams Razor** which says “Everything should be as simple as it can be, but not simpler".
## Cross Validation
To get the accurate measure of how well model works, its important to have take some data out before training of model and use this unseen data to test the data. Often, 80% of the data is set for training and 20% for testing. Instead of putting all faith in the test data, we can use cross validation to cross check the model performance and more importantly to tune hyper-parameters.

> Hyper-parameters are parameters that are fixed before training the model. They are not learnt but instead guessed with trail and error like batch size, number of neighbors in knn etc..

Cross-validation is a process of taking some data out called validation data and train the model without validation data. Test the trained model using the validation data. Now change the hyper parameters and repeat process again to get best hyper-parameters as well know model performance.

There are various forms of cross validation. 
1. K-fold CV

The most popular one. We randomly split the training data into k splits. Eachtime we take one split of k splits as validation data and train the model on remaining splits and note the error. Now take another split into validation data and repeat the process. Finally take the average error as the overall error of the model. 

2. Leave One Out (LOO) CV

Same as k-fold but each split will only have a single sample of the data. It gives an unbiased estimate of the error but computationally expensive and time taking.

3. Stratified CV

What if our random split contains same label of the data. This means our split is not a proper estimate of the training data. This testing our model on this data doesn't give a true estimate. So many prefer stratified cv where each fold has roughly the same proportion of target classes as the original dataset. For example: If we have a binary classification problem with 80% positive examples and 20% negative examples, stratified CV ensures that each fold of the CV also contains roughly 80% positive and 20% negative examples.
### Hyper Parameter Tuning
We have several methods to test different hyper-parameters. These methods are used along with cross validation to get true estimate of a hyper-parameter.

1. Grid Search CV

It involves defining a grid of possible hyperparameter values and then systematically evaluating the performance of the model for each combination of hyperparameters in the grid. For example, we define following `C_values = [0.1, 1, 10], kernel_values = ['linear', 'rbf', 'poly']` for SVM. Gridsearch will try all combinations (9 for example). As we can see, this method is slow and computationally expensive.

2. Random Search CV

We give a range of values for each hyper parameter. We randomly select values from the search space for each hyperparameter and use them to train and evaluate the model. We can set the number of iterations or trials to perform, and in each iteration, we randomly select a value for each hyperparameter of interest from its search space. This is fast but not accurate as it is random.

3. Bayesian Optimization
In contrast to random or grid search, here we keep track of past evaluation results which the optimization use to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function: P(score|hyperparameters). It starts with a few random combinations but it chooses next set of parameters by analyzing the results from previous chosen parameters.
## Evaluation Metrics
### Root Mean Square Error
It is a commonly used evaluation metric in regression tasks. It measures the square root of the average squared difference between the predicted values and the true values.

$$ RMSE = \sqrt(1/n * \sum((y_{pred} - y_{true})^2)) $$

RMSE is preffered over Mean Absoulte Error (MAE) because RMSE penalizes large errors due to squaring but MAE treats all error same.
### R squared
as the coefficient of determination, is a statistical measure that represents the proportion of the variance in the dependent variable (i.e., the target variable) that is explained by the independent variables (i.e., the features) in a regression model. In other words, it represents how well our regression model fits the given data. R-squared is a value between 0 and 1, where 0 indicates that the model does not explain any of the variability in the dependent variable, and 1 indicates that the model explains all of the variability in the dependent variable.

We first calculate variance(sum of squared residuals) along the mean line of the data and later calculate variance along the new model line that is fitted to the data.

$$ R^2 = \frac {Var(mean line)-Var(fitted line)}{Var(mean)} $$
### Confusion Matrix
Used for classification tasks. In a confusion matrix, `True Positives` are the *positive* samples that are predicted as **positive**, `False Positives`(also called **Type 1 errors**) are the *negative* samples that are predicted as **positive**. Similarly `True Negatives` are the *negative* samples that are predicted as **negative**, `False Negatives` (Type 2 errors) are the *positive* samples that are predicted as **negative**.
### Accuracy
It is calculated as the number of true positives and true negatives divided by the total number of instances in the dataset. In most of the cases, the distribution of classes is imbalanced, meaning that some classes have far fewer instances than others. In such cases, a classifier that always predicts the majority class will have a high accuracy, even though it is not very useful. So just accuracy is not good enough estimate.
### Precision, Recall and F1 Score
Precision says From total predicted positives, how many are actual postives. $Precision = \frac{True Positives}{True Positives + False Positives}$. Precision means percentage of the results that are relavant.

Recall says From total positives how many are actual positives. $Recall = \frac{True Positives}{True Positives + False Negatives}$. Recall means the percentage of total relevant results that are correctly classified.

Lets consider we are doing a medical test. Diagnoizing a disease is high priority. We have to take care of positive tests marked as negative (False Negative). We can for an extent misdiagnoize a negative test as positive (False Positive). Here Recall is preferred over Precision.

Say we are doing email spam detection. We can not mark a normal mail as spam (False Positive) but we can for an extent mark a spam mail as normal(False Negative). Here Precision is preferred over recall.

If we don't know what to prefer over precision and recall, F1 score is used as it is a **harmonic mean of both**. $F1 Score = \frac{2* Precision * Recall}{Precision + Recall}$.
### Mean Average Precision (MAP)
MAP stands for Mean Average Precision, which is a common evaluation metric used in information retrieval and object detection tasks. MAP measures the average precision at different recall levels.

For example, suppose the search engine retrieves 20 documents for the "machine learning" query. We evaluate the precision and recall of the search engine at different recall levels by examining the top 5, 10, and 20 retrieved documents.

Suppose we find that the search engine correctly retrieved 2 out of the 5 relevant documents, 5 out of the 10 relevant documents, and 8 out of the 20 relevant documents. We calculate the precision at each recall level and then take the average to obtain the MAP for the "machine learning" query.

For example, the precision at recall level 5 is 0.4, the precision at recall level 10 is 0.5, and the precision at recall level 20 is 0.4. We take the average of these values to obtain the MAP for the "machine learning" query, which in this case is 0.433.
### Area Under Curve (AUC)
It is used in binaru classifiers. True Positive rate(TPR) is the number of positives correctly classified out of all positives (TP/(TP+FN)) and False Positive rate(FPR) is number of negatives classified as positives out of all negatives (FP/(TN+FP)). Receiver Operator Characteristic (ROC) curve is plotted with FPR on X-axis and TPR on Y-axis and the curve is determined by threshold. 

Say we have a binary classification problem of yes and no. We set a threshold(0.5) above which we classify something as yes and below the threshold as no. We train a logistic regression and we calculate TPR and FPR for this. If TPR=1 and FPR=0, it is a perfect classifier. If TPR=1 and FPR=1, then it is a random classifier. We plot the point on the graph. Now we can change the threshold and plot the calculated values of TPR and FPR again. We repeat this process for several thresholds to get ROC curve which shows us the best threshold. 

![ROC](Images/auc.png)

We can change the model now to say Random Forest and plot the ROC curve. We can find the areas of both ROC curves. The more the area the better the model.

AUC-ROC provides a useful summary of the classifier's performances, and can also help us to choose the optimal threshold for our specific problem.
## Gradient Descent
What to if we **dont have a closed form solution** like linear regression? This is where iterative approaches come into play. Lets have a look at convexity first.
### Convexity
Convexity is a mathematical property of a function, and it plays an important role in optimization problems. A function is said to be convex if its graph lies entirely above or on a line connecting any two points on the graph. Intuitively, this means that the function curves upward or is flat, and does not curve downward. 

![convex](Images/convex.png)

More formally, $f((1-\lambda)x+\lambda y)\le (1-\lambda)f(x)+\lambda f(y)$ .This inequality essentially states that the value of the function f(x) is always above or on the line connecting any two points on its graph.

Why care convexity?
* All local minimum in convex functions are global minima.
* Efficient to optimize.

If our loss functions are convex functions, we have to find the minima of the function to find the minimum parameter where loss is less. If these functions are non-convex, there might be multiple minima. So we reach a local minima, we might assume it is the global minima.
### GD
Gradient descent is an optimization algorithm used to minimize a function by iteratively adjusting the parameters in the direction of steepest descent of the function.

In gradient descent, we start with an initial guess (random) for the parameters (w) of the function(loss/cost fn) we want to optimize. We then calculate the gradient of the function with respect to the parameters, which tells us the direction of the steepest ascent. To minimize the function, we move in the opposite direction of the gradient, i.e., in the direction of steepest descent.

The size of each step we take in the direction of steepest descent is controlled by a parameter called the step size/learning rate($\alpha$). If the learning rate is too small, the optimization process can be slow, while if it is too large, the process can be unstable and diverge.

We continue this process of computing the gradient and adjusting the parameters until we reach a point where the gradient is close to zero, i.e., we have found a local minimum of the function. This point is the optimal set of parameters that minimize the function.

![gd](Images/gd.gif)

$$ w_t \leftarrow w_t - \alpha\frac{\partial L(w)}{\partial w_t} $$
### Batch/Vanilla GD
Vanilla is just a fancy term used for saying something basic without additional features. We will consider our loss function explicitly as $L(w) = \frac{1}{2} (w^Tx - y)^2$. Using half for ease of calculation. If we calculate the gradient of this loss function we will get $\frac{\partial L(w)}{\partial w_t} = (w^Tx - y)x$. We can use this value in the gradient calculations.

In batch gradient descent, the gradient is computed over the entire training set at each iteration. This can be computationally expensive for large datasets, but it leads to a more accurate estimate of the true gradient.

for t=1,...T do

$w \leftarrow w + \alpha\Sigma_{i=1}^n (y^i - w^Tx^i)x^i$

return w
### Stochastic/Incremental GD
The gradient is computed on a single training example at each iteration. This can be faster and more memory-efficient than batch gradient descent, but it can lead to a noisy estimate of the gradient, which can make it difficult to converge to the optimal solution. Difference in Stochastic and Incremental is, we randomize the data point in SGD and don;t randomize in Incremental. Below is logic for incremental as i is not randomize.

for t-1,...T do

  for i=1,...n do

$w \leftarrow w + \alpha(y^i - w^Tx^i)x^i$

return w

### Mini-Batch GD
the gradient is computed on a small random subset (mini-batch) of the training set at each iteration. This combines the advantages of batch gradient descent and stochastic gradient descent, as it is more efficient than batch gradient descent while providing a less noisy estimate of the gradient than stochastic gradient descent. We divide the data, D into partition of $D_1,D_2,..D_k$ similar to k-fold with equal size in folds.  

for t-1,...T do

  for l=1,...k do

$w \leftarrow w + \alpha\Sigma_{i\in V_l} (y^i - w^Tx^i)x^i$

return w

### Coordinate Descent
The goal is still the same, we need to get minimum parameter, w that minimizes the loss function. But when we have a lot of coordinates (multivariate function/multiple features), it becomes difficult to minimize w in all coordinates, so we choose only one coordinate to minimize.

for t-1,...T do

  Choose a random coordinate, c from $x_1,x_2,..$ or loop through a list of values
  
$w \leftarrow min L(w)$

return w

We randomly choose a coordinate and try to minimize weights/paramter only in that coordinate. All other coordinate's parameter will be fixed. Thus we are only optimizing a single coordinate at a time. There is no learning rate/step size in coordinate descent. 

![CD](Images/CD.png)

In the above convergence plot, there are two co-ordinates say $x_1,x_2$ plotted in X and Y-axis respectively. We start with optimizing at $x_2$ and then $x_1$ and then $x_2$ and so on till we reach convergence.

The advantage of coordinate descent is that it can be efficient when the function to be optimized has some structure that makes it separable in terms of the variables like lasso regression. In other words, when the function can be expressed as a sum of functions, each depending on a single variable. In this case, coordinate descent can converge more quickly than other optimization methods.

However, coordinate descent can be slow when the variables are highly correlated or when the function is not separable.
# Regularization Models
## Ridge Regression (L2)
For a linear model, $y=b+ w_1x_1 + w_2x_2 + ... + w_dx_d$. If model overfits, then w is large thus small changes in x will make the model vary a lot. So by limiting weights, we can improve model generalization. W.K.T, loss/cost function is to find minimum weights, w such that loss is the least. In ridge regression (also called shrinkage method), we use a regularizer $||w||^2_2$ (l2-norn) to control weights.

$$ \hat{w}\_{ridge} = \arg\min_{w} \Sigma_{i=1}^n (y_i - x_i^Tw)^2 + \lambda||w||\_2^2 $$

where $\lambda$ is a regularization coefficient. More the value more the regularization.

$$ \hat{w}\_{ridge} = (X^TX + \lambda I)^{-1}X^Ty $$
## Lasso Regression (L1)
Lasso (Least Absolute Shrinkage and Selection Operator) is similar to ridge, except uses L1 instead of L2 norm.

$$ \hat{w}\_{lasso} = \arg\min_{w} \Sigma_{i=1}^n (y_i - x_i^Tw)^2 + \lambda||w|| $$

Unlike ridge where the model can shrink the values close to zero, lasso can shrink the weights to zero making the model sparse. Thus eleiminating the features that are not necessary. 
## Elastic Net Regression
This is a combination of L1 and L2. In this regularization coefficient looks like,

$$ \lambda L1 + (1-\lambda)L2 $$
# Classification Models
## Logistic Regression
Similar to linear regression except the target is categorical instead of continous (so its not exactly regression problem). Input of logistic can be continous or categorical. The model aims to categorize the input samples into classes. The decision boundary is linear. The main difference from linear regression is the how we fit the line.

![](Images/logreg.png)

The predicted values should stay within the range of 0 and 1 and we need something like a S function to classify the samples. This function make a sample close to 1 or 0 if it is sure (high probability). Generally sigmoid function is used in this case. As we know in linear case y = wx+b, instead of directly using y, we first map it within range of 0,1 by using $\sigma (y) = \frac{1}{1+e^{-wx+b}}$ We also have a threshold value, usually 0.5, to map values between 0 to 1. For example, if we need have a value of 0.7 as the output of sigmoid function, then we make that prediction as 1 as 0.7 $\ge$ 0.5. Similarly if it is 0.3, then the prediction is 0.

To better fit (optimize) the sigmoid, we need a loss function. We cant use the L2 loss as it will become non convex for logistic. So we use something called logistic loss or binary categorical cross entropy. 

$$ Loss = \(-\sum_{i=1}^{C} y_i \log(p_i)\) $$

where:
C : Number of classes; 
y_i : True label (ground truth) for class i; 
p_i : Predicted probability for class i

Negative sign is because log of values between 0 and 1 are negative, but for general narrative of less value of loss is preferred, we add a negative sign. 
For binary classification, loss is often calculated as average cross entropy of all n samples. 

$$Loss=-\frac{1}{n} \sum_{i=1}^{n} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)$$

Here $y_i$ will be either 0 or 1 and $p_i$ is the value we get after the sigmoid function. The weights are optimized using this loss and gradient descent similar to linear regression. 

### Other Loss Functions
#### 0-1 Loss
![0-1](Images/binaryloss.png)

This is the ideal loss function. If we make a correct prediction, loss is 0 and wrong prediction gives loss 1. But we can't use this, as the loss function is not convex so can't be optimized.
#### Quadratic Loss
![](Images/quadraticloss.png)

Similar to loss function used in linear regression. If observed, we get high loss even if we predicted correctly. See the left side parabola increasing loss when it should be 0. 
#### Sigmoid Loss
![](Images/sigmoidloss.png)

$$ l(\hat{y},y) = \frac{1}{1+e^{y\hat{y}}} $$

This is a differentiable approximation of 0-1 loss. The edges of funtion is flat which makes gradients long time to converge.
#### Categorical Cross Entropy/Logistic Loss
![](Images/logisticloss.png)

Resoles all our issue. Thus quite popular choice.

## Support Vector Machines
To choose a best linear classifier to seperate say two classes, how should the decision boundary be? Should the decision boundary be close to class A or class B. Ideal boundary will be far from the both classes because small change should change the output drastically thus making the model more generalizable. In a 2D case, decision boundary is a line and in higher dimensions it is a hyperplane. The distance is measured from a point in each class which are nearest to the hyperplane. So only the points closest to the hyperplane matter in finding the decision boundary. Those points from each class are called support vectors. The distance from a point i is measured by:

$$ d = \frac{y_i (w^Tx_i + b)}{||w||\_2} $$

What if the data is not linearly seperable (for higher dimenions)?

One get away method is to introduce a slack where we ignore few points while optimizing the distance. A better way is to lift the features to higher dimensions (kernel method). The general idea is that making a non seperable 2D features to a 3D features making them seperable. Say we have X1 and X2 features, we can apply non linear transformation to make it 3D. For example, X1 and X2 can be transformed to \[X1^2, X1X2, X2^2] with degree 2 and can increase further dimensions by using features like X1^2+X2^2. But its hard to know which feature map will work to get linearly seperable feature space. Also if we go to much higher dimensions, we can not use entire feature space as it requires a lot of memeory. This is where kernels trick helps. 
### Kernel Trick
The kernel trick overcomes this limitation by implicitly mapping the original feature space into a higher-dimensional space using a kernel function. Before jumping in details lets provide some context. 

Say we have n samples(rows) of data and p features (columns). In general we expect n >> p but in many real world problems it is the opposite like number of pixels in image is much greater than number of images. In machine learning problems we will be handling size of np (size of x). If $p >> \implies np >> n^2$. Now we have something called dual formation which is an alternative approach of minimizing loss function where we only need to solve $x^Tx'$ where x and x' are two different data points. This dot product requires n^2 dot products instead of np which is more optimized. Which means when we go to higher dimenions, we don't need to explicilty store the values of \[X1^2, X1X2, X2^2] but we only need the dot products between the data points. So when we go to higher dimensions, we first *explicitly* transform the the np data to nP(P>>p higher dimenions transformation) and then calculate dot products T(x).T(X') (n^2) to use in our loss functions. The route is $np \rightarrow nP \rightarrow n^2$.

As we can see nP is too large to store due to higher dimensions. We need an alternative route to reach T(x).T(x')  from x.x' without using nP. This is where kernel trick comes. It starts with np and then calculate the original dot products n^2 and then using kernel function it transforms original dot product to transformed dot products (n^2). The route here is $np \rightarrow n^2(original) \rightarrow Kernel function \rightarrow n^2(transformed)$. Kernel function computes the similarity or inner product between pairs of data points in the higher-dimensional space, without explicitly calculating the coordinates of those points.  Thus everytime if we want to do a dot product we can do a kernel function and use it implicitly. See below where we have different kernel functions and how they use original data points dot products but *implicitly* gives the transformed dot products. 

![](Images/kernels.png)

Note: Though we cant see the dot product in RBF kernel, if we break down the norm, we get a dot product for infinite dimensions projection! The sigma is a hyperparameter which determines how rigid or smooth our hyperplane looks.

Watch this video for a great visual explanation: https://www.youtube.com/watch?v=Q7vT0--5VII
## Naive Bayes
Like the name mentioned, this model is based on the Bayes theorem and conditional probabilities. We will look into Multinominal Naive Bayes first. Since the model is mainly used for text based classifications, will take an example of classifying a message as spam or normal. Training data will look like this:

|Occurence of word 'Money'|'Hello'|'Dear'|Label|
|--|--|--|--|
|5|2|0|Spam|
|0|1|2|Normal|

Say we have a message 'Hello Money'. We need to calculate P(Spam|Hello, Money) and P(Normal|Hello, Money). Based on the probablities we decide the output label. We will take P(Spam|Hello, Money) for calculation. From applying Bayes theorem at every step below,

$$ P(Spam|Hello, Money) = \frac{P(Spam, Hello, Money)}{P(Hello, Money)} $$

$$ P(Spam, Hello, Money) = P(Hello | Money, Spam)P(Money, Spam) $$

As we know by doing bayes $P(Money, Spam) = P(Money|Spam)P(Spam)$,

$$ P(Spam, Hello, Money) = P(Hello | Money, Spam)P(Money|Spam)P(Spam) $$

We do a **naive** assumption that features are independent i.e word Money is independent of word Hello (order of occurence of these words is not important). Thus even if the message is very large, we can ignore all the complex calculations.

$$ P(Spam, Hello, Money) = P(Spam)\*P(Hello|Spam)\*P(Money|Spam) $$

This is the only formula we need to calculate output label. To do a prediction for a new sample, say Hello Money Money will calculate two probabilities as `P(Normal Message)*P(Hello|Normal)*P(Money|Normal)^2` and `P(Spam Message)*P(Hello|Spam)*P(Money|Spam)^2`. Whichever probability is higher decides the output.

### Gaussian Naive Bayes
Now for the Gaussian Naive Bayes (features are continous unlike nominal as above), the algorithm is as follows:
1. Lets take an example of outputs being buys a ticket or doesn't buy a ticket. Let the features be distance of theatre, price of ticket,..
2. Firstly, prior porbabilites, P(Buys) and P(Doesn't Buy) are calculated similar to above case.
2. From the samples, a gaussian distribution is plotted for every feature given buys a ticket and doesn't buy a ticket i.e from data, for all 'distance' samples where a person buys a ticket, we plot normal distribution using mean and std from the data and similarly for all 'distance' samples where a person doesn't buy ticket, we plot normal distribution and we repeat this for all features. 
3. To do a prediction for a new sample, we calculate probabilities as 

$$ P(Buys Ticket|distance=value,price=value) = P(Buys Ticket)\*P(distance=value|Buys Ticket)\*P(price=value|Buys Ticket) $$

Except for prior probabilties, other probabilties are likelihoods calculated from gaussian distribution we plotted. Likelihood is the y axis value at X-axis (price or distance = given value).

5. Sometimes the above probabilities are calculated in log scale to prevent underflow (if likelihood is nearly zero, the system can't represent such a small value) as log scale turn product to addition.
6. Similarly we calculate the probability of not buying ticket and compare the value.

6. Sometimes the model can even be used to find the most important features from the distribution plots if the features differs a lot.
## K-NN
K Nearest Neighbors is a non parametric model that can be used for both classification and regression. 
1. Given a new, unseen input feature vector, the algorithm identifies the k nearest neighbors to that vector in the training dataset. "k" is a user-defined parameter that determines the number of neighbors to consider.
2. The distance metric (e.g., Euclidean distance or Manhattan distance) is typically used to measure the distance between the new input vector and the training instances.
3. The majority class label (for classification) or the mean/median of the target values (for regression) among the k nearest neighbors is then assigned as the predicted label or target value for the new input vector.

The key idea behind the KNN algorithm is that similar instances are likely to have similar labels or target values. By considering the labels or target values of the k nearest neighbors, the algorithm makes predictions based on the local structure of the data. Remember to pick od number of k to avoiding tie between categories, scale the features (normalize) first as this is a distance based algorithm and for categorical features, appropriate distance metrics (such as Hamming distance or Jaccard similarity) need to be used instead of Euclidean or Manhattan distance.
# Tree Methods
Tree methods are the best choice for the tabular data adn can be used for both classification and regression. Lets see how they work.
## Decision Trees
1. Decision trees start with the root and have Nodes (decisions). Each of these nodes can again have nodes. The node without any following nodes (children) is called a leaf.
2. Imagine we have a dataset with features 
3. Each of these node decisions are decisions taken on a feature of the dataset (Hates Rose<--(left is no)loves Titanic(right is yes)-->Loves Rose). Generally the leaves are output.
4. If there are multiple features, to decide which feature goes first on the tree, following method is used (**Gini Impurity**).
    * Each feature is soley used to predict the outcome. Say the outcome is Buys ticket or doesn't buy. Feature be, Likes Actor (Yes or No). 
    * Out of all samples, number of yes and no in output is calculated for liking actor and not liking actor.
    * Which means each children (leaf) of this node will have a number of yes and no.
    * Gini Impurity for each leaf is calculated as $1-(Probability of Yes)^2-(Probability of No)^2$.
    * Total gini impurity of the node is calculated by averaging impurity of each leaf.
    * If the number of samples on each of the leaf is not equal take the weighted average of each leaf and do average for node i.e `Weighted avg of each leaf = (Number of samples in leaf/Total samples in the node)*gini impurity of the leaf`.  

    What if the feature has numeric data?  
    * Let the feature be distance of theatre. Sort that column and then calculate avergae of each of the adjacent cells and calculated gini impurity.
    * For example, let the sorted values be 7,12,18,35,38. Calulating avg of each adjacent cell gives 9.5,15,26.5,36.5.
    * Each node will be like Distance < 9.5, then each of the leaf will have outputs yes and no. Then calculate gini impurity for this node.
    * Repeat the above step for all 15,26.5,36.5 and decide the best value.
    * This best value node, Distance < 15 will compete with other features like, likes actor to be decided to go first.
4. Now we will have one feature decision as the root. If the children of this root have any impurity i.e few Yes and few No (Pure means full yes or full no), then that children will become a Node and step 3 is repeated again till we reach a node that is pure.
5. If the node is pure then it becomes a leaf.
6. If the leaf has No, then the decision is Doesn't buy ticket and vice-versa.

What if the output is a regression value?  
1. Choosing a feature is done by gini impurity but instead of taking yes and no probability, we take squared residual errors. 
2. Say distance be a feature and price be output. If distance < 3, the children will be Price = (Average outputs in data when distance is less than 3) and other children will other end average.
3. Each children gini impurity is calculated by `(Average - Actual price)^2` for each sample and add all to get impurity of that leaf.
4. Thus choose a distance < value, such that value gives least residual errors.
5. Compare this way for all other features to choose root.
6. The root's children will become a leaf if there is no further reduction. This will create a leaf for each sample. To avoid this we will set a parameter ex:7 which means if a node cant avg 7 or more samples, it becomes a leaf displaying the avg value which is nothing but output.

Decision trees are powerful but they generally overfit as trees grow too long trying to learn everything about the data. So we prune the tree to avoid overfitting. There are two types of pruning. 
1. Pre-pruning technique involves tuning hyper-paramters like `max_depth`, `max_features`, `min_samples_leaf` and `min_samples_split` before training the model. This type of pre-fitting optimization is called early fitting. 

2. Other technique is called post-pruning. We typically tune only `ccp_aplha` which stands for Cost Complexity Pruning. It is similar to regularization where aplha is similar to lambda. As the tree size increases the loss value increases. More the value of aplha more the decion tree is pruned. 

Decision Trees are scale invariant, visually interpretable, and gives feature importance as these are decisions made on features.

### Entropy
To decide which feature to consider for root node, we use Gini impurity. Alternatively we can even use entropy. It is a measure of the impurity or uncertainty in a set of data. It is calculated as $Entropy = -\Sigma p_i log_2(p_i) bits$. p_i represents the proportion of samples belonging to the i-th class or category. Suppose we have a dataset with 100 samples, and the target variable has two classes: "A" and "B". Let's assume the distribution of classes is as follows:

Class "A": 60 samples
Class "B": 40 samples

To calculate the entropy, we first determine the proportion of samples belonging to each class:

p_A = 60 / 100 = 0.6
p_B = 40 / 100 = 0.4

Next, we substitute these proportions into the entropy formula:

$$ Entropy = - (p_A * log_2(p_A) + p_B * log_2(p_B)) $$

It comes around 1.47 bits. Higher the entropy higher the dataset is balanced. As balanced datasets will have all classes evenly distributed and no class is favored thus high uncertainity.
Similarly gini impurity is calculated without logarithm (thus simple calculations) as $Gini = 1-\Sigma p_i^2$. 
## Ensemble Models
Ensemble methods are combinations of several base estimators. There are two types of ensemble methods:  
1. Averaging:  

They combine several base estimators and then average their predictions to give final predictions. It is like voting with each vote carrying same power i.e each base estimator has equal effect on final output. **Reduces Variance**  

Ex: Bagging (Mix of bagging and any other base estimator), Random Forest (Combination of Decision trees)
    
2. Boosting:  

Base estimators are built sequentially i.e one after another. Present estimator decides the next estimator effect on final predictions. It is also voting but not all votes have equal power. The motivation is to combine several weak models to produce a powerful ensemble. **Reduces Bias**  

Ex: AdaBoost, Gradient Tree Boosting
### Random Forest
Decision trees are almost perfect but they just overfit. Random forest push performance trading explainability and computational performance. Random forest algorithm involves following steps:  
1. Bootstrap the dataset (take random samples from the dataset with repetition and make another dataset with equal size as original).
2.  For each decision tree in the Random Forest, a random subset of features is selected. This subset is typically smaller than the total number of features available in the dataset (usually for classification its sr root of number of features and for regression one third of features).
3. Repeat the step 1 and 2 for n times creating a forest of trees using random features (Hence the name Random Forest).
4. Predict the output of a new sample is done through maximum similar output resulted from n trees (voting). 
### AdaBoost
1. The core principle of AdaBoost is to fit a sequence of weak learners on repeatedly modified versions of the data.  
2. These weak learners are many a times stumps(decision tree with only one node(leaf<--Root-->Leaf). We create forest of these stumps.  
3. Each sample is given a sample weight. Initially sample weight is equal for all sammples (1/No.of.samples).  
4. To create our first stump, out of all features, best feature is choosed as the node(similar to decision tree, Gini Index). This stump is used to predict the output. The amount of say is then calculated for the stump using $0.5\*log((1-TotalError)/Total Error)$.  
5. Training samples that are incorrectly classified will now have increased sample weight and correctly classified will have decreased sample weight. Amount of increase is determined by $New Weight = sample weight * e^{amount of say}$. Similarly decrease is given by `New Weight = sample weight * e^-(amount of say)`.  
6. Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.  
7. Using these new sample weights, repeat above steps again i.e create a stump, predict and calculate amount of say, ...
8. After creating all these stumps, to give the final prediction, each of predictions from a stump is weighted using respective amount of say and then the final prediction is decided.

### Gradient Boosting
In a way, gradient boosting is similar to adaboost where gradient boost uses log loss function (gradient) but the latter uses exponential loss function. Also trees in gradient boosting are not restricted to stumps (usually 8 to 32 leaves). Will have a closer look to understand better.  

In regression:
1. Choose a differentiable loss function. `d/dPred(obs-Pred)^2`
2. Initialize the model with an initial tree (it will be a single value in first case).
3. Our first tree is generally the initial predicted value which will be the avg of the all samples output.
4. Start the loop `for m = 1 to M` where M is the number of trees:  
    1. Calculate residuals using Observed - Predicted where predicted value is the value predicted by previous tree (In first case, it is the value from step 3). Residuals are calculated for all the samples.
    2. Build a decision tree (using gini) to predict the calculated **residuals**. The tree might look like `Residual value 1<--(Feature < value)-->Residual value 2, 3` (Ex: -17.3<--(Height<1.55)-->14.7,2.7) where the residual says how off the predicted value is to original for that sample.
    3. The leaves contain the average residual values if there are multiple samples coming under that decision. (Ex: -17.3<--(Height<1.55)-->8.7)
    4. New prediction for the sample is given by $Previous prediction + learning rate * Avg residual$. (Ex: if the input is height = 1.6 then pred = previous pred + rate * 8.7 (Because height  > 1.55))
    5. Thus new prediction comes closer to the observed value.
    6. If the learning rate is 1, then pred = prev pred + residual which will straight give the observed avalue.
    7. So we generally set is a small learning rate to approach the observed value slowly.
5. Note the above steps are repeated for M trees.
6. After calling all M trees, `Predicted value = Previous Predicted value (from M-1 tree) + learning rate*Residual of M-1 tree` which is nothing but `Predicted value = First prediction (avg of output samples) + learning rate * (residual of tree 1 + residual of tree2 + ... + residual of tree M-1)`.
7. Which means new input goes through all the trees to predict the output.

In classification, the algorithm almost remains same except for loss function. Before going into this, let me provide some context on odds.
#### Odds
Odds represent the ratio of the probability of an event happening to the probability of it not happening. For example, if the odds of winning a game are 3 to 1, it means that the probability of winning is three times higher than the probability of losing. But instead of odds we use log odds (reason explained shortly). These log odds can be shrinked down to 0 to 1 probabilities using sigmoid function. Log odds have symmetric distribution around the zero making calculations easy and they have normal distribution. Lets get back to gradient boosting.

1. Choose a differentiable loss function like logistic loss.
2. Initialize the model with a single tree (it will be a single value in first case).
3. But unlike regression, we can't use average value as the starting prediction. So we use log odds. Say we have 10 samples with output classes of survived (4) and not survived(6), then log(odds) that a passenger survives is log(no.of times survived in all sample outputs/no.of times in not survived in all sample outputs)= log(4/6). This value is our first tree.
4. For classificaion, we convert this log odds to probability by using the sigmoid function, $\frac{e^{log(odds)}}{1+e^{log(odds)}}$. Say we got a value of 0.7
5. We calculate residuals for all samples. For all survived (0), residual will be (0-0.7)=0.3 and for all not survived, residual value will be 0.3
6. Using these residual values we build our second tree with all the features(columns).
9. Start the loop `for m = 1 to M`
    1. Calculate residuals which might be like (Obs - Predicted Probability)
    2. Build a decision tree to predict the residuals similar to regression case.
    3. The residuals are obtained by using probabilities but our first tree is obtained by log odds. So we have to convert the leaves of decision tree (which are the residuals) into log odds. It is given by $\frac{\Sigma Residuals}{\Sigma(PreviousProability*(1-PreviousProbability)}$ for each leaf. Ex: Say we got a value of -0.7,0.3 as the leaves for a node. This is converted as $\frac{-0.7+0.3}{0.7*(1-0.7)+0.7*(1-0.7)}$ The 0.7 in the denominator is the previous predicted probability we got by using sigmoid function. If there is only one value as the leaf for a node, then just remove summations.
    4. Decision tree looks `NewOddsValue <--like Likes Actor-->NewOddsValue`
    5. New prediction for the sample is given by $Previous tree prediction + learning rate * NewOddsValue$. (Ex: Our first tree is log(4/6) so new prediction is log(4/6)+0.8\*1.4)
    5. We convert this value back to probability using sigmoid again to calculate new residuals.
    8. After calling all M trees, `Predicted value = Previous Predicted value (from M-1 tree) + learning rate*Residual of M-1 tree` which is nothing but `Predicted value = First prediction (avg of output samples) + learning rate * (residual of tree 1 + residual of tree2 + ... + residual of tree M-1)`.

The gradient boost regression kind of looks like decision trees built on linear regression and classification looks like decision trees built on logistic regression. So the residuals that are claculated in gradient boost are called **Pseudo Residuals**.

### XG-Boost
Extreme Gradient Boosting trees are perhaps the most used and successful model on Kaggle competetions. Though it is just a variant of gradient boosting, it has a lot more to offer. Most of the below parts in XGboost are adapted from Statquest videos.

#### Regression
1. Similar to gradient boosting, we will start with an initial prediction. Unlike taking averages, we will use a default value of 0.5.
2. Again same as gradient boosting, we will take residuals from the data points to this prediction and will build a tree as the residual values as the leaves. The difference is XGboost is how we build this tree.
3. To choose the root node, we first need to sort the values of the feature. Choose first two data points and find average value and use this avg value to split the residuals. Only one residual will be on left and all remaining residuals to right. To know how well this split works, we calculate a quality/similarity score given by $S.S = \frac{{\Sigma Residuals_i}^2}{Number of Residuals + \lambda}$. Here $\lambda$ is a regularization parameter. Ex: Say we have four data points with a feature like drug dosage 10, 20, 30, 40 with corresponding residuals as {-10.5, 6.5, 7.5 -7.5}. We will use avg value of 10 and 20 which is 15 to split the residual into two groups {-10.5} and {6.5, 7.5 -7.5} and with no regularization, we will have a similarity score of 110.25 and 14.08 respectively. 
4. We will also calculate S.S without any split using all residuals {-10.5, 6.5, 7.5 -7.5} in the S.S formula whihc gives a score of 4.
5. To find how well the split of `dosage<15` works, we calculate gain given by $Gain=LeftLeafSimilarity + RightLeafSimilarity-RootSimilarity$
6. Now we take a new split by taking average of 2nd and 3rd value and repeat the whole process. Now we compare gains of all splits and decide the best split.
7. Say we got `dosage<15` as best split, then we have only one residual value to left and three values to right. We further split the three values to right as we did the before. In general we split upto 6 levels.
8. Pruning in XG boost is based on the gain value (even the $\lambda$ does pruning to an extent). We declare $\gamma$ with some value say 130. If the lowest branch in our tree is less that 130, then we remove that branch and move to the upper branch. We start pruning from lowest branch not from root!
9. Similar to gradient boosting, `New Prediction = Previous tree prediction + learning rate * OutputValue` where $OutputValue=\frac{Sum of Residuals}{Number of Residuals + \lambda}$. In general learning rate is referred as eta in XGBoost.
10. We repeat this for M trees making each new tree with previous residulas.
#### Classification
1. Similar to gradient boosting, we will start with an initial prediction. Unlike taking averages, we will use a default value of 0.5.
2. Again same as gradient boosting classification, we will take residuals from the data points to this prediction and will build a tree as the residual values as the leaves. 
3. To choose the root node, we first need to sort the values of the feature. Choose first two data points and find average value and use this avg value to split the residuals. Only one residual will be on left and all remaining residuals to right. To know how well this split works, we calculate a quality/similarity score given by $S.S = \frac{\Sigma Residuals_i^2}{\Sigma(PreviousProability*(1-PreviousProbability)) + \lambda}$. Here $\lambda$ is a regularization parameter.
4. We will also calculate S.S without any split using all residuals.
5. To find how well the split works, we calculate gain given by $Gain=LeftLeaveSimilarity + RightLeaveSimilarity-RootSimilarity$
6. Now we take a new split and repeat the whole process. Now we compare gains of all splits and decide the best split.
9. $OutputValue=\frac{Sum of Residuals}{\Sigma(PreviousProability*(1-PreviousProbability)) + \lambda}$. Previous tree prediction which will be 0.5 at the start used to calculated output value.
10. After calculating output value, we convert the previous probability to log odds using $log(\frac{p}{1-p})$ and then use `New Prediction = Previous tree prediction + learning rate * OutputValue`.
11. We then convert the output value into probability using logistic function. This value will be the previous prediction value for next tree.
12. We repeat this for M trees making each new tree with previous residulas.

#### Extra Stuff
1. Under the cover of similarity scores, when we do the sum of residuals and squares, we are actually doing gradients of L2 loss function and in the denominator when we say number of residuals it is actually hessian (the second derivative of loss function plus lambda).
2. We see that XGBoost uses greedy algorithm by checking every threshold in every feature. With large datasets, this could be an issue.
3. So XGboost uses approximate greedy algorithm. Instead of using every threshold, it only considers quantiles. For example, if we use one quantile, then we only have one threshold. Default quantiles is about 33.
4. To be precise, XGBoost use quantile sketch algorithm where the large dataset is splitted into multiple sets and having those sets on multiple computers/nodes (parallel computing). The algorithm combines the values from each node and makes an approximate histogram and then approximate quantiles are used on this histograms. These approximate quantiles are weighted quantiles whihc means each data sample has a corresponding weight and each quantile split the data into equal weights. In XGboost weights are given by hessian which means in regression weight is 1 for all data (just normal quantiles) but in classification weight is given by `PreviousProability*(1-PreviousProbability)`. Thus these weights will give extra significance to the mislabeled leaves.
5. Another feature of XGBoost is handling missing values. With missing values, residuals are calculated by splitting the missing values into a seperate table. We split the other table as usual. Then we add the missing data to the left leaf and calculate gain and repeat same with missing data on right leaf and calculate gain. We repeat this for all thresholds and use the best overall gain. For example, if we got best gain with missing data on the left leaf, all new missing input data will go to this leaf.
6. Also XGBoost gradients and hessians on cache memory, making the training fast. 
7. If dataset is too large for cache and main memory, then XGboost compress the data. Uncompressing data by CPU is faster than reading data from hard disk.
8. If there are multiple disks available, XGboost splits data into both disks making reading data faster. Also it can build trees with only some amount of data instead of all data.
9. XGboost can also be trained on GPUs!!

## Which Model to choose? Pros and Cons
### Linear Regression
1. There should be a linear relationship between features and output.
2. Its simple and computationally efficient.
3. Linear Regressions have several assumptions like linearity, homoscedasticity (variance of residuals is constant), independence (all observations are independent of each other), and normality (target values are assumed to be normal but there are **Generalized Linear Models** that can use other distributions on top of linear regression like poisson regression).
### Logistic Regression
1. Ideal for binary classification.
2. Choose when ranking or probabilities of labels are needed.
3. Highly interpretable.
4. Not best in non linear case.
5. Not suitable for highly correlated features.
### Naive Bayes
1. Best for high number of categorical features (like text)
2. Highly scalable due to being naive.
3. Not best if features are correlated or complex.
### SVM
1. Works well if there is a clear margin seperation between classes.
2. Ideal choice when there are more features than samples.
3. Effective in high dimensions (thanks to kernels)
4. Memory Efficient.
5. But not suitable for large datasets.
6. Performs poorly on imbalanced datasets.
### KNN
1. This should be the first choice of model when there are many data samples with few features.
2. Better suited for small to medium sized dataset.
3. It doesn't make any assumptions about the data distribution (non parametric) and can capture complex non linear relationships.
4. It is robust to outliers.
5. KNN is not best for high dimensional data.
6. Not best for imbalanced data where one class dominates thus influencing neighbours more than it should.
### Trees
1. Best for tabular data even with non linear relations.
2. Feature Importance can be interpreted from decision trees
3. Decision trees works best but often overfits.
4. Random forest reduces overfitting caused by decision trees.
5. Gradient boosting methods are the most powerful models but interpretability is not so good.

# Dimensionality Reduction
As the name says it is used for decreasing dimensions/features. Often used for better computation or visualization or interpretation of important features. I will only discuss the unsupervised ways. (Linear Discriminant Analysis is a supervised dimensionality reduction technique)
## PCA
Principal Component Analysis is quite a popular choice for dimensionality reduction. It preserves the most important information or patterns present in the data. It achieves this by finding a set of orthogonal variables, known as principal components, that capture the maximum variance in the original data.

Intuitive Explanation:
1. Say we have two features and few samples. We plot those feature points onto the graph where axis are feature 1 and feature 2.
2. We then find the mean of both features which gives us a point P(mean1, mean2).
3. Shift the data points such that P is at the origin. This doesn't change the relative positions of the points.
4. We will try to fit a line through the points such that it passes through the origin.
5. To find the best-fit line, start with a random line passing through origin. Project the data points onto the line.
6. Goal is to minimize the distances from the data points to the line or maximize the distance from data points from origin. 
7. Either case works because, we draw a perpendicular line from project point to the actual point, we will get a right angled triangle with the data point vector as hypotenuse (say a) which makes other two lines (b, c) are inversely proportional to each other. b,c are described in the above point.
8. It is generally easier to calculate c which is maximize the distance from data points to origin. So PCS calculates all the distances for every projected data point and finds the sum of squared distances (d1^2+d2^2+...).
9. Now we rotate the line and calculate distances again and we repeat this to maxime the distance.
10. The line with maximum value is called Principled Component 1. The slope of the line tells us on which axis the data is more spread/varied. Say we have a slope of 0.25 this means every 4 units we go along feature 1, we go 1 unit along feature2. This is like 4:1 feature mixture (linear combination of features). This 4 and 1 forms a right angled triangle on graph with 4.12 as hypotenuse. Scale theese values such that hypotenuse is 1. This gives us new values but with same ratio. This hypotenuse is now the unit vector that consists 0.97 feature 1 and 0.242 feature 2 is called an eigen vector for PC1. The eigen value is given by $\frac{Sum of squared distances}{Number of data points-1}$ which is nothing but measure of variation. The eigen value shows the amount of say a PC1 has.
11. The PC2 is given by a unit vector perpendicular to PC1. Scree plot is the plot with all the amounts of say for each PC.
12. We remove everything from graph except the PC1 and PC2 lines with projected points on them. Then we rotate the PC1 and PC2 lines such that PC1 is horizontal to the screen for better visualization and map those points back onto the graph.
13. When we have more than 2 features, we do the same thing as above, but instead of just using perpendicular PC2 we try all perpendicular lines to PC1 as there will  be more than 1 in 3D. And we just use the unit vector perpendicular to both PC1 and PC2 for PC3.

Actual Mathematical Implementation:

Before we go into dirty stuff, will get into basics first.

### Eigen Values and Vectors
An eigenvector of a square matrix is a non-zero vector that, when multiplied by the matrix, only changes by a scalar factor. In other words, if we have a matrix A and a vector v, the eigenvector v remains in the same direction but may be scaled by a scalar λ (the eigenvalue) when multiplied by matrix A: $Av=\lambda v$ Each eigenvector has an eigenvalue. The eigenvectors with distinct eigenvalues are linearly independent and orthogonal. Eigenvectors and values are quite useful in finding linear transformations. 

To calculate eigenvalues and vectors for a matrix A, we solve $|A-\lambda I| = 0$ This gives us eigenvalues. We use these values in $Av=\lambda v$ to get eigenvectors. 
### Lagrange Multipliers
We have a formula for say loss function or success rate or some optimization as $u^TSu$ where S is some matrix. Say we need to maximize this function but we have a constraint that $u^Tu=1$ (In place of one we can have any value since we are not interested in magnitude here). To solve such problems we use lagrange multipliers. 

The basic idea behind Lagrange multipliers is to incorporate the constraints into the optimization problem by introducing additional variables called Lagrange multipliers ($\lambda$. These multipliers are used to construct a new function called the Lagrangian, which combines the objective function (the function to be optimized) and the constraints. 

So we update the function as $u^TSu + \lambda (1-u^Tu)$. Now this function even has constraint involved. So to find maximize we take derivate after equating to zero. W.K.T derivate of $x^TAx = 2Ax$ so when we differentiate the whole function w.r.t u, we get $2Su-\lambda 2u=0$. Rewriting this we get $Su=\lambda u$ which is our eigenvector equation. Multiplying with $u^T$ on both sides we get $u^TSu=\lambda$ which means the biggest eigenvalue is what we need to maximize our function with constraint.

## PCA
Back to PCA, here are the steps we follow:
1. Standardize the data first. When the ranges are different then the range of variance will also change making one feature dominant over another. So standardize ((value-mean)/std) the data before doing any transformations.
2. We will have the data points in the feature space (similar to PCA discussed above). We need to figure out a vector $u_1$ (best fitting line) such that when we project all the data points onto that vector with maximum variance preserved.
3. How do we find that vector? Will start with how projections look. Projection of $x_i$ (data point/vector) onto a vector $u_1$ is given by $u_1^Tx_iu$ and the mean of all projections is given by $u_1^T\bar{X} u$.
4. We need to find the variance of these projections which is given by $\frac{1}{N} \Sigma_{n=1}^N (u_1^TX_n - u_1^T\bar{X})^2$ (remember variance of a point is square of the difference between point and mean divided by number of points, here length of each projected vector is given by $u_1^Tx$). when we expand this, we get $u_1^T\[\frac{1}{N} \Sigma_{n=1}^N (X_n - \bar{X})(X_n - \bar{X})^T]u_1$. This can be rewritten as $u_1^TSu_1$ where S is the covariance matrix.
5. We need to maximize this equation such that $u_1$ is a unit vector because our projected vector is trying to get the direction not the magnitude (so we can use any magnitude). Reframing the constraint as $u_1^Tu_1=1$ as it is a unit vector.
6. We saw the same optimization problem in lagrange multipliers. We need to find $u_1$ which is the **eigenvector of covariance matrix S with the maximum eigenvalue, $\lambda$**
7. So we just need to find all eigenvalues of covariance matrix S and find the biggest eigenvalue. The corresponding eigenvector of this max eigenvalue is our required vector (PC1).
8. We got our first eigenvector/principal component. If we need more dimensions, then find the second largest eigenvalue and its eigenvector which will be our PC2. Since both eigenvectors are linearly independent, there won't be any redundant information.

To summarize the steps:
1. Standardize the data.
2. Find the covariance matrix. The covariance matrix will be of dimensions p x p where p is the number of features.
3. Find eigenvalues and eigenvectors of the covariance matrix and sort the eigenvectors in descending order of their eigenvalues to get principal components.
### Watchouts
1. How many principal components to consider? In case there is no definite goal of reaching 2D or 3D, then it's better to plot the scree plot which has eigenvalues on the y-axis and principal component number of the X-axis. As we expect the plot decreases with increasing PCs. Generally, people stop with the PC when the plot stops falling rapidly.
2. PCA assumes there is a linear correlation among features. If the features are already independent or not linear (plot the data points to know if not linear) then PCA is of no use.
3. There is another version called kernel PCA which can be used for non-linearly related features. Similar to SVM, the data is transformed into higher dimensions and then covariance matrix is computed. But this doesn't always work. So many use tSNE for non linear relations.
4. Also mentioned before, PCA is scale variant so we standardize the data first.
5. Interpretability is a key issue in PCA as Principal components say nothing about the data.

## t-SNE
t-distributed Stochastic Neighbour Embedding is one of the techniques in Manifold Learning (nonlinear dimensionality reduction). Unlike linear techniques like PCA, t-SNE is based on a nonlinear approach that focuses on preserving the pairwise similarities or distances between data points. It is particularly effective at revealing the local structure and clusters in the data, making it well-suited for visualizing complex datasets.

The core idea behind t-SNE is to construct a probability distribution that measures the similarity between pairs of high-dimensional data points. This is done by modeling the probability of a point choosing another point as its neighbor based on their similarities. The similarity is calculated using a Gaussian kernel, which assigns higher values to points that are closer in the original space.

Next, t-SNE constructs a similar probability distribution in the low-dimensional space, where the embedded points will reside. It aims to minimize the Kullback-Leibler (KL) divergence (it is similar to cross-entropy loss) between the two distributions, ensuring that points with high similarities in the original space are mapped to nearby locations in the low-dimensional space.

The algorithm starts by randomly placing the data points in the low-dimensional space and iteratively adjusts their positions to minimize the KL divergence. It uses gradient descent optimization to update the positions based on the differences between the similarities in the high-dimensional and low-dimensional spaces. In each iteration, t-SNE calculates the gradient of the KL divergence and adjusts the positions of the points accordingly.

During the optimization process, t-SNE places more emphasis on preserving the local structure rather than the global structure. This means that nearby points in the original space will have a stronger influence on the resulting embedding than distant points. Consequently, t-SNE is effective at revealing clusters and identifying similar points that are grouped together.

It's important to note that t-SNE does not preserve the exact distances between points from the high-dimensional space. Therefore, the resulting visualization should be interpreted in terms of relative distances and relationships rather than exact magnitudes. Additionally, the algorithm's performance can be sensitive to the choice of parameters, such as the perplexity, which determines the balance between preserving local and global structure.

### In detail
1. The main idea is to cluster nearby neighbors together. But how do we know the nearby data points? 
2. Say we want to use euclidean distance $|x_i-x_j|$ to know the distance between two points. Now we need to create a probability distribution that represents the similarity between points. This can be done by using conditional probability $P_{j|i}$ i.e probability of picking a data point $x_j$ as a neighbor given $x_i$. To map it to a distribution we choose gaussian. So we take **conditional probability of point $x_j$ to be next to point $x_i$ which is represented by a Gaussian distribution centered at $x_i$ with a standard deviation $\sigma _i$**
3. Geometrically, we take our point $x_i$ and put it as mean (center) with some $\sigma$(will explain shortly) and map all other points on that distribution.

![](Images/tsne.png)

4. But this has an issue. If the similar data points are quite close then the distribution we got will separate the clusters well. But if the data points are spread, then clusters will also spread (probability is decreased). To tackle this, we **scale all the values to have a sum equal to 1** i.e we divide the probability by the sum of all conditional probabilities of all points given our $x_i$.
5. Coming to standard deviation of our distribution, it is chosen by a parameter called **perplexity**. It is a number that determines number of neighbors similar to k in knn. With the value we set for perplexity, tSNE calculates $\sigma$(ignore formula needed) such that only the set number of neighbors are in our distribution.
6. The combined formula to find the similarity looks like

$$ p_{j|i} = \frac{exp(-||x_i-x_j||^2/2{\sigma_i}^2)}{\Sigma_{k\neq i} exp(-||x_i-x_k||^2/2{\sigma_i}^2)} $$

7. We calculate these similarities for all data points. But we look carefully, $p_{j|i}$ should be equal to $p_{i|j}$ since the similarity value between two points shouldn't change from whichever point we calculate. But gaussian distributions change as the point of focus is set as mean thus the final values will be different. So we will take joint probability distribution as final similarity value $p_{ij} = \frac{p_{j|i}+p_{i|j}}{2N}$ where N is number of dimensions/features.
8. Uff we are done with calculating similarity values for higher dimensional data. Now we have to go to lower dimensions.
9. Lower dimension is constructed by placing our datapoints randomly in a lower dimension subspace ((usually in 2D or 3D). Our goal is to optimize the placement of data points such that they form clusters using the similarity values calculated in higher dimensions.
10. First we will calculate the similarity value of these randomly placed data points. It is calculated same way as higher dimensions but we use student t distribution (hence name **t**SNE) as it has longer tails. This was data points won't be cluttered too close. The new formula for lower dimensions looks like:

$$ q_{i|j} = \frac{(1+||y_i-y_j||^2) ^{-1}}{\Sigma_{k\neq i} (1+||y_k-y_l||^2) ^{-1}} $$

11. Since its optimization problem, we have a loss function here called KL divergence that optimizes clusters by minimizing $D_{KL} = \Sigma_i \Sigma_j p_{ij} log \frac{p_{ij}}{q_{ij}}$
12. The above loss function optimizes in such a way that if any two data points have high p value (points close in high dimensions) but low q value (points not close in lower dimensions), then the model is penalized more thus concentrating more on local structure.  

# Unsupervied Learning
The most ideal way of doing machine learning is to train a model without labeling the data. This means the model should figure out the labels on its own. This is called Unsupervised learning. In general, supervised learning works better than unsupervised but the latter is quite useful in dimensionality reduction.
## K means Clustering
It is used for partitioning a dataset into distinct groups or clusters based on their similarities. It aims to minimize the intra-cluster variance while maximizing the inter-cluster variance. The algorithm is quite simple.
1. Select the number of clusters, K, that you want to create.
2. Initialize K points called centroids randomly or using a predefined method. These centroids represent the center of each cluster. The data points on a graph look like (sample1'sfeature1 value, sample1'sfeature2 value,..)
3. For each data point, calculate the distance (e.g., Euclidean distance) to each centroid.
4. Assign the data point to the cluster with the nearest centroid. This step is often referred to as the "nearest neighbor" assignment.
5. After assigning all data points to clusters, calculate the new centroids for each cluster.
6. The new centroid is computed as the mean of all data points assigned to that cluster. It represents the center of gravity for the cluster.
7. Repeat again from step 3 until convergence is achieved. Convergence occurs when the centroids no longer change significantly or when a maximum number of iterations is reached.
8. The algorithm outputs the final K clusters, where each cluster contains a group of data points that are similar to each other in terms of distance.

K-means is simple and fast. But knowing the k value beforehand is a bummer. Also, as it is distance based, it is affected by outliers. It also can't handle the non-spherical clusters due to using the sum of squared errors for optimizing the distance. 

## Hierarchical Clustering
If you are annoyed with knowing the number of clusters beforehand as in k means, then you can use hierarchical clustering.  It is called "hierarchical" because it creates a hierarchy or tree-like structure of clusters. In this approach, the data points are progressively grouped together based on their similarity, starting from individual points and gradually forming larger clusters.

The main idea behind hierarchical clustering is to determine the proximity between data points and use it to build a hierarchy of clusters. There are two main types of hierarchical clustering algorithms: agglomerative and divisive.

Agglomerative clustering is the most commonly used type of hierarchical clustering. It starts by considering each data point as an individual cluster and then merges the closest clusters iteratively until all points are in a single cluster.
The steps involved in agglomerative clustering are as follows:

1. Calculate the proximity or distance between each pair of data points. The choice of distance measure depends on the nature of the data and the problem at hand. Common distance measures include Euclidean distance, Manhattan distance, and cosine similarity.
2. Treat each data point as a separate cluster.
3. Identify the two closest clusters based on the distance measure. These clusters can be individual data points or clusters formed in the previous iterations.
4. Merge the two closest clusters into a single cluster.
5. Recalculate the proximity matrix or distance matrix to reflect the distances between the newly formed cluster and the remaining clusters.
6. Repeat steps 3 to 5 until all data points are merged into a single cluster or until a stopping criterion is met (e.g., a predefined number of clusters or a threshold distance).
7. The output of agglomerative clustering is a tree-like structure called a dendrogram, which represents the hierarchy of clusters. The dendrogram can be visualized to determine the number of clusters at different levels of similarity.

![](Images/hc.png)

In the dendrogram, the height of the vertical lines represents the relative distance between the clusters. To figure out the number of clusters we need, draw a horizontal line over the dendrogram. The number of vertical lines passing through these horizontal lines are the clusters. To figure out where to draw the horizontal line is to see how freely the line can move up and down without passing new vertical lines. 

Divisive clustering is just the opposite. We start with the entire data and will break it down into individual points. This is more computationally expensive. In general, hierarchical clustering has high time and space complexity so not suitable for large data.

## DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm used to discover clusters of arbitrary shape in a dataset. DBSCAN does not require a predefined number of clusters. Instead, it identifies clusters based on the density of data points in the feature space.

Here's how DBSCAN works:

Density-based notion: DBSCAN defines clusters based on the density of data points. It assumes that clusters are regions of high density separated by regions of low density. The algorithm requires two parameters: eps/epsilon (ε), which defines the radius around each data point, and minPts, which specifies the minimum number of points within ε to form a dense region. Based on these two parameters, DBSCAN classify a data point into one of below three:

* Core points: DBSCAN identifies core points as data points that have at least minPts neighboring points within distance ε. These points are considered to be at the core of a cluster.

* Border points: Border points are data points that have fewer neighbors than minPts within ε, but they are within the ε-neighborhood of a core point. These points are part of a cluster but are not considered as core points themselves.

* Noise points: Noise points are data points that are neither core points nor border points. They are isolated points that do not belong to any cluster.

![](Images/dbscan.png)

In this case, minPts is 4. Red points are core points because there are at least 4 points within their surrounding area with radius eps. This area is shown with the circles in the figure. The yellow points are border points because they are within a radius eps from a core point and have less than 4 points within their neighborhood. The points B and C have two points (including the point itself) within their neighborhood (i.e. the surrounding area with a radius of eps). Finally N is an outlier because it is not a core point and cannot be reached from a core point.

Algorithm steps:
1. Randomly select an unvisited data point from the dataset.
2. If the selected point is a core point, a new cluster is created. All points that are density-reachable from this core point (i.e., within ε distance) are added to the cluster. This process continues until no more density-reachable points can be found.
3. If the selected point is a border point, it is assigned to the cluster of its corresponding core point.
4. If the selected point is a noise point, it is disregarded.
5. Repeat these steps until all data points have been visited.
6. Resulting clusters: At the end of the algorithm, each data point is assigned to one of the clusters or labeled as noise. The clusters are formed based on the density-connected components in the dataset.

![](Images/dbscan.gif)

DBSCAN is best used when clusters are of different sizes and is more computationally efficient than hierarchical clustering and can handle outliers. But setting the parameters epsilon and minPts is tricky in a few cases. 
## Gaussian Mixture Models
Often used as an alternative to K means because K means is a deterministic algorithm, which means there is no randomness involved, a point belongs to one cluster always. GMMs are a probabilistic approach that its more of soft clustering meaning a data point might belong to one or more clusters involving randomness. A Gaussian Mixture is a function that is comprised of several Gaussians, each identified by k ∈ {1,…, K}, where K is the number of clusters of our dataset. Each Gaussian k in the mixture is comprised of the following parameters:

* A mean $\mu$ that defines its centre.
* A covariance matrix $\Sigma$ that defines its width.
* A mixing probability $\pi$ that defines how big or small the Gaussian function will be and $\Sigma_{k=1}^K \pi_k = 1$.

![](Images/gmm.png)

Say we have two classes/clusters in our data, Dogs (D) and Cats (C). Each of these cluster distributions can be represented by Gaussian distribution. In general, a gaussian distribution is given by mean and variance. A multivariate gaussian distribution is given by mean and covariance matrix(tells us the shape of distribution). So Dogs distribution is: $N(x| \mu_D, \Sigma_D)$ and Cats: $N(x| \mu_C, \Sigma_C)$. x is our feature matrix. If there is a new data point that is to be assigned a cluster, our GMM will try to maximize $P(x) = \pi_D N(x| \mu_D, \Sigma_D) + \pi_C N(x| \mu_C, \Sigma_C)$ by optimizing the $[\mu, \Sigma, \pi]$. Here the $\pi$(probability of being in a class) determines the amount of influence a distribution will have on a new point, so they are called mixing coefficients.

The optimization of these parameters is done by the Expectation-Maximization (EM) algorithm. It starts with randomly initializing all three parameters and alternates between the expectation step (E-step) and the maximization step (M-step):

E-step: In this step, the algorithm computes the probability that each data point belongs to each component, given the current estimates of the parameters. This is done using Bayes' theorem and is known as the responsibility or posterior probability.

M-step: In this step, the algorithm updates the parameters of the Gaussian components and the mixing coefficients based on the computed responsibilities. The means and covariances are updated using weighted maximum likelihood estimation, where the weights are the responsibilities, and the mixing coefficients are updated based on the average responsibility of each component across all data points.

The E-step and M-step are iteratively performed until convergence, where the parameters no longer significantly change. I left the mathematics part as its not too important unless you want to reimplement it yourself.

# References
The information is pulled from various sources from internet. Major sources are:
1. [CSE 546 University of Washington Autumn 22](https://courses.cs.washington.edu/courses/cse446/22au/schedule/)
1. [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
2. [Python Data Science Handbook by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/)
3. [The Hundred-Page Machine Learning Book](https://themlbook.com/)
4. [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
5. [Ace the Data Science Interview](https://www.acethedatascienceinterview.com/)
6. ChatGPT :)

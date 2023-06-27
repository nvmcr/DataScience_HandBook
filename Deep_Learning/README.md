# Deep Learning
Deep learning algorithms are inspired by the structure and function of the human brain, allowing machines to learn from vast amounts of data, recognize patterns, and make accurate predictions. From image and speech recognition to natural language processing and autonomous vehicles, deep learning is everywhere these days. I will try to cover the most talked about topics in this chapter.
# Loss Function
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
# Optimizer
Using the loss function we get to know how our model is working. But to make our loss less or make our model work well we have to optimize our weights. Optimizers are the functions used to optimize our weights. We already discussed gradient descent(GD) and its variants in machine learning chapter. So we will discussed few advanced optimizers here.

## GD Pitfalls
Not all convex functions are good for GD. There are non-smooth functions that have sharp corners where function is not differentiable.
> The derivative of a function at a point is defined as the slope of the tangent line to the function at that point.

![GD Pitfalls](Images/GD1.png)

A subgradient is the gradient at a point on the function. At the sharp corners, there can be multiple tangents leading to multiple subgradients at a single point. This makes convergence go crazy.

Other two main challenges with Gradient Descent are local minima's, saddle points and plateau. 

![GD Pitfalls](Images/GD2.png)

If the random initialization starts the algorithm on the left, then it will converge to a local minimum, which is not as good as the global minimum. If it starts on the right, then it will take a very long time to cross the plateau (flat regions), and if you stop too early you will never reach the global minimum. 


Saddle points are not local minima. They are the points where in one direction loss goes up and in other loss goes down making gradient at the point zero. But the point is not local or global minima. Our GD can't get out of saddle point regions.

Most of these pitfalls are common for deep learning models as we have large number of dimensions involved in optimization. So most of the below discussed optimizations are used in deep learning.
## SGD with Momentum
This is a technique used to accelerate the convergence of the optimization process by adding a fraction of the previous gradient to the current gradient estimate. The idea is to allow the optimization algorithm to build up speed in directions that have consistent gradients and to dampen oscillations in directions with inconsistent gradients thus passing local minima with velocity gathered from rolling down the hill. Similarly crossing saddle points and plateaus (flat regions).

Specifically, at each iteration of the SGD algorithm with momentum, the gradient is computed on a small subset of the training data (a mini-batch), and then a "velocity" vector is updated by adding a fraction (the momentum coefficient or $\rho$) of the previous velocity vector to the current gradient estimate. The parameters of the model are then updated by subtracting the updated velocity vector from the current parameter estimate.

The momentum coefficient is typically set to a value between 0 and 1, usually 0.9 with higher values leading to more momentum and faster convergence, but potentially overshooting the optimal solution.

for t=1,...T do

  for i=1,...n do

$v \leftarrow = \rho v + (y^i - w^Tx^i)x^i$

$w \leftarrow w + \alpha v$

return w

But, we observe we have two hyperparameters now, $\rho$ and $\alpha$. They need tuning to get the best results.

Nesterov momentum is an extension of the standard momentum optimization algorithm. It improves upon traditional momentum by taking into account the expected future position of the parameters when computing the gradient update. This helps to accelerate convergence and enhance the optimization process.

Standard momentum update involves two steps: 1) calculating the momentum term by accumulating a fraction of the previous update, and 2) updating the parameters based on this momentum term. Nesterov momentum modifies the second step to incorporate a look-ahead update, which allows the algorithm to "look ahead" and make a more informed gradient update.
## AdaGrad
AdaGrad, which stands for Adaptive Gradient, is an optimization algorithm that is designed to automatically adapt the learning rate during training. 

During training, AdaGrad maintains a set of per-parameter gradient accumulators, which are initialized to zero. At each iteration, the gradient of the loss function with respect to the parameters is computed, and the accumulated sum of the squared gradients for each parameter is updated as follows: $grad accumulator += gradient^2$. The parameter is calculated as:

$$ parameter -= \frac{initial learning rate}{\sqrt{grad accumulator} + offset to avoid zero}* gradient $$

The negative sign is turned into positive by the gradient (negative gradient means going down the slope). The initial learning rate is generally set to 0.1 or 0.01 but it doesn't matter because it is adaptively scaled. Adagrad even address issue in higher dimensional space where one parameter converges faster than the other.

In Adagrad, as the gradient decreases, the step size keeps decreasing as we approach convergence which is useful in convex but if there is a saddle point or flat, then the optimization becomes super slow.
## RMSProp
This is an updated version of Adagrad.

$$ grad acc = decay rate * grad acc + (1 - decay rate) * gradient^2 $$

$$ parameter -= \frac{initial learning rate}{\sqrt{grad accumulator} + offset to avoid zero}* gradient $$

The first line of the update rule computes the moving average of the squared gradients using an exponential decay, which is a hyper-parameter usually set to 0.9. This effectively scales down the learning rate for parameters with large gradients and scales up the learning rate for parameters with small gradients. Thus giving us smoother convergence.
## Adam
Perhaps the most used optimizer for neural networks. Adam, short for Adaptive Moment Estimation, is a popular optimization algorithm in machine learning that combines ideas from both RMSprop and momentum-based gradient descent. It is an adaptive learning rate optimization algorithm, which means it adjusts the learning rate of each parameter based on the history of the gradients for that parameter. 

The Adam algorithm maintains a set of exponentially decaying average of past gradients and past squared gradients for each parameter. The decaying rate of the averages is controlled by two hyperparameters, $\beta_1$ and $\beta_2$, which are typically set to 0.9 and 0.999, respectively. The average of past gradients is used to calculate the momentum term, while the average of past squared gradients is used to calculate the scaling term. These two terms are then combined to obtain the update for each parameter.

Say the computed gradient of loss function be $dx$ and first and second moment are set to zero initially.

$$ first moment = \beta_1\*first moment + (1-\beta_1)\*dx $$

$$ second moment = \beta_2\*second moment + (1-\beta_2)\*dx\*dx $$

$$ parameter -= \frac{\alpha*first moment}{\sqrt{second moment} + offset} $$

Adam also includes a bias correction mechanism that corrects for the fact that the estimates of the first and second moments are biased towards zero, especially in the early stages of training when the estimates are very inaccurate.

*Adam with weight decay (L2 regularization) should be the default choice for most of the problems.*
# Neural Networks
Let's consider a supervised image classification. Using a linear classifier is not enough for images as they don't capture more than one template. It's important that we use a nonlinear classifier. For many years in the field of computer vision, people tried different approaches to extract/learn features of images using techniques like Histogram of Oriented Gradients (HOG), Bag of Words, etc. But most of them are not effective except neural networks. 

Neural networks are a bunch of linear classifiers with activation functions in between. Activation functions are nonlinear functions that add nonlinearity and help the model learn complex relations. Without these, the neural network is just a linear classifier. 

In a neural network, there is an input layer, an output layer, and any number of hidden layers in between. Each hidden layer consists of neurons. We try to learn the weights/parameters of these neurons. See the below image to know what happens in a fully connected neural network/multi-layer perception. 

![](Images/fcn.png)

Each input is multiplied (dot product) by the weight of each neuron. This dot product is made nonlinear by passing through the activation function. This output of one hidden layer is passed as the input to the next hidden layer. Each hidden layer will have a bias term just as we have in any other model. (Bias term is to have the offset/intercept as not every function is symmetrical around the origin). The more neurons and layers the more powerful the network is but beware of overfitting and use regularization effectively. 
## BackPropagation
In order to optimize, if we use gradient descent for neural networks, then we have to derive the gradient for weight matrices of each layer, and computing gradients become too complex. The better idea is to use the technique called backpropagation. Watch the [video](https://youtu.be/Ilg3gGewQ5U) by 3Blue1Brown for a visual understanding of backpropagation. Let's see the mathematical implementation of backpropagation.

![](Images/bp.png)

Here we have a fully connected 3-layer neural network with layers represented with $a^{layer_num}$, input later being $a^1$, the last hidden layer is $a^l$ and the output layer is $a^{l+1} and weight matrix of layer $l$ are represented as $\theta^{l-1}$. The weight and biases are randomly initialized. The input $x$ is a single sample image from our dataset.

In the forward pass,

$$ a^1 = x $$

$$ z^2 = \theta^1 a^1 $$ 

$$ a^2 = g(z^2) $$ 

$$ a^l = g(z^l) $$

$$ z^{l+1} = \theta^l a^l $$

$$ a^{l+1} = g(z^{l+1}) = \hat{y}$$

where $g(z) = \frac{1}{1+e^{-z}} $$ which is a sigmoid activation function. 

Backpropagation:

We will chose loss function as a categorical cross entropy $L(y,\hat{y}) = ylog(\hat{y}) + (1-y)log(1-\hat{y})$. We need to optimize the weights and biases. For now, I am removing bias terms as it is handled in the same way as weights. To know in which direction the weights should move, we do gradients. So in the backward pass, we have to find $\frac{\partial L(y,\hat{y})}{\partial \theta_{i,j}^l}$. We start computing gradients of the last layer first and then keep moving back layers. This is an efficient way as we can use the computed gradients of this layer in the previous layer. We will see an example of this. Say we need to compute how the loss changes based on the last hidden layer,

$$ \frac{\partial L}{\partial \theta^{l-1}} $$

(Note: I am taking the entire weight matrix here but it is actually computed for each neuron of the layer, $\theta_1^{l-1}$) 

This is where the chain rule helps. The chain rule says that we can calculate the effect of changing something can be expressed as multiplying changes done along the way. So we can break down the derivative as inner functions using chain rule. As $z^l = \theta^{l-1} a^{l-1}$, changing the weight will also change $z^l$. Again $a^l = g(z^l)$ so changing $z^l$ will change $a^l$. Thus

$$ \frac{\partial L}{\partial \theta^{l-1}} = \frac{\partial z^l}{partial \theta^{l-1}} \frac{\partial a^l}{\partial z^l} \frac{\partial L}{\partial a^l} $$

$$ = a^{l-1} g'(z^l) L'(a^l) $$

Not doing explicit calculations as that is not our focus here. We do similarly for the bias term. The result of the bias term is almost the same as above except for the first inner function after using chain rule. So we have the gradient equations for last layer weight and biases so that we can use them in gradient descent to find optimal values. Now we go to the previous layer. Here we need to calculate,

$$ \frac{\partial L}{\partial \theta^{l-2}} = \frac{\partial z^{l-1}}{\partial \theta^{l-2}} \frac{\partial a^{l-1}}{\partial z^{l-1}} \frac{\partial z^l}{\partial a^{l-1}} \frac{\partial a^l}{\partial z^l} \frac{\partial L}{\partial a^l} $$

We can see that to calculate the gradients for this l-1 layer, we need gradients from the last layer too which we already calculated (so we start from back instead of first layer). Thus we move backward calculating each layer gradients till we reach the first layer and do gradient descent to get the updated values for weights and biases for the next forward pass.

When using PyTorch or TensorFlow all these local gradients (gradients within the hidden layers) are calculated automatically using auto differentiation. So we need not explicitly calculate all the gradients. But remember that these gradients occupy space within the memory as there will be thousands of parameters in a neural network. If you are getting CUDA out of memory, it's probably because gradients are occupying the memory. 

## Convolutional Neural Network
FCNs are computationally too expensive as they take the entire image by flattening it (32x32x3) and that was one of the reasons why neural networks are not popular for more than 50 years. But it all changed with CNNs. In an image, we don't need an entire image to predict a class. We only need to know the local information to know the edges. So instead of connecting all nodes/neurons of the previous layer to the next layer, we only take the nearby nodes for the next layers. This can be further decreased to shared weight concept where local connections use same weights. This can be thought as using same weight at any location of the image. Using one weight at the top left cornor and using another weight at bottom of the image doesn't make sense because its the same image so should use same weights. This is implemented by applying same filter at different spatial locations of the image.

![](Images/weights.png)

The filter is convoluted over the image. Convolution is nothing but an element-wise dot product between the kernel and image. It's the same as the linear layer but we share weights/use the same kernel over the entire image.

![](Images/convolution.png)

Say a convolution layer has 10 nodes. This means there are 10 filters of the same size but different values/weights applied to the image. After this, an activation function is applied for each node similar to the FCNs we saw before. ReLU is the most used activation function. Sometimes, there will be a pooling layer in order to reduce the dimensions. Below is an image of max pooling where the output contains only the max value in the region of interest. 

![](Images/maxpool.png)

There could be additional layers. How many layers and how many filters/nodes/neurons per each layer is given by the network architecture. At the end, the image is flattened (make into a 1d array by spreading out each pixel value) in order to input into a softmax function to get the probability values. 

The filters have a few important hyperparameters (settings that need to be manually set by us and can't be learned by the model). Padding is often added to the image for making calculations feasible and preserving the input size. This is mostly done by adding zeros to all sides of the image. Next is kernel size which sets the amount of information retained from the image. A smaller kernel means more information. Mostly 3x3 or 5x5 kernels are used. Last one is the stride. It tells us how to move our kernel both horizontally and vertically across the image.

This is for a single-channel image. If it is an RGB image, then we have 3 channels so we follow a similar approach for each channel and stack them (sum them) up on each other. 

Refer to this [link](https://poloclub.github.io/cnn-explainer/#article-convolution) to visually understand what's happening inside a CNN.

It's important to know how the input size of the image changes at each layer. Say we have an input image of size $W_1* H_1 * C$ with K number of filers of size F and stride S with zero padding P, then the output is $W_2 * H2 * K$ where 

$$ W_2 = \frac{W_1-F+2P}{S} + 1 $$

$$ H_2 = \frac{H_1-F+2P}{S} + 1 $$

and the number of parameters is $F^2CK$ and K biases.
## Training
### Activation Functions
As discussed above activation functions add nonlinearity to the model. Here are the popular choices.

![](Images/af.png)

#### Sigmoid
This can be seen as a function that squashes every input to range [0,1]. Quite popular before 2012. There are several issues with sigmoid. 

1. When the gradients are small at the output layer and as the backpropagation moves backward, it has to go through the sigmoid gate. As we see in the graph, small input to the sigmoid returns zero. If the gradients flowing back is zero, all gradients from that layer to the input layer remain zero and weights will never change thus no learning happens. This result of no convergence due to small gradients is called **vanishing gradients**. Similarly, when the gradients are too large, the sigmoid output will be zero (see the graph) causing a similar problem. This is called **exploding gradients**.
2. Another issue with sigmoid is that, it is not zero-centered. It always gives a positive output. Thus gradients are always all positive or all negative thus not optimizing well.
3. Another minor issue is we need to calculate exponential which can be computationally expensive.
### Tanh
This squashes numbers to range [-1,1] and now its zero centered. But the issue of vanishing and exploding gradients remains the same. Just observe the output graph, when there is a constant region, then the gradients are getting saturated.
### ReLU
The gradients don't saturate in the positive region and are computationally efficient and converge faster than sigmoid and tanh in practice. Thus this is the most common choice of activation functions. But the issue is in the negative region and the issue of vanishing gradients remains the same. Also, it is not zero-centered as the outputs are never negative.
### LeakyReLU
It's the same as ReLU except for the negative region. Here the gradients will never die in both positive and negative regions.
### ELU
Exponential linear unit introduces an exponential function in the negative region. This has all features of ReLU along with zero centered and is more robust to noise in the negative regions than LeakyReLU. 
### GeLU
Gaussian Error Linear Unit is a quite popular choice for transformers. When the input is positive, it is similar to ReLU. In the negative region, we will have a threshold. If the value is below the threshold then they are allowed and other input values are squashed to zero. This means all negative inputs are not allowed thus adding randomness. We try out various thresholds and take **expectation over randomness**.

![](Images/Gelu.png)
### Data Preprocessing
It is often preferred to make the data zero-centered and normalize the data. With normalization, the model is less sensitive to small changes in weights.

![](Images/normalization.png)

For images, in practice, the mean is calculated for each channel of the images, and the mean is subtracted from the respective channel pixels values making each channel zero-centered.
### Weight initialization
While discussing backprop, we said we initial starting weights randomly. What if we used `W=constant init` i.e use same weights for all nodes? If weights are all the same at the start then gradients will also be the same for all nodes. Then no matter how many layers we add it is still same as a single layer.

What if we use random values? We can use `W = 0.01 * np.random.randn(Din, Dout)` the output weight matrix will be Din x Dout dimensions. This works for smaller networks but not for deeper networks(more layers). Since the weights are small, taking gradients will result in a vanishing gradients problem. Then we might use larger weights like `W = 0.05 * np.random.randn(Din, Dout)`. But these weights are too big and result in exploding gradients. Uff, so weight initialization is tricky and important. In practice, we use something called **Xavier** initialization, `W = np.random.randn(Din, Dout)/ np.sqrt(Din)` this solves the gradients issues. Another popular coice is **He** initialization (ResNet), `W = np.random.randn(Din, Dout)/ np.sqrt(2/Din)`
### Batch Normalization
As we have deeper networks, the distribution of each layer changes due to the weight updates. This is called internal covariate shift. When previous layer distributions change, the next layer needs to constantly adapt to these changes thus more time to converge. Batch normalization explicitly makes our data zero means and unit variance in between layers. 

![](Images/batchnorm.png)

$\gamma$ and $\beta$ are learnable parameters to use different standard deviations and biases respectively. Batchnorm is used just like an extra layer usually inserted after convolutional layers and before activation functions. During training, BatchNorm keeps track of two sets of statistics for each batch-normalized layer: the running mean (μ) and the running variance (σ^2). These statistics are computed by accumulating the mean and variance of the activations across mini-batches during training.

During testing, instead of using the statistics from the mini-batch, BatchNorm uses the accumulated running mean and running variance. This can be an issue as the testing doesn't happen as it should instead uses training data statistics.

The testing procedure can be summarized as follows:
1. For each mini-batch or individual example during testing, the input activations of the batch-normalized layer are normalized using the running mean (μ) and running variance (σ^2).
2. The normalized activations are then scaled and shifted using the learned scale (γ) and shift (β) parameters, respectively, which are also accumulated during training.
3. The scaled and shifted activations are passed through the rest of the network for prediction or inference.

Batch normalization makes deep networks much easier to train, improves gradient flow, allows higher learning rates, robust to weight initialization, and also acts as a regularizer. Only issue is different behaviour during train and test.

Layer normalization is another variant where normalization is done across the dimensions instead of batches thus having the same behavior during train and test.

### Learning Rate Schedule
Learning rate is a hyperparameter and choosing a learning rate is quite tricky. Because using a low learning rate increases training time and also isn't helpful to get out of local minima. If we choose a high learning rate, convergence is unstable. The best choice is to use a high learning rate at the start (its fast) and use a small learning rate while reaching the global minima (doesn't overshoot). This is done by learning rate schedular for learning rate decay.

One option we have is to manually input the learning rate at fixed points. For example, multiplying the learning rate by 0.1 after epochs 30, 60, 90 for ResNets. We can generally decide at which epoch by looking at epoch vs loss graph. The epoch were the curve plateaus is where we need to decrease our learning rate. But finding these values for another task is a cumbersome process. One popular schedular is **Cosine** where we fix the number of epochs($T$) and initial learning rate($\alpha_0$) at first. Learning rate an epoch t is given by

  $$ \alpha_t = \frac{1}{2} \alpha_0(1+cos(\frac{t\pi}{T})) $$

This is how the decay looks:

![](Images/cosine.png)

Another popular choice is linear, $\alpha_t = \alpha_0(1-\frac{t}{T})$. 

There is one issue with choosing a high learning rate at first. With high LRs, we might go too far away from global minima and it disturbs all the initializations we carefully crafted. So we use something called **linear warmup** which starts with a very small learning rate for a few iterations or epochs and then moves to a higher learning rate.

> Epochs means the number of times we loop over our entire training data to learn. We do multiple iterations within a single epoch. For example, we have 1000 training samples and a batch size of 100. So in each iteration, our model tries to learn 100 samples. 10 iterations will equal one epoch.
### Dropout
It is a regularization technique where we randomly set some neurons to zero (removing). We have a hyperparameter, the probability of dropping. If it is set to 0.5, then only half of our neurons will be active in a layer.  During test time, since we can't randomly drop neurons, we multiply the output of hidden layers with some probability percentage p to decrease the activations.
# Self-Supervised Learning
Supervised learning is quite expensive due to the manual annotations/labels. Apart from cost, it is more prone to errors in annotations. Even the standard ImageNet has wrong labels. Options outside of supervised learning are semi-supervised, unsupervised, and self-supervised. Semi-supervised learning is to train on unlabeled data along with some labeled data. This is quite popular over a period but is still an active area of research. Getting good results from unsupervised learning is hard as there is no feedback or guidance. That left us with self-supervised learning (SSL). 

> Self-supervised learning aims to learn useful representations or features from unlabeled data without explicit human annotations.

The confusing part is how is it different from unsupervised. In SSL, we do some prediction, $\hat{y}$ which is within the input data itself unlike learning some distribution. There are two steps in SSL.

1. Pretrain a network on a pretext task (will be explained) that doesn't require supervision.
2. Transfer the learned network to a downstream task via linear classifiers, KNN, and finetuning.

Pretext tasks are in general 3 types. Generative pretext tasks predict part of the input signal like Autoencoders, GANs. Discriminative predicts something about the input signal like contrastive learning, rotation+clustering, and multimodal use of additional signals in addition to RGB images like Video, 3D, Audio, etc.

A common example of a pretext task is we give cropped images to the model and make the model predict the complete image or rotate the image and make the model predict the non-rotated image etc. A downstream task could be anything like object detection, image classification etc. Pretext tasks need not be related to downstream tasks. Evaluation of an SSL model is only based on the downstream task. We don't care about pretext task performance.

![](Images/SSL.png)

A detailed example of a pretext task is to predict rotations. From our data, we rotate the image is 4 ways and the prediction is which way the image is rotated. This became a supervised classification problem but no annotation is required. This way we are forcing the model to learn about image representations and use these learned features for some downstream tasks like image classification(1 way).

![](Images/rotation.png)

Our pretext tasks shouldn't be too easy as the model might learn shortcuts/irrelevant features and it also shouldn't be too hard as the model might be too fixated on pretext tasks and wouldn't work on downstream. Other good pretext tasks are inpainting, jigsaw puzzles, colorization, etc. We don't usually know which pretext works well unless we use it for our downstream task. 

### Contrastive Learning
A generalized method for a pretext task would be to use different transformations at the same time. For example, we have a cat and dog image. We then apply different transformations on the cat image and on the dog image. In a latent space (lower dimensional space), the transformed image features of a cat should be closer to each other (attract) and far from the transformed image features of a dog (repel). This method is called contrastive learning.

![](Images/contrastive.png)

Our goal is to $score(f(x),f(x^+)) >> score(f(x), f(x^-))$ i.e our score (similarity score) between reference sample and positive sample should be higher than a score between reference sample and negative sample.

A loss function for contrastive learning given 1 positive sample and N-1 negative samples:

$$ L = -E_X[log \frac{exp(s(f(x),f(x^+)))}{exp(s(f(x),f(x^+))) + \Sigma_{j=1}^{N-1}exp(s(f(x),f(x_{j}^{-})))}] $$

This is called InfoNCE loss (Information Noise Contrastive Estimation) and kind of looks like softmax loss where we have our reference images as the true class and its transformations (positives) as also the true class and every other negative as another class. 
#### SimCLR
A simple framework for Contrastive Learning of visual Representations (SimCLR) was one of the first models to achieve comparable results with supervised learning. 
1. The model starts with a data augmentation model that transforms a data sample image, $x$ into two correlated views by transformations $\hat{x_i}, \hat{x_j}$ like random crop, crop and resize, color distortion etc. Thus our dataset will now be 2N as each image is augmented with two transformations.
2. The transformed images $\hat{x_i}, \hat{x_j}$ are passed through a base encoder, $f$ to extract the image representations, $h_i, h_j$. The paper uses a ResNet50 (the last FCN is removed) to extract the representations.
3. Instead of applying the loss functions on representations directly, representations are passed through a small hidden layer neural network projection head with nonlinearity, $g$ (not entirely clear why but the paper says due to nonlinearity, representations learned will be better with projection head). The goal is to maximize the score for representations that we after projection head, $z_i, z_j$.
4. In practice, we randomly sample a minibatch of N examples and define
the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in 2N data points. We do not sample negative examples explicitly. All other images, 2(N-1) other than two correlated transformed images are given as negative. Similarity score is given by cosine similarity, $sim(u,v) = \frac{u^Tv}{||u|| ||v||}$. This ranges from 1 to -1 where 1 means both vectors, u and v, point to same direction and high similarity. -1 means vectors are pointing opposite and 0 means vectors are uncorrelated.
5. The loss function is same InfoNCE loss which is here modified slightly as

$$ L = -log \frac{exp(sim(z_i,z_j))}{\Sigma_{k=1}^N 1_{[k\neq i]} exp(s(z_i,z_j))} $$

![](Images/simclr.png)

After training the ResNet model using representations, we use this ResNet in some downstream tasks like image classification by adding a linear classification layer at the end. The paper achieved similar results as supervised models on the ImageNet dataset. But with a batch size of over 8000 and with 4 times wider ResNet architecture. 

SimCLR needs a huge batch size in order to train effectively. The paper uses LARS optimizer to optimize for higher batch sizes. **Decoupled Contrastive Learning** (DCL) is a modification of the contrastive learning framework, specifically designed to address some of these batch size limitations and challenges associated with the negative-positive coupling (NPC) effect in the InfoNCE loss.

The issue of the NPC effect in the loss arises from the way negative samples are chosen during training. When we take a batch of images, we select the positive pairs and all the remaining samples are treated as negative. When we have a large batch size, there wouldn't be any issue as there are enough negative samples for the model to learn underlying representations. But with a lower batch size, the negative samples chosen for a positive pair are too easy to distinguish from the positive sample, leading to an overly optimistic loss landscape and poor learning of representations.

This is where Decoupled Contrastive Loss helps. As the name says,  the positive sample is removed from the denominator, resulting in a loss calculation that considers only the negative samples. By excluding the positive sample the loss becomes more focused on the relative similarities among the negative samples. This decoupling reduces the influence of the positive sample, making the loss more informative and less prone to the NPC effect, and also can be used with lower batch sizes.
#### Momentum Contrastive Learning (MoCo)
In MoCo, instead of having every other sample as a negative sample, we keep a running queue of negative examples (called keys) for all of the images in our batch. The batch size is decoupled with the number of keys. For example, we have a running queue of 2000 keys/negative samples. Say our batch size is 1000, we use all the 2000 keys to compute gradients and update the encoder. 

![](Images/moco.png)

We have a similar encoder setup as SimCLR for the query/reference image and get the feature representations (all these will be positives) on the other side we have running keys and then the loss is compared and gradients flow back only through the query encoder (left side). How the negative samples are chosen is simple. For the current batch, all the images from the previous batch are chosen as negative samples (some extra images can also be added or the previous two or three batch sizes can be combined), hence called *running* keys. If there are no gradients passed through the momentum encoder then how are its parameters updated? The momentum encoder is updated similarly to SGD. The key network's parameters are updated using a moving average of the query network's parameters. This update is performed iteratively, where at each step, a fraction of the query network's parameters are mixed with the key network's parameters. This process helps create a more stable and consistent representation space.

MoCo V2 is also released by combining MoCo and SimCLR by using a projection head on the query encoder. 
#### DINO
This is a combination of SimCLR and MoCO but more of a distillation problem. 

![](Images/dino.png)

1. The reference image is transformed into two images. One image is transformed using local augmentations like crop and resize and other image is global augmentation like grayscale, and color distortion (no crop). The local augmented image is given a student encoder model and the global one is given to a teacher encoder model.
2. Both encoder models are vision transformers and as you can see there are no negative samples involved here.
3. Centering is similar to normalizing where dimensions are centered to remove any dominant dimension.
4. The teacher network's parameters are updated using a momentum update mechanism. This involves copying a fraction of the student network's parameters and updating the teacher network's parameters towards the student's parameters. This update process is performed iteratively, and it helps stabilize the training and ensures that the teacher network slowly tracks the student network.
5. The student network is trained to predict the intermediate representations produced by the teacher network. This is achieved by minimizing a contrastive loss, which encourages similar instances to have high similarity scores and dissimilar instances to have low similarity scores.

# Vision and Language
## Recurrent Neural Networks
We are now moving towards language from vision. The main difference between language from vision is the sequences (video is an exception) like given a word/letter what is the next word/letter? RNNs are the starting models for processing sequences. The key idea of RNN is they have an internal hidden state (green box) that is updated as a sequence is processed and it is updated recurrently after every output. 

![](Images/RNN1.png)

The hidden state keeps track of all the previously seen inputs. It is updated after every new input by applying a recurrence formula as follows:

$$ h_t = f_W(h_{t-1}, x_t) $$

The present state is given by some function with weights W (a neural network) which takes in input at the present time step and the hidden old state. Observe here that the weights are shared so it doesn't matter how long the input is. The initial hidden state is usually zeros. In the case of a vanilla RNN, $f_W$ looks like:

$$ h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t) $$

and output is given by $y_t = W_{hy}h_t$. The final loss is given by summing all the individual losses. 

The above case is if we have multiple outputs like the next words. Incase of sentiment analysis we might have just one y based on multiple inputs something like whether the statement is positive or negative. Then the loss is only given by the final y. 

Sometimes there might be multiple outputs but a single input like image captioning (describing what's happening in the image). Then the process remains the same keeping inputs of the next hidden units are zeros or can be the output of the previous input. 

Lets see an example now, say we are training our model to predict the next character, the model looks like:

![](Images/RNN2.png)

As you can see the inputs are given in the form of one hot-encoded vector. Our weight matrix for the looks like this:

![](Images/RNN3.png)

During the test time based on the output score, the next character is given as the input to the next state and so on the word is predicted.

Similar to a normal neural network, we do backpropagation. But it has to load the memory with the weights of every character. This will blow up our memory for long articles. So instead of backpropagating the entire article we take chunks of data at a time and backpropagate and then go for the next chunk in the document.

Image captioning is as simple as combining CNN with RNN and using start and end tokens to say when to stop generating captions.

![](Images/RNN4.png)

Even with all these RNNs are not popular. Because when we think of backpropagation, we know that tanh squashes everything to [-1,1]. When the gradients are flowing back, many values less than 1 keep getting multiplied. By the time gradients from the last hidden state come to the first hidden state the gradients become very very small causing a vanishing gradients problem. This problem is solved by LSTMs.
## Long Short Term Memory 
It is a variant of RNNS. There are many new terms involved here. In RNN we only combine the previous hidden state with the present input using a neural network and passing through a tanh. But in LSTM we have four such activation functions with their own task.
* i: input gate: decides whether to write to cell
* f: Forget gate: decides whether to erase cell
* o: Output gate: decides how much to reveal cell
* g: info gate: decides how much to write to cell

![](Images/lstm.png)

# References
1. [Deep Learning by Ranjay Krishna and Aditya Kusupati](https://courses.cs.washington.edu/courses/cse493g1/23sp/schedule/)
2. [Machine Learning CSE 446 UW](https://courses.cs.washington.edu/courses/cse446/22au/)

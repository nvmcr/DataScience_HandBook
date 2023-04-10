<details>
<summary>Table of Conetents</summary>

## Table of contents
Please use the github's interactive navigation. (Too lazy to write/generate TOC)
![toc](Images/toc.gif)

</details>

## Note
Ability to think probabilistically is important to data scientists. But only few basic concepts are enough for interviews and these concepts will go in depth in the statistics. So the meat of the concepts are in statistics.
## Basics
## Counting
Just basics of permuatations and combinations. If order matters go for $^nP_k$ else $^nC_k$.

$$ ^nP_k = n*(n-1)....*(n-k+1) = {n!\\over (n-k)!} $$

$$ ^nC_k = {n!\\over k!(n-k)!} $$

For example, if you have a set of 5 letters {A, B, C, D, E}, the number of ways you can arrange 3 of these letters in a specific order (i.e., the number of 3-letter permutations) is: $^5P_3$

Similarly, if you have the same set of 5 letters {A, B, C, D, E}, the number of ways you can select 3 of these letters in any order (i.e., the number of 3-letter combinations) is: $^5C_3$
## Conditional Probability
$$ P(A|B) = {P(A,B)\\over P(B)} $$

where $P(A,B)$ is A intersection B. If A and B are independent then $P(A,B)=P(A)P(B)$
## Chain Rule
$$ P(A,B,C,....Z) = P(A)P(B|A)P(C|A,B)P(D|A,B,C,D)...P(Z|A,B,..Y) $$
## Bayes Theorem
One of the most asked questions are based on this. For example: What is the proability of a patient having a disease, given that the patient tested positive for the disease ? The questions generally asks What is the probaility of an event A given that an event B has occurred? 

$$ P(A|B) = {{P(B|A)P(A)}\\over P(B)} $$

$P(A)$ is called Prior, $P(B|A)$ is likelihood and $P(A|B)$ is posterior. 
This can be extended to to a third event case too

$$ P(A|B,C) = {{P(B|A,C)P(A|C)}\\over P(B|C)} $$
## Law of Total Proability
If we have several disjoint events within B having occurred, we can then break down the probability of an event A having also occurred with law of total probability.

$$ P(A) = P(A|B_1)P(B_1) + ... + P(A|B_n)P(B_n) $$

This equation provides a handy way to think about paritioning events. If we want to model the proability of an event A happening, it can be decomposed into weighted sum of conditional probabilities based on each possible scenarios having occurred. If there is a *tree of outcomes*, then this eqn is useful. One example is the probability that a customer makes a purchase, conditional on which segment that customer falls in.
## Bayes 2
Using law of total probability,

$$ P(A|B) = {{P(B|A)P(A)}\\over P(B)} = {{P(B|A)P(A)}\\over {P(B|A)P(A} + P(B|A^C)P(A^C)} $$

# Random Variables
Say there is an outcome space $S$ which represents set of possible outcomes of an experiment and event space $F$ which is a set of subsets of $S$ then a random variable, $X$ is a **map** from $S$ to $R$ where $R$ is real numbers. $X:S->R$ i.e a random variable takes on numerical values based on the outcome of a random experiment or process. In other words, A random variable is a quantity with an associated proability distribution. A probability distribution is a function that describes the **likelihood** of different outcomes in a random experiment or process.

For example, consider rolling a fair six-sided die. The possible outcomes are the numbers 1, 2, 3, 4, 5, and 6. Let X denote the result of rolling the die. X is a random variable, since its value depends on the outcome of the random experiment (i.e., the roll of the die).

Random variables can be classified as either discrete or continuous. A discrete random variable takes on a countable number of distinct values, such as the number of heads obtained when flipping a coin multiple times. A continuous random variable takes on an uncountable number of values within a given range, such as the height of individuals in a population.
## CDF
Cumulative Distribution Function (CDF) is a function that describes the probability that a random variable X is less than or equal to a given value x, for all possible values of x.

The CDF of a random variable X is denoted by $F(x)$ and is defined as:

$$ F(x) = P(X ≤ x) = \int_{-\infty}^x f_X(x)dx $$

where $P(X ≤ x)$ represents the probability that X is less than or equal to x and $f_X(x)$ represents PDF.

For example, suppose we have a fair six-sided die. Let $X$ be the event of a number rolled on the die, and let P(X = x) be the probability of rolling a certain number x. Then, the CDF of X can be calculated as follows:

$$ P(X ≤ 1) = P(X = 1) = 1/6 $$

$$ P(X ≤ 2) = P(X = 1) + P(X = 2) = 1/6 + 1/6 = 1/3 $$

$$ . $$

$$.$$ 

$$P(X ≤ 6) = P(X = 1) + P(X = 2) + P(X = 3) + P(X = 4) + P(X = 5) + P(X = 6) = 1$$

Similarly, let's consider a simple continuous random variable X that is uniformly distributed between 0 and 1. That is, $X ~ U(0,1)$. The PDF of X is:

f(x) = 1 if `0 <= x <= 1`; `0` otherwise

Suppose we want to find the probability that a randomly chosen value of X is less than or equal to 0.5. Then, we can calculate the CDF of X at x = 0.5 as follows:
```math
F(0.5) = P(X ≤ 0.5) = \int_{0}^{0.5} f(x) dx
= \int_{0}^{0.5} 1 dx
= 0.5
```

The CDF has several properties, including:

1. F(x) is a non-decreasing function of x. It is cummulative, just adds up.
2. F(x) approaches 0 as x approaches negative infinity.
3. F(x) approaches 1 as x approaches positive infinity.
4. The probability of X taking on a value between a and b (where a and b are real numbers and a ≤ b) is given by $F(b) - F(a)$.

The CDF can be used to determine various probabilities associated with a random variable. For example, the probability that X lies in a certain interval $\[a, b]$ can be found by taking the difference between the values of F at b and a. Additionally, the CDF can be used to calculate the expected value and variance of a random variable.
## PDF and PMF
A probability mass function (PMF) is a function that describes the probability distribution of a discrete random variable. The PMF gives the probability of each possible value(x) that the random variable can take on. Specifically, the PMF of a discrete random variable X is defined as:

$$ P_X(x) = Pr(X=x) $$

where x is a possible value of X, and Pr(X=x) is the probability that X takes on the value x.

On the other hand, a probability density function (PDF) is a function that describes the probability distribution of a continuous random variable. The PDF gives the relative likelihood that the random variable takes on a specific value, by providing the rate at which the probability changes as the value of the random variable changes. Specifically, the PDF of a continuous random variable X is defined as:

$$ f_X(x) = {dF(x)\\over dx} $$ 

where $F(x)$ is the CDF of X, and $dF(x)\\over dx$ is the derivative of the CDF with respect to x.

Both the PMF and the PDF have the property that the area under the curve equals one. For a PMF, this means that the sum of the probabilities for all possible values of X is equal to one, while for a PDF, this means that the integral of the PDF over all possible values of X is equal to one.
## Quartiles
Using PDF:

The first quartile (Q1) is the value of x for which the area under the PDF to the left of x is 0.25. Similarly, the second quartile (Q2) is the value of x for which the area under the PDF to the left of x is 0.5, which is also the same as the median of the distribution. The third quartile (Q3) is the value of x for which the area under the PDF to the left of x is 0.75.

Using CDF:

The first quartile (Q1) is the value of x for which the CDF is 0.25. Similarly, the second quartile (Q2) is the value of x for which the CDF is 0.5, which is also the same as the median of the distribution. The third quartile (Q3) is the value of x for which the CDF is 0.75.
# Multiple Random Variable
Random variables are often analyzed with respect to other random variables. Below concepts assume continuous r.v, can extend to discrete r.v too.
## Joint Distribution Functions
The joint CDF of two random variables X and Y is a function that gives the probability that both X and Y are less than or equal to certain values x and y, respectively. It is denoted by F(x,y) and defined as:

$$ F(x,y) = P(X ≤ x, Y ≤ y) = \int_{-\infty}^x \int_{-\infty}^y f_{X,Y}(x,y)dxdy $$

where $f_{X,Y}(x,y)$ is the joint PDF.

The same concept can be extended for more than two random variables. Properties of CDF remain the same for joint CDFs too.
## Marginals
From a joint CDF, PDF we can derive marginal CDFs and PDFs respectively. The concept is to marginal w.r.t to one r.v, extend other r.v to infinity. Say we need marginal PDF of X, integrating out Y will give mariginal of X.

$$ f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)dy $$

Now that we have marginal PDF for X, we can calculate its CDF in usual way of integrating.
## Conditional
For two r.vs X, Y; the conditional density function is given by:

$$ f_X(x) = \int_{-\infty}^{\infty} f_Y(y)f_{X|Y}(x|y)dy $$

where 

$$ f_{X|Y}(x|y) = {f_{X,Y}(x,y)\\over f_Y(y)} = {f_{Y|X}(y|x)f_X(x)\\over f_Y(y)} $$

# Expectations
In general central tendency is represented by mean, median and mode. Expectation refers to expected value in this case its mean/average value. Expectation of r.v is given by

$$ E(X) = \int_{-\infty}^{\infty} xf_X(x)dx $$

For discrete

$$ E(X) = \sum_i x_iP(X=x_i) $$
## Variance
It tells us how far is our r.v from its mean i.e $|X-E[X]|$ To know how far a r.v from its mean on *average*:

$$ Var(X) = E\[(X-E[X])^2] = E\[X^2]-(E\[X])^2 $$
## Conditional Expectation
Let X, Y be jointly distributed r.v's. If X is continous (Y could be discrete/cont), then

$$ E\[g(X)|Y=y] = \int_{-\infty}^{\infty} g(x)fX|Y(x|y)dx $$
## MGF
$E(X^2) = \int_{-\infty}^{\infty} x^2 f_X(x)dx$ This is called a moment. The nth moment of X is given by $E[X^n]$. Intuitively, the Moment Generating Function is a way to generate the moments of the distribution of X. The MGF can be used to find all of the moments of X.

$$ M_x(t) = E\[e^tX] $$

![MGF](mgf.png)
# Inequalities
## Markov's Inequality
It provides an upper bound on the probability that a non-negative random variable is greater than or equal to a certain value. Applies to any non-negative random variable.

$$ P(X ≥ a) ≤ {E(X)\\over a} $$
## Chebyshev's Inequality
It provides a bound on the probability that a random variable deviates from its expected value by more than a certain amount(k). The inequality applies to any random variable, regardless of its distribution.

$$ P(|X-E(X)| ≥ k) ≤ {var(X)\\over k^2} $$
# Covariance and Correlation
Both are statistical measures that quantifies the degree to which two variables are linearly related.
## Covariance
$$ Cov(X, Y) = {E\[(X - E(X))(Y - E(Y))]} $$

It measures how much two variables *vary*/change together. Intuitively, the covariance measures the degree to which X and Y tend to vary together. A positive covariance indicates that X and Y tend to increase or decrease together, while a negative covariance indicates that X tends to increase as Y decreases, or vice versa. A covariance of zero indicates that there is no linear relationship between X and Y. It is important to note that covariance is affected by the scales of the variables. If the scales of X and Y are different, the covariance may be difficult to interpret. To address this, the correlation coefficient is often used.
## Correlation
$$ \rho(X, Y) = {Cov(X, Y)\\over \sqrt(var(X) \sqrt(var(Y)} $$

Correlation is a standardized measure of covariance. Intuitively, the correlation measures the strength and direction of the linear relationship between X and Y. A correlation of +1 indicates a perfect positive linear relationship, meaning that when X increases, Y increases proportionally. A correlation of -1 indicates a perfect negative linear relationship, meaning that when X increases, Y decreases proportionally. A correlation of 0 indicates no linear relationship between X and Y. It ranges from -1 to +1.
# Probability Distributions
## Discrete
### Binomial
It gives the probability of *k* number of successes in *n* independent trials where each trail has probability *p* of success. The PMF is given by:

$$ P(X = k) = ^nC_k * p^k * (1 - p)^(n - k) $$

$\mu=np$ and $\sigma^2=np(1-p)$. 
Common applications include number of defective products in a batch, the number of successes in a fixed number of medical trials, or the number of heads in a fixed number of coin flips.
## Poisson
It gives the probability of the number of events occurring wihin a particular fixed interval where the known, constant rate of each event's occurrence is $\lambda$. The PMF is given by:

$$ P(X = k) = {λ^k * e^{(-λ)}\\over k!} $$

$\mu=\sigma^2=\lambda$ 
Common applications include number of customers arriving at a store during a given time period, or the number of defects in a manufacturing process.
## Continuous Distribution
### Uniform
It assumes a constant probability of an $X$ falling between values on the interval a to b. PDF is given by:

$$ f(x) = {1\\over (b-a)} $$

$\mu={a+b\\over 2}$ and $\sigma^2={(b-a)^2\\over 12}$. The most common applications are in sampling and hypothesis testing cases.
### Exponential
It gives the interval length between events of a poisson process having a set rate parameter of $\lambda$. The PDF is given by:

$$ f(x) = \lambda e^{-\lambda x} $$

$\mu={1\\over \lambda}$ and $\sigma^2={1\\over \lambda^2}$
Common applications include time until a customer makes a purchase or the time until a default in credit occurs. 
### Normal
The well-known bell curve.

$$ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \ e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$

# References
The information is pulled from various sources from internet. Major sources are:
2. [UW EE 505 Prabability by Rahul Trivedi](https://sites.google.com/view/ee505uwfall2022/course-material)
3. [UW CSE312 by Alex Tsun](https://courses.cs.washington.edu/courses/cse312/20su/)
4. [Ace the Data Science Interview](https://www.acethedatascienceinterview.com/)
5. ChatGPT :)

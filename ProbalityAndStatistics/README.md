<details>
<summary>Table of Conetents</summary>

## Table of contents
Please use the github's interactive navigation. (Too lazy to write/generate TOC)
![toc](Images/toc.gif)

</details>

\begin{equation}
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\end{equation}

## Note
Ability to think probabilistically is important to data scientists. But only few basic concepts are enough for interviews and these concepts will go in depth in the statistics. So the meat of the concepts are in statistics.
# Probaility
## Conditional Probability
One of the most asked questions are based on this. For example: What is the proability of a patient having a disease, given that the patient tested positive for the disease ? The questions generally asks What is the probaility of an event A given that an event B has occurred? 

$$ P(A|B) = {{P(B|A)P(A)}\\over P(B)} $$

$P(A)$ is called Prior, $P(B|A)$ is likelihood and $P(A|B)$ is posterior.
## Law of Total Proability
If we have several disjoint events within B having occurred, we can then break down the probability of an event A having also occurred with law of total probability.

$$ P(A) = P(A|B_1)P(B_1) + ... + P(A|B_n)P(B_n) $$

This equation provides a handy way to think about paritioning events. If we want to model the proability of an event A happening, it can be decomposed into weighted sum of conditional probabilities based on each possible scenarios having occurred. If there is a *tree of outcomes*, then this eqn is useful. One example is the probability that a customer makes a purchase, conditional on which segment that customer falls in.
## Counting
Just basics of permuatations and combinations. If order matters go for $^nP_r$ else $^nC_r$.

$$ ^nP_r = n*(n-1)....*(n-r+1) = {n!\\over (n-r)!} $$

$$ ^nC_r = {n!\\over k!(n-r)!} $$

For example, if you have a set of 5 letters {A, B, C, D, E}, the number of ways you can arrange 3 of these letters in a specific order (i.e., the number of 3-letter permutations) is: $^5P_3$

Similarly, if you have the same set of 5 letters {A, B, C, D, E}, the number of ways you can select 3 of these letters in any order (i.e., the number of 3-letter combinations) is: $^5C_3$

## Random Variables
Say there is an outcome space $S$ which represents set of possible outcomes of an experiment and event space $F$ which is a set of subsets of $S$ then a random variable, $X$ is a **map** from $S$ to $R$ where $R$ is real numbers. $ X:S->R $ i.e a random variable takes on numerical values based on the outcome of a random experiment or process. In other words, A random variable is a quantity with an associated proability distribution. A probability distribution is a function that describes the **likelihood** of different outcomes in a random experiment or process.

For example, consider rolling a fair six-sided die. The possible outcomes are the numbers 1, 2, 3, 4, 5, and 6. Let X denote the result of rolling the die. X is a random variable, since its value depends on the outcome of the random experiment (i.e., the roll of the die).

Random variables can be classified as either discrete or continuous. A discrete random variable takes on a countable number of distinct values, such as the number of heads obtained when flipping a coin multiple times. A continuous random variable takes on an uncountable number of values within a given range, such as the height of individuals in a population.
### CDF
Cumulative Distribution Function (CDF) is a function that describes the probability that a random variable X is less than or equal to a given value x, for all possible values of x.

The CDF of a random variable X is denoted by $F(x)$ and is defined as:

$$ F(x) = P(X ≤ x) = \int_{-infty}^x f_X(x)dx $$

where $P(X ≤ x)$ represents the probability that X is less than or equal to x and $f_X(x)$ represents PDF.

For example, suppose we have a fair six-sided die. Let $X$ be the event of a number rolled on the die, and let P(X = x) be the probability of rolling a certain number x. Then, the CDF of X can be calculated as follows:
$$
P(X ≤ 1) = P(X = 1) = 1/6 
P(X ≤ 2) = P(X = 1) + P(X = 2) = 1/6 + 1/6 = 1/3 
. 
. 
P(X ≤ 6) = P(X = 1) + P(X = 2) + P(X = 3) + P(X = 4) + P(X = 5) + P(X = 6) = 1
$$

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
### PDF and PMF
A probability mass function (PMF) is a function that describes the probability distribution of a discrete random variable. The PMF gives the probability of each possible value(x) that the random variable can take on. Specifically, the PMF of a discrete random variable X is defined as:

$$ P_X(x) = Pr(X=x) $$

where x is a possible value of X, and Pr(X=x) is the probability that X takes on the value x.

On the other hand, a probability density function (PDF) is a function that describes the probability distribution of a continuous random variable. The PDF gives the relative likelihood that the random variable takes on a specific value, by providing the rate at which the probability changes as the value of the random variable changes. Specifically, the PDF of a continuous random variable X is defined as:

$$ f_X(x) = dF(x)\\over dx $$ 

where $F(x)$ is the CDF of X, and $dF(x)\\over dx$ is the derivative of the CDF with respect to x.

Both the PMF and the PDF have the property that the area under the curve equals one. For a PMF, this means that the sum of the probabilities for all possible values of X is equal to one, while for a PDF, this means that the integral of the PDF over all possible values of X is equal to one.


# References
The information is pulled from various sources from internet. Major sources are:
1. [Practical Statistics for Data Scientists](https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/)
2. [EE 505 Prabability by Rahul Trivedi](https://sites.google.com/view/ee505uwfall2022/course-material)
5. [Ace the Data Science Interview](https://www.acethedatascienceinterview.com/)
6. ChatGPT :)

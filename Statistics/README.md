<details>
<summary>Table of Contents</summary>

## Table of contents
Please use the github's interactive navigation. (Too lazy to write/generate TOC)

![toc](../Machine_Learning/Images/toc.gif)

</details>

# Data and Sampling Distributions
## Random Sampling and Sample Bias
A sample is a subset of data from a larger data set called population. 
> Random Sampling is a process in which each member in population has equal chance of being chosen for sample.

If sampling is not random, it will result in sample bias i.e the sample does not represent the population.

> Descriptive statisics are used to summarize and describe the characteristics of a dataset. Measures like mean, median, and standard deviation are examples of descriptive statistics.
>  Inferential statistics are used to draw conclusions about a population based on a sample of data. Techniques like hypothesis testing and confidence intervals are examples of inferential statistics.

Data distribution is the frequency distribution of **individual values** in a data set. Sampling distribution is the frequency distribution of **sample statistic**(mean, median etc) over many samples. People deal with samples beacuase dealing with millions of large populations is not practically ideal.
## Sample Statistic
A sample statistic is a numerical measure that summarizes the characteristics of a sample of data. It is calculated from the sample data and is used to estimate the corresponding population parameter. It could be mean, standard deviation, variance, median, mode etc. Sample statistics are subject to sampling variability, which means that different samples of the same size from the same population may produce different sample statistics. This is why we use histograms, boxplots, violin plots, standard errors, confidence intervals and hypothesis testing to quantify the uncertainty in our estimates and make statistical inferences.
## Standard Error
W.K.T standard deviation represents variability of individual data values. To get variability of sampling distribution, we calculate **standard error**. For a standard deviation of s and sample size of n, SE is given by:

$$ SE = {s\\over \sqrt(n)} $$

## Confidence Intervals
Population proportion refers to the proportion or percentage of individuals in a population who have a particular characteristic of interest. For example, if we are interested in the proportion of people in a city who own a car, the population proportion ($p$) would be the percentage of all people in that city who own a car. But again estimating something from all population is not practical so we use sample proportions $\hat{p}. We could take a random sample of 500 people from the city, and count how many of them own a car. Let's say we find that 300 people in the sample own a car. The sampling proportion is then 300/500 = 0.6, or 60%.

Though sampling proportions help with estimating the population proportions, how confident are we about samples? We might be different proportions with different samples. So we try to give a range of values instead of a single value called confidence interval.
> A confidence interval is a range of values that likely contains the true value of a population parameter with a certain degree/percentage of confidence.

In other words, it is the mean of your estimate plus and minus the variation in that estimate. This is the range of values we expect our estimate to fall between if we redo our test, within a certain level of confidence. The level/percentage of confidence tells us the percentage of times we expect to reproduce an estimate between the upper and lower bounds of the confidence interval set by $CL = 1 - \alpha$. If we use $\alpha$ of 0.005, we represent 95% confidence level.
## 
## Limit Theorems
### Sample Mean
Let $X_1$,.. $X_n$ be a sequence of iid random variables. The sample mean is given by

$$ S_n = {1\\over n}{\sum_{i=1}^n X_i} $$
### Weak Law of Large Numbers
$$ \lim_{n\to\infty} P(|S_n-\mu|>k)=0 $$

As the sample size n approaches infinity, the sample mean **converges in probability** to the true population mean. This means that as the sample size gets larger and larger, the probability that the sample mean deviates from the true population mean by more than k approaches zero.
### Strong Law of Large Numbers
$$ P(\lim_{n\to\infty} S_n = \mu) = 1 $$

As sample size n approaches infinity, the sample mean **converges almost surely** to mean. This provides a theoretical justification for the use of sample means as unbiased estimators of population means.

For example, a coin might land on heads 5 times in a row, but over a much larger n we would expect the proportion of heads to be approximately half of the total flips. Similarly, a casino might experience a loss on any individual game, but over the long run should see a predictable profit over time.
### Central Limit Theorem
It states that if we repeatedly sample a random variable a large number of times, the distribution of the sample mean will approach a normal distribution regardless of the initial distribution of the random variable.

$$ S_n = {{X_1+X2+...X_n}\\over n} \sim N(\mu,{\sigma^2\\over n}) $$

In other terms,

$$ {(S_n - \mu)\\over \sqrt(n\sigma^2)} \sim N(0,1) $$
# References
The information is pulled from various sources from internet. Major sources are:
1. [Practical Statistics for Data Scientists](https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/)
5. [Ace the Data Science Interview](https://www.acethedatascienceinterview.com/)
6. ChatGPT :)

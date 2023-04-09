<details>
<summary>Table of Conetents</summary>

## Table of contents
Please use the github's interactive navigation. (Too lazy to write/generate TOC)
![toc](Images/toc.gif)

</details>

## Note
Ability to think probabilistically is important to data scientists. But only few basic concepts are enough for interviews and these concepts will go in depth in the statistics. So the meat of the concepts are in statistics.
# Probaility
## Basics
### Conditional Probability
One of the most asked questions are based on this. For example: What is the proability of a patient having a disease, given that the patient tested positive for the disease ? The questions generally asks What is the probaility of an event A given that an event B has occurred? 

$$ P(A|B) = {{P(B|A)P(A)}\\over P(B)} $$

$P(A)$ is called Prior, $P(B|A)$ is likelihood and $P(A|B)$ is posterior.
### Law of Total Proability
If we have several disjoint events within B having occurred, we can then break down the probability of an event A having also occurred with law of total probability.

$$ P(A) = P(A|B_1)P(B_1) + ... + P(A|B_n)P(B_n) $$

This equation provides a handy way to think about paritioning events. If we want to model the proability of an event A happening, it can be decomposed into weighted sum of conditional probabilities based on each possible scenarios having occurred. If there is a *tree of outcomes*, then this eqn is useful. One example is the probability that a customer makes a purchase, conditional on which segment that customer falls in.
### Counting
Just basics of permuatations and combinations. If order matters go for $^nP_r$ else $^nC_r$.

$$ ^nP_r = n*(n-1)....*(n-r+1) = {n!\\over (n-r)!} $$

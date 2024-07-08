---
layout: post
title: "2. Variational Inference, Expectation Maximization and VAE"
posted: "4 June, 2024"
categories: Deep-Generative-Modelling
live: true
---

In part 1 of this series, we converted the problem of finding an approximate parametric probability distribution into a dual optimization problem. That is, finding the $\theta$ and $\phi$ that simultaneously 

1. Maximize the marginal likelihood of data <span>$P_{\theta}(x)$</span>
2. Make the approximate posterior <span>$Q_{\phi}(z | x)$</span> similar to <span>$P_{\theta}(z | x)$</span>

Such a problem is a common setup for [Expectation-Maximization algorithm](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm) (EM algorithm). 

But since <span>$P_{\theta}(z | x)$</span> is intractable, we cannot use the EM algorithm this way. Instead, we will first wrangle the objective functions to make them tractable. As we proceed with the math, we will realize that both parts of the objective function can be approximately optimised using a single loss function. That loss function is named Evidence Lower BOund (ELBO).

## Evidence Lower Bound (ELBO)

Let’s start by analyzing part 2 of our objective - making approximate posterior <span>$Q(z|x,\phi)$</span> similar to the posterior <span>$P(z|x,\theta)$</span>. A common metric for measuring similarity between 2 probability distributions is the [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) (KL Divergence). KL Divergence, denoted as <span>$D_{KL}( \ P \ || \ Q \ )$</span>, is the expected excess [surprise](https://en.wikipedia.org/wiki/Information_content) from using $Q$ as a distribution when the actual distribution is $P$. KL Divergence is not symmetric and <span>$D_{KL}( \ P \ || \ Q \ ) \neq D_{KL}( \ Q \ || \ P \ )$</span>.

We will use the reverse KL divergence between <span>$P$  and <span>$Q$ so that the areas in latent space which are truly low probability, but are assigned high probability by <span>$Q$, are penalized more. The reverse KL Divergence between the 2 posteriors can be written as

$$
\begin{aligned}
D_{KL}\left( \ Q(z|x,\phi) \ || \ P(z|x,\theta) \ \right) &= \int_{Z}Q(z|x,\phi)\log\frac{Q(z|x,\phi)}{P(z|x,\theta)}dz \\ \\
&= \int_{Z}Q(z|x,\phi)\left[\log Q(z|x,\phi) - \log P(z|x,\theta)\right]dz \\ \\
&= \int_{Z}Q(z|x,\phi)\left[\log Q(z|x,\phi) - \log P(x,z|\theta)   + \log P(x|\theta)\right]dz
\end{aligned}
$$

Now the last term, <span>$\log P(x|\theta)$</span> does not depend on $z$. And therefore we can rewrite it to

$$
\begin{aligned}
&= \int_{Z}Q(z|x,\phi)\left[\log Q(z|x,\phi) - \log P(x,z|\theta)  \right]dz + \log P(x|\theta) \\ \\
&= \int_{Z}Q(z|x,\phi)\left[\log \frac{Q(z|x,\phi)}{P(x,z|\theta)} \right]dz + \log P(x|\theta) \\ \\
&= D_{KL}(Q(z|x,\phi) \ || \ P(x,z|\theta)) + \log P(x|\theta)
\end{aligned}
$$

Thus we have

$$
D_{KL}\left( \ Q(z|x,\phi) \ || \ P(z|x,\theta) \ \right)= D_{KL}( \ Q(z|x,\phi) \ || \ P(x,z|\theta) \ ) + \log P(x|\theta)
$$

Rearranging the equation, we get,

$$
D_{KL}\left( \ Q(z|x,\phi) \ || \ P(z|x,\theta) \ \right) - D_{KL}( \ Q(z|x,\phi) \ || \ P(x,z|\theta) \ ) = \log P(x|\theta)
$$

Lets denote

$$
L(\phi,\theta) = - D_{KL}( \ Q(z|x,\phi) \ || \ P(x,z|\theta) \ )
$$

Then we have

$$
 L(\phi, \theta) = \log P(x|\theta) - D_{KL}\left( \ Q(z|x,\phi) \ || \ P(z|x,\theta) \ \right)
$$

Now the second term on the right hand side is the second part of our objective function that we wish to minimize. The first term on the right hand side, also called evidence, is the first part of objective function, that we wish to maximize. So if we maximize <span>$L(\phi, \theta)$</span>, we automatically minimize the KL divergence between the approximate posterior <span>$Q(z|x,\phi)$</span> and posterior <span>$P(z|x,\theta)$</span> and simultaneously also maximize the likelihood of data.

Also, since KL divergence is a positive quantity, <span>$L(\phi, \theta)$</span> is a lower bound on <span>$\log P(x|\theta)$</span>, the evidence. Therefore, <span>$L(\phi, \theta)$</span> is known as the Evidence Lower Bound, or in short ELBO. The bound is tight, i.e. <span>$L(\phi, \theta)$</span> is exactly equal to <span>$\log P(x|\theta)$</span> when <span>$Q(z|x,\phi)$</span> is equal to <span>$P(z|x,\theta)$</span>.

## Alternative Derivation

Maximizing the “efficiently” computable ELBO automatically minimizes the intractable KL divergence between the approximate posterior <span>$Q(z|x,\phi)$</span> and posterior <span>$P(z|x,\theta)$</span> and simultaneously also maximize the likelihood of data. 😎

You could arrive at the same equation by starting with first part of our objective i.e. maximizing the log likelihood of data <span>$\log P(x|\theta)$</span>. 

$$
\begin{aligned}
\log P(x|\theta)&= E_{Q(z|x,\phi)}\log P(x|\theta)\\ \\
&= E_{Q(z|x,\phi)}\log \frac{P(x,z|\theta)}{P(z|x,\theta)} \\ \\
&= E_{Q(z|x,\phi)}\log \frac{P(x,z|\theta)Q(z|x,\phi)}{P(z|x,\theta)Q(z|x,\phi)}\\ \\
&= E_{Q(z|x,\phi)}\log \frac{Q(z|x,\phi)}{P(z|x,\theta)} + E_{Q(z|x,\phi)}\log \frac{P(x,z|\theta)}{Q(z|x,\phi)} \\ \\
&= E_{Q(z|x,\phi)}\log \frac{Q(z|x,\phi)}{P(z|x,\theta)} - E_{Q(z|x,\phi)}\log \frac{Q(z|x,\phi)}{P(x,z|\theta)} \\ \\
&= D_{KL}(Q(z|x,\phi) \ || \ P(x,z|\theta)) + L(\phi, \theta)
\end{aligned}
$$

Thus, optimizing ELBO loss is akin to MLE.

## Computing ELBO

Given a dataset $X = \{x_1, x_2, ..., x_N\}$, the objective we need to minimize is the expected value of negative of ELBO. Since 

$$
\begin{aligned}
-\frac{1}{N}\sum_{i=1}^N L(\phi,\theta,x_i) &= \frac{1}{N}\sum_{i=1}^N D_{KL}( \ Q(z|x_i,\phi) \ || \ P(x_i,z|\theta) \ )\\
&=-\frac{1}{N}\sum_{i=1}^N \left[ E_{Q(z|x_i,\phi)}(\log Q(z|x_i,\phi) - \log P(x_i,z|\theta))\right]\\
&=\frac{1}{N}\sum_{i=1}^N \left[ E_{Q(z|x_i,\phi)}(\log P(x_i,z|\theta) - \log Q(z|x_i,\phi))\right] \\
&=\frac{1}{N}\sum_{i=1}^N \left[ \int Q(z|x_i,\phi)(\log P(x_i,z|\theta) - \log Q(z|x_i,\phi))dz\right] \\
\end{aligned}
$$

Wait! To evaluate objective function, we need to integrate over all values of $z$ for every sample in dataset! That is, for every image $x_i$ in our dataset of images, we need to evaluate the expression inside the integral for all $z$ where <span>$Q(z|x_i,\phi)$</span> is positive. Since that is intractable, we can resort to [Monte Carlo sampling](https://en.wikipedia.org/wiki/Monte_Carlo_method). Monte Carlo methods use law of large numbers to obtain expected value of a random variable (or function of random variable) by repeated sampling. Given a data point $x_i$, we sample $\{z_1, z_2, ..., z_M\}$ values of $Z$ from <span>$Q(z|x_i,\phi)$</span>. Then our Monte Carlo estimate for ELBO becomes

$$
\begin{aligned}
&=\frac{1}{N}\sum_{i=1}^N \left[ \frac{1}{M}\sum_{j=1}^M(\log P(x_i,z_j|\theta) - \log Q(z_j|x_i,\phi))\right] \\
\end{aligned}
$$

 We can actually simplify our integral a bit so that the Monte Carlo estimate has lower variance.

$$
\begin{aligned}
-\frac{1}{N}\sum_{i=1}^N L(\phi,\theta,x_i) &= 
\frac{1}{N}\sum_{i=1}^N \left[ \int Q(z|x_i,\phi)(\log P(x_i,z|\theta) - \log Q(z|x_i,\phi))dz\right] \\
&= \frac{1}{N}\sum_{i=1}^N \left[ \int Q(z|x_i,\phi)(\log P(x_i|z,\theta) + \log P(z | \theta) - \log Q(z|x_i,\phi))dz\right] \\
\end{aligned}
$$

Since, $z$ does not depend on $\theta$, <span>$P(z|\theta) = P(z)$</span>, the prior distribution of latent variable $Z$.

$$
\begin{aligned}
&= \frac{1}{N}\sum_{i=1}^N \left[ \int Q(z|x_i,\phi)(\log P(x_i|z,\theta) + \log P(z) - \log Q(z|x_i,\phi))dz\right] \\
&= \frac{1}{N}\sum_{i=1}^N \left[ \int Q(z|x_i,\phi)\log P(x_i|z,\theta)dz - D_{KL}( \ Q(z|x_i,\phi) \ || \ P(z) \ )\right] \\
\end{aligned}
$$

now only <span>$\int Q(z|x_i,\phi)\log P(x_i|z,\theta)dz$</span> part of the equation is intractable while <span>$D_{KL}( \ Q(z|x_i,\phi) \ || \ P(z) \ )$</span> is easy to compute for a given $x_i$. So a better estimate would be

$$
\begin{aligned}
-\frac{1}{N}\sum_{i=1}^N L(\phi,\theta,x_i) &= 
\frac{1}{N}\sum_{i=1}^N \left[ \frac{1}{M}\sum_{j=1}^M \log P(x_i|z_j,\theta) - D_{KL}( \ Q(z|x_i,\phi) \ || \ P(z) \ )\right]
\end{aligned}
$$

where $\{z_1, z_2, ..., z_M\}$ are sampled from <span>$Q(z_j|x_i,\phi)$</span>.

The original VAE paper claims that having $M = 1$, that is, sampling just one $Z$ for every given $x_i$ works well enough as the objective function as long as $N$ is large (around 100 or more) ⚠️. So the loss we minimize is

$$
\frac{1}{N}\sum_{i=1}^N \left[ \log P(x_i|z_i,\theta) - D_{KL}( \ Q(z|x_i,\phi) \ || \ P(z) \ )\right]
$$

where $z_i$ is a single sample from <span>$Q(z|x_i,\phi)$</span>.

## Intuition of ELBO objective

The first term in the loss, <span>$\log P(x_i|z_i,\theta)$</span> is called the reconstruction error. This is because, given a data point $x_i$, we sample a $z_i$ from <span>$Q(z|x_i,\phi)$</span>. Then we evaluate the probability of the original $x_i$ being reconstructed from <span>$P(x|z_i,\theta)$</span>.

<div class="callout">
🤔 <b>I understand term reconstruction, but why the term error? Shouldn’t it be reconstruction probability?</b><br/>
For some standard distributions (like Gaussian), minimizing the log probability is equivalent to minimizing the squared error between data point and the mean of the distribution. Try it out by expanding <span>$\log P(x_i|z_i,\theta)$</span> by substituting the formula for Gaussian and mean and standard deviation ${\mu_i, \sigma_i} = f_{\theta}(z_i)$, our decoder function.
</div>

The second term <span>$D_{KL}( \ Q(z|x_i,\phi) \ || \ P(z) \ )$</span> can be viewed as a regularizing term that says the approximate posterior <span>$Q(z|x_i,\phi)$</span> should be close to the prior <span>$P(z)$</span>. Prior <span>$P(z)$</span> is usually a simple distribution like a isotropic, zero mean, unit variance, multivariate Gaussian.

The final objective function could have additional term to help with the optimization process. For example, for MAP estimates of parameters $\theta$ and $\phi$, we use an appropriate prior for parameters’ distribution too. Adding a function weight regularizer imposes a penalty on the sum squared value of the parameters. This is equivalent to saying that parameters come from a Gaussian with mean 0 and standard deviation 1.

## Variational Auto-Encoders (VAE)

The original VAE paper makes 2 significant changes to the problem. These changes enable us to estimate $\theta$ and $\phi$ using mini batch stochastic gradient descent rather than using the EM algorithm. In the next post, we shall learn more about implementation of variational inference using VAEs.

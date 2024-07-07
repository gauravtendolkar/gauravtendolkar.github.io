---
layout: post
title: "1. Learning Probability Distribution of Data"
posted: "1 June, 2024"
categories: Understanding-Deep-Generative-Modelling
live: true
---

In generative AI, we search for functions (probabilistic functions in most cases) that can generate data (like images, text, audio, video, graphic designs, etc) either unconditionally or conditioned on some input. Such functions are also called models. In unconditional generation, you wish to just generate random examples of data. For instance, generating an image of a random human face each time you call the model is an example of unconditional generation. In conditional generation, the model generates an output for a given input. For example, generating an image of a random human face given that we want a “male face with beard” is an example of conditional generation.

In this series, we will focus on the problem of “realistic” image generation - specifically generating 32x32 resolution images of a particular domain - like handwritten digits, wallpaper patterns, human faces, etc. But the theory, as well as much of the code, is transferable to other modalities like text or audio generation too.

To find the function (or a model), we first collect lots of examples of the type of data we want to generate. We call this our dataset. Then, we search for a mathematical function that maps inputs to outputs. For example, for generating 32x32 colored images from text, we find a mathematical function that takes input text (represented as sequence of numbers) and outputs 32x32x3 numbers corresponding to every pixel’s red, green and blue color values. We search for a function that can output “realistic” images. Realistic means that it should be difficult to say whether a generated image was part of the real examples in dataset or was synthetically created by the model. The crux of AI lies in searching algorithms for finding such mathematical functions.

Note that we are looking for probabilistic functions (also known as Probability Distribution Functions or PDFs).

## Data Generating Process

We assume that the data (in our case - a 32x32 grid of pixel values), is a random variable (denoted as $X$) generated from a Probability Distribution Function (PDF) unknown to us. We call it the true probability distribution function and denote it by <span>$P^*(x)$. 

We don’t know the function <span>$P^*(x)$</span> but we do have samples from it (which is our dataset of images). 

Using that dataset, we wish to find an approximation to this <span>$P^*(x)$. The approximation is another probability distribution function - either a parametric one <span>$P(x; \theta)$</span> with parameters $\theta$, or a non-parametric PDF. 

<div class="callout">
🤔 <b>Why not just sample random images from the dataset? Why estimate a probability distribution to sample from?</b><br/>
An obvious advantage is that we can generate interesting images that are outside our limited dataset. But there is more to this approach. We can also achieve tasks like in-painting or out-painting where we are provided with only a part of the image and we need to complete the image based on that part. We can perform de-noising, compression, image super-resolution and many more tasks. Additionally, if our distributions employ latent random variables,  we can use the latent space to cluster images or vary isolated properties of images. We shall learn about these approaches later.
</div>

To find a parametric distribution function, we first define a function of $x$ and $\theta$ denoted as <span>$P(x; \theta)$</span> (using our intuitions about the function’s structure) and then search for a value of $\theta$ that will make <span>$P(x; \theta)$</span> similar to the unknown <span>$P^*(x)$</span> using hints from the dataset.

Even for the task of generating $32\times 32$ resolution RGB images, the random variable is a vector (or a matrix/tensor) of $32\times 32\times 3 = 3072$ dimensions. Therefore, we have a task of estimating a PDF over 3072 dimensional space (do not attempt to visualize). The distribution could be an extremely complicated function and it is not apparent that it takes a form that can be approximated by simple [parametric distribution functions with known properties](https://en.wikipedia.org/wiki/Category:Continuous_distributions) (like multivariate normal distribution function). But fortunately, most applications do not require a [closed form solution](https://en.wikipedia.org/wiki/Closed-form_expression) for <span>$P(x; \theta)$. 

<div class="callout">
💁 <b>Some Common Notations</b><br/>
Capital letters, like $X$, denote a random variable. Small letters, like $x$, denote a particular value of the random variable. <span>$P(x)$</span> denotes the Probability Distribution Function (PDF) of random variable $X$. It is a function, not a value. On other hand, <span>$P(X = x)$</span> denotes the value of PDF when random variable $X$ has value $x$. The notation <span>$P(x; \theta)$</span> is also a PDF but it additionally means that means that the probability of random variable $X$ taking value $x$ depends on the value of $x$ and parameters $\theta$. <span>$P(x|\theta)$</span> denotes the conditional PDF of $X$ for a given $\theta$. $C(x)$</span> denotes the Cumulative Distribution Function (CDF) of random variable $X$.

</div>

For example, for the task of generating random images, we only need an algorithm that lets us efficiently sample from <span>$P(x; \theta)$. The closed form solution, or even the value of <span>$P(x; \theta)$</span> for the sampled $x$, is not necessary. Sampling from a PDF without knowing the PDF sounds strange, but as we shall see, it is possible. We will use this to our advantage.

The only constraints on this approximate probability distribution <span>$P(x; \theta)$</span> are that -

1. It should be “similar” to <span>$P^*(x)$.
2. It should be easy to sample from in order to generate new images efficiently.

If it satisfies these 2 criteria, we can generate new images that are “similar” to the ones in our dataset.

For this series, we will stick to methods for approximating <span>$P^*(x)$</span> with parametric distributions <span>$P(x; \theta)$. Also, we will use data and Maximum Likelihood Estimation (MLE) or Maximum A Posteriori (MAP) methods to obtain a point estimate for our parameters $\theta$.

## Parameter Estimation Methodology

Bayesian statistics, the parent field of generative AI, views the parameters $\theta$ of the distribution <span>$P(x; \theta)$</span> as unknown random variables. Therefore, there is a probability distribution associated with $\theta$ which assigns higher probabilities to parameter values that can generate data similar to the samples in the provided dataset. The objective of Bayesian statistics is to learn this distribution of parameters given the fixed dataset (which we shall denote as $D$). In other words, we try to estimate the conditional distribution <span>$P(\theta | x=D)$</span> - the probability distribution of parameters that  generated the fixed data. This way, not only do we get statistics like the expected value of parameters, but also get the uncertainty about those parameters. 

But, estimating this is hard. Assuming that our data samples are independent and identically distributed (I.I.D.). Then laws of probability and Bayes rule gives,

$$
P(\theta | x=D) = \frac{P(X=D|\theta)P(\theta)}{P(X=D)} = \frac{P(\theta)\prod_{x\in D}P(X=x|\theta)}{\prod_{x\in D}P(X=x)} 
$$

Here, <span>$P(\theta)$</span> is called the prior, and it is a function of $\theta$ that measures our belief in probabilities of parameter values without observing any data. <span>$P(X=x | \theta)$</span> is called the likelihood function. It is also a function of $\theta$ and measures the probability of observing a given output $x$ for parameter value $\theta$. <span>$P(\theta | x=D)$</span> is called the posterior, and it is a function of $\theta$ that measures our updated belief in probabilities of parameter values after observing the data. Finally, <span>$P(X=x)$</span> is called the marginal distribution. Since data is fixed, <span>$P(X=x)$</span> is a fixed constant for all $x$ in dataset $D$. 

For a given $\theta$ value, the output of likelihood function <span>$P(X=x|\theta)$</span> is easy to compute (just evaluate the PDF <span>$P(x; \theta)$</span> with parameters $\theta$ and sample $x$). The prior PDF <span>$P(\theta)$</span> is either known to us from experience or assumed to be a uniform distribution. But, the marginal distribution, even for individual sample <span>$P(X=x)$</span> is not tractable. Even though it is just a constant, we cannot compute its value. This is because we need to get the probability of data marginalized over all possible values of $\theta$. In other words, we need to compute the following integral which is difficult (because it is computationally expensive and not because integrals look scary).

$$
P(X=x) = \int_{\theta}P(X=x | \theta)P(\theta) d\theta
$$

Therefore, we won’t try to estimate the entire probability distribution of parameters. We will make our peace with not knowing the uncertainties in parameters. Instead, we will try to obtain the $\theta$ that maximises <span>$P(\theta | x)$. Such an estimate of $\theta$ is also known as a point estimate. The denominator marginal distribution, being constant, plays no part in maximization and therefore can be ignored. We find the point estimate $\theta^*$ by maximizing the numerator,

$$
\theta^* = \text{argmax}_{\theta}P(X=D \ | \ \theta)P(\theta)
$$

Such an estimate is also called the Maximum a Posteriori (MAP) estimate.

In cases when we assume a uniform distribution for <span>$P(\theta)$, the optimization reduces to, 

$$
\theta^* = \text{argmax}_{\theta}P(X=D \ | \ \theta)
$$

Such an estimate is also called the Maximum Likelihood Estimate (MLE).

These are tractable optimization problems compared to estimation of entire probability distribution <span>$P(\theta | X)$.

In this series, we shall focus on MLE to find optimal parameter values for generating “realistic” data. Note that, such optimizations are performed on a computer. Since probabilities lie between 0 and 1 and the datasets are usually large, multiplying many such probabilities leads to numerical issues.

$$
\theta^* = \text{argmax}_{\theta}P(X=D \ | \ \theta) = \text{argmax}_{\theta}\prod_{x\in D}P(X=x \ | \ \theta)
$$

To avoid such numerical issues, in practice, we find the parameters that maximize the log likelihood (or equivalently, minimize negative of log likelihood). Since $\log$ is a monotonous increasing function, we get the exact same $\theta^*$ as solution.

$$
\theta^* = \text{argmax}_{\theta}\sum_{x \in D} \log P(X=x | \theta)
$$

## Sampling

Let’s say we found the optimal parameters of our approximate probability distribution function using MLE. To generate new images, we would have to sample from it.

<div class="callout">
💡 Sometimes, instead of finding parameters of a parametric PDF, we find parameters of a function that directly returns a sample. For example, a generator in GAN.
</div>

To sample from a known Probability Distribution Function (PDF), you need a random number generator that can generate a uniform random number between 0 and 1 and a way to compute the inverse of Cumulative Distribution Function (CDF) from the PDF. For random variables that take continuous values, The CDF is

$$
C(X=x) = \int_{\{z \in X | z \leq x\}}{P(X=z)dz}
$$

Then, the traditional sampling algorithm is,

1. Pick a random number $r$ between 0 and 1 with uniform probability.
2. Choose your sample $x^*$, such that, 

$$
x^* = C^{-1}(r)
$$

As you can see, to sample using this algorithm efficiently, you need to know the closed form solution for the inverse of CDF (inverse CDF function of a random variable is also called the quantile function of a random variable).

There are other techniques that do not require having a closed form solution to the inverse CDF. For example, [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling), is an ingenious method that can sample from arbitrary PDFs. The family of [Markov Chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) algorithms are another way of sampling without knowing closed form solution to the inverse CDF. But these methods can be computationally expensive for high dimensional data where the probability mass is distributed sparsely within the entire domain.

A well approximated <span>$P(x; \theta)$</span> is not useful without an efficient way to sample from it.

## Using a Simpler Family of PDFs

We could make sampling efficient if we had a closed form solution for the inverse CDF of our random variable. A few simple parametric probability distribution functions either have a closed form equation for their inverse CDF or a good approximation for it.

If we could approximate our true distribution with one of these simple parametric distributions, we could efficiently generate samples similar to our observed data. Alas, most real world problems, like ours, cannot be approximated well by such simple parametric probability distributions.

A common trick is to use Probabilistic Graphical Models. Probabilistic Graphical Models use a simple parametric output probability distribution. The parameters of this distribution are conditional on another random variable from another simple parametric probability distribution. Such intermediate random variables are called latent variables. Extending this one step to multiple latent variables and a dependency graph between them leads to the approach of Probabilistic Graphical Models. 

## Implicit Parametric Probability Distribution

We imagine a new random variable $Z$ which has one of these simple parametric probability distribution functions - let’s say a multivariate isotropic Gaussian PDF.  Do not worry about what $Z$ represents. It is just an invented random variable that will make our problems easier to solve. As we formulate the mathematics, we will arrive at an intuition for $Z$. We will call this variable a latent variable and denote its PDF as <span>$P(z)$. <span>$P(z)$</span> can be used to sample different values $z$ of random variable $Z$.

<div class="callout">
📝 <b>Short note on Multivariate Gaussian</b><br/>
First, something that confuses me if I am not paying attention - Multivariate Gaussian is different from mixture of Gaussians. Multivariate Gaussian is a Gaussian distribution over a multidimensional vector but has a single mode. You can also have a mixture of multivariate Gaussians which will give you multimodal distributions. Another important thing to note about a Multivariate Gaussian is that, in higher dimensions, the distribution does not look or behave like you expect in 1D and 2D cases. This is due to the [Gaussian Soap Bubble Effect](https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/).

</div>

Along with <span>$P(z)$, we choose a sufficiently complex function $f_{\theta}$ with parameters $\theta$. The input to $f_{\theta}$ is a sample of latent variable $z$ and the outputs are numbers used to parametrize another different simple parametric probability distribution - let’s say another multivariate isotropic Gaussian PDF. Let’s denote the conditional PDF of $X$ given $z$ as <span>$P_{\theta}(x | z)$. Now, sampling a data point involves 

1. Sampling $z$ from <span>$P(z)$. This step is computationally tractable thanks to [Box-Muller](https://en.wikipedia.org/wiki/Box–Muller_transform) method.
2. Computing $f_{\theta}(z)$ to get $\mu$ and $\sigma$ parameters of output multivariate isotropic Gaussian distribution. This step is also computationally tractable.
3. Sampling a data point $x$ from $N(\mu, \sigma)$. Like step 1, this step is also computationally tractable.

<div class="callout">
🤔 <b>Why use multivariate isotropic Gaussian distributions? Will other distributions work?</b><br/> 
We choose a multivariate isotropic Gaussian PDF for <span>$P(z)$</span> and <span>$P_{\theta}(x | z)$</span> because, even though we do not have a closed form solution for the inverse CDF of a Gaussian, we still have efficient algorithms to sample from them. One such method is the Box-Muller method that lets you create random samples from a Gaussian distribution given samples from a uniform distribution. Java’s [Random](https://docs.oracle.com/javase/8/docs/api/java/util/Random.html#nextGaussian) class uses this method for sampling numbers from standard Gaussian. The method can be easily modified for the multivariate isotropic case.

</div>

Thus we have found an efficient way to sample data points from a parametric probability distribution that is more powerful than simple known parametric distributions! 😎 (satisfying the [requirements](https://www.notion.so/Learning-Probability-Distribution-of-Data-37cb35fa0de844e3b1435ff42267c5c2?pvs=21) specified before)

## Intractable posterior

First, to make equations easier to read,  lets denote <span>$P(x | \theta)$</span> as <span>$P_{\theta}(x)$. Now, it is important to note that even though, in this approach, sampling a data point $x$ is efficient, we cannot calculate <span>$P_{\theta}(x)$. That is, we cannot compute the marginal probability of data point $x$ (marginalized over $z$) being sampled from the parametric distribution <span>$P_{\theta}$ efficiently. Because, by Bayes rule, we have

$$
P_{\theta}(x) = \frac{P_{\theta}(x | z)P_{\theta}(z)}{P_{\theta}(z|x)}
$$

Here <span>$P_{\theta}(x|z)$</span> is tractable since this is a forward pass of $z$ through our model $f_{\theta}$ to get parameters $\mu$ and $\sigma$ of a Gaussian distribution followed by computing probability of sampling $x$ from the Gaussian distribution with parameters $\mu$ and $\sigma$.

$P_{\theta}(z)$, which is the probability distribution of sampling $z$ from the prior standard normal distribution (Gaussian distribution with mean 0 and standard deviation 1) is also tractable. This is not a function of $\theta$.

But we do not have any efficient way to compute <span>$P_{\theta}(z|x)$. That is, we do not have any efficient way of saying what was the probability distribution over latent variable $z$ that could have resulted in data point $x$ being sampled. <span>$P_{\theta}(z|x)$</span> is called the posterior distribution of latent variable $Z$. 

But we do need efficient way to compute <span>$P_{\theta}(x)$</span> since to find the optimal parameters $\theta$ via MLE approach, we need to maximize the likelihood of data.

$$
\theta^* = \text{argmax}_{\theta}\sum_{x \in D} P_{\theta}(X=x)
$$

## Encoder decoder architecture

We saw that since <span>$P_{\theta}(z|x)$</span> is intractable, we cannot learn the parameters of $f_{\theta}$ - our decoder (a decoder converts a latent sample into a data point or a distribution over data point). 

To make this problem tractable, we find another parametric model $g_{\phi}$ that takes input a data point $x$ and outputs $\mu$ and $\sigma$ of a Gaussian distribution for $z$. Using this model, we can compute the proxy for probability <span>$P_{\theta}(z|x)$</span> efficiently and we denote it <span>$Q_{\phi}(z|x)$</span>.

This leads to the popular encoder decoder architecture.

So to sample data points similar to true distribution, we need to find $f_{\theta}$ and $g_{\phi}$ jointly such that 

1. KL divergence (a metric for distance between two probability distributions) between <span>$P_{\theta}(z|x)$</span> and <span>$Q_{\phi}(z|x)$</span> is minimized. This is the optimization objective for our encoder part since it encodes a data point into latent space. 
2. The likelihood of data $X$ under marginal distribution <span>$P_{\theta}(X)$</span> is maximized. This is the optimization objective for our decoder part since it converts a latent sample into a data point.

Such encoder decoder architecture is a very popular paradigm of generative models. Not only do they give us a latent space succinctly capturing patterns from high dimensional data, but also a probability distribution complex enough to model data and an efficient way to sample from it.

This dual optimization approach used to find parameters of encoder and decoder is called variational inference. Variational Auto-Encoders (VAEs) are a popular example of encoder-decoder architecture using variational inference to find parameters of both. Another example is Diffusion models which we will discuss in later part this series.
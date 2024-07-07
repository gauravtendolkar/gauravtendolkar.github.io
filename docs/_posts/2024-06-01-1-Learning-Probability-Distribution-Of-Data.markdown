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

We assume that the data (in our case - a 32x32 grid of pixel values), is a random variable (denoted as $X$) generated from a Probability Distribution Function (PDF) unknown to us. We call it the true probability distribution function and denote it by $P^{*}(x)$. We don’t know the function $P^*(x)$ but we do have samples from it (which is our dataset of images). Using that dataset, we wish to find an approximation to this $P^*(x)$. The approximation is another probability distribution function - either a parametric one $P(x; \theta)$ with parameters $\theta$, or a non-parametric PDF. 

<aside>
🤔 **Why not just sample random images from the dataset? Why estimate a probability distribution to sample from?**
An obvious advantage is that we can generate interesting images that are outside our limited dataset. But there is more to this approach. We can also achieve tasks like in-painting or out-painting where we are provided with only a part of the image and we need to complete the image based on that part. We can perform de-noising, compression, image super-resolution and many more tasks. Additionally, if our distributions employ latent random variables,  we can use the latent space to cluster images or vary isolated properties of images. We shall learn about these approaches later.

</aside>

To find a parametric distribution function, we first define a function of $x$ and $\theta$ denoted as $P(x; \theta)$ (using our intuitions about the function’s structure) and then search for a value of $\theta$ that will make $P(x; \theta)$ similar to the unknown $P^*(x)$ using hints from the dataset.

Even for the task of generating $32\times 32$ resolution RGB images, the random variable is a vector (or a matrix/tensor) of $32\times 32\times 3 = 3072$ dimensions. Therefore, we have a task of estimating a PDF over 3072 dimensional space (do not attempt to visualize). The distribution could be an extremely complicated function and it is not apparent that it takes a form that can be approximated by simple [parametric distribution functions with known properties](https://en.wikipedia.org/wiki/Category:Continuous_distributions) (like multivariate normal distribution function). But fortunately, most applications do not require a [closed form solution](https://en.wikipedia.org/wiki/Closed-form_expression) for $P(x; \theta)$. 

<aside>
💁 **Some Common Notations**
Capital letters, like $X$, denote a random variable. Small letters, like $x$, denote a particular value of the random variable. $P(x)$ denotes the Probability Distribution Function (PDF) of random variable $X$. It is a function, not a value. On other hand, $P(X = x)$ denotes the value of PDF when random variable $X$ has value $x$. The notation $P(x; \theta)$ is also a PDF but it additionally means that means that the probability of random variable $X$ taking value $x$ depends on the value of $x$ and parameters $\theta$. $P(x|\theta)$ denotes the conditional PDF of $X$ for a given $\theta$. $C(x)$ denotes the Cumulative Distribution Function (CDF) of random variable $X$.

</aside>

For example, for the task of generating random images, we only need an algorithm that lets us efficiently sample from $P(x; \theta)$. The closed form solution, or even the value of $P(x; \theta)$ for the sampled $x$, is not necessary. Sampling from a PDF without knowing the PDF sounds strange, but as we shall see, it is possible. We will use this to our advantage.

The only constraints on this approximate probability distribution $P(x; \theta)$ are that -

1. It should be “similar” to $P^*(x)$.
2. It should be easy to sample from in order to generate new images efficiently.

If it satisfies these 2 criteria, we can generate new images that are “similar” to the ones in our dataset.

For this series, we will stick to methods for approximating $P^*(x)$ with parametric distributions $P(x; \theta)$. Also, we will use data and Maximum Likelihood Estimation (MLE) or Maximum A Posteriori (MAP) methods to obtain a point estimate for our parameters $\theta$.
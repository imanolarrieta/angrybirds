'''
Helper code for MS&E 338 Reinforcement Learning implementation assignment.

The functions in the script below are designed to help you update posteriors
for the reward and transition functions. We will use a simple (and standard)
set of conjugate families.

You may find it helpful to use these functions to update and sample from
posterior distributions for PSRL. Note however that none of this code is
particularly optimized/vectorized. There may be better pre-packaged solutions
if you want to do this in practice, but this is a nice way to look "beneath the
hood" in a simple example.

Rewards
    Distribution approximation: Normal unknown mean, unknown variance
    Conjugate prior: Normal Gamma
    Wikipedia: http://en.wikipedia.org/wiki/Normal-gamma_distribution

Transitions
    Distribution approximation: Multinomial distribution
    Conjugate prior: Dirichlet
    Wikipedia: http://en.wikipedia.org/wiki/Dirichlet_distribution

author: iosband@stanford.edu
'''

import numpy as np

#---------------------------------------------------------------------------
# Rewards functions

def convert_prior(mu, n_mu, tau, n_tau):
    '''
    Convert the natural way to speak about priors to our paramterization

    Args:
        mu - 1x1 - prior mean
        n_mu - 1x1 - number of observations of mean
        tau - 1x1 - prior precision (1 / variance)
        n_tau - 1x1 - number of observations of tau

    Returns:
        prior - 4x1 - (mu, lambda, alpha, beta)
    '''
    prior = (mu, n_mu, n_tau * 0.5, (0.5 * n_tau) / tau)
    return prior

def update_normal_ig(prior, data):
    '''
    Update the parameters of a normal gamma.
        T | a,b ~ Gamma(a, b)
        X | T   ~ Normal(mu, 1 / (lambda T))

    Args:
        prior - 4 x 1 - tuple containing (in this order)
            mu0 - prior mean
            lambda0 - pseudo observations for prior mean
            alpha0 - inverse gamma shape
            beta0 - inverse gamma scale
        data - n x 1 - numpy array of {y_i} observations

    Returns:
        posterior - 4 x 1 - tuple containing updated posterior params.
            NB this is in the same format as the prior input.
    '''
    # Unpack the prior
    (mu0, lambda0, alpha0, beta0) = prior

    n = len(data)
    y_bar = np.mean(data)

    # Updating normal component
    lambda1 = lambda0 + n
    mu1 = (lambda0 * mu0 + n * y_bar) / lambda1

    # Updating Inverse-Gamma component
    alpha1 = alpha0 + (n * 0.5)
    ssq = n * np.var(data)
    prior_disc = lambda0 * n * ((y_bar - mu0) ** 2) / lambda1
    beta1 = beta0 + 0.5 * (ssq + prior_disc)

    posterior = (mu1, lambda1, alpha1, beta1)
    return posterior

def sample_normal_ig(prior):
    '''
    Sample a single normal distribution from a normal inverse gamma prior.

    Args:
        prior - 4 x 1 - tuple containing (in this order)
            mu - prior mean
            lambda0 - pseudo observations for prior mean
            alpha - inverse gamma shape
            beta - inverse gamma scale

    Returns:
        params - 2 x 1 - tuple, sampled mean and precision.
    '''
    # Unpack the prior
    (mu, lambda0, alpha, beta) = prior

    # Sample scaling tau from a gamma distribution
    tau = np.random.gamma(shape=alpha, scale=1. / beta)
    var = 1. / (lambda0 * tau)
    # Sample mean from normal mean mu, var
    mean = np.random.normal(loc=mu, scale=np.sqrt(var))
    return (mean, tau)


#---------------------------------------------------------------------------
# Transition functions

def update_dirichlet(prior, data):
    '''
    Update the parameters of a dirichlet distribution.
    We assume that the data is drawn from multinomial over n discrete states.

    Args:
        prior - n x 1 - numpy array, pseudocounts of discrete observations.
        data - n x 1 - numpy array, counts of observations of each draw

    Returns:
        posterior - n x 1 - numpy array, overall pseudocounts.
    '''
    # Updating dirichlet is trivial
    posterior = prior + data
    return posterior

def sample_dirichlet(prior):
    '''
    Sample a multinomial distribution from a Dirichlet prior.

    Args:
        prior - n x 1 - numpy array, pseudocounts of discrete observations.

    Returns:
        dist - n x 1 - numpy array, probability distribution over n discrete.
    '''
    n = len(prior)
    dist = np.zeros(n)
    for i in range(n):
        # Sample a gamma distribution for each entry
        dist[i] = np.random.gamma(prior[i])

    # Normalize the probability distribution
    dist = dist / sum(dist)
    return dist




'''
Example use of posterior_sampling.py

We show how to update the priors and sample from the posterior.
You may be able to improve performance through vectorization or using existing
compiled libraries for computation.

author: iosband@stanford.edu
'''

import numpy as np
import posterior_sampling

#-----------------------------------------------------------------------
# Updating rewards (normal gamma)

# Make a very simple prior
mu = 0.
n_mu = 1.
tau = 1.
n_tau = 1.

# Convert it to our nice format
prior_ng = posterior_sampling.convert_prior(mu, n_mu, tau, n_tau)

# Generate some real data
real_mu = 1.
real_prec = 4.
n_data = 100

data = np.zeros(n_data)
for i in range(n_data):
    data[i] = np.random.normal(real_mu, np.sqrt(1. / real_prec))

print 'True Normal distribution: ' + str((real_mu, real_prec)) + '\n'

# Sampled data from the posterior
posterior_ng = posterior_sampling.update_normal_ig(prior_ng, data)
n_samp = 10
for i in range(n_samp):
    sample_norm = posterior_sampling.sample_normal_ig(posterior_ng)
    print 'Sampled Normal distribution: ' + str(sample_norm)

print '\n \n '

#---------------------------------------------------------------------
# Updating transitions

# Make a very simple prior
n_state = 5
prior_dir = np.ones(n_state)

# Imagine we have observed the following
p_true = np.random.gamma(shape=1, size=n_state)
p_true = p_true / np.sum(p_true)
n_data = 100
counts = np.random.multinomial(n_data, p_true)

print 'True multinomial distribution: ' + str(p_true) + '\n'

# Sample data from the posterior
posterior_dir = posterior_sampling.update_dirichlet(prior_dir, counts)
n_samp = 10
for i in range(n_samp):
    sample_mult = posterior_sampling.sample_dirichlet(posterior_dir)
    print 'Sampled multinomial distribution: ' + str(sample_mult)

print '\n'

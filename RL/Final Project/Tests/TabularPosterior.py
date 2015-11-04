"""Tabular representation"""

from rlpy.Representations.Representation import Representation
import numpy as np
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Imanol Arrieta Ibarra"


class TabularPosterior(Representation):
    """
    Tabular representation that assigns a binary feature function f_{d}() 
    to each possible discrete state *d* in the domain. (For bounded continuous
    dimensions of s, discretize.)
    f_{d}(s) = 1 when d=s, 0 elsewhere.  (ie, the vector of feature functions
    evaluated at *s* will have all zero elements except one).
    NOTE that this representation does not support unbounded dimensions
    
    Additionally, this representation has two types of weights. 

    """
    observed = None
    observed_rewards = None
    observed_transitions = None

    def __init__(self, domain, discretization=20):
        # Already performed in call to superclass
        self.setBinsPerDimension(domain, discretization)
        self.features_num = int(np.prod(self.bins_per_dim))
        super(TabularPosterior, self).__init__(domain, discretization)
        self.observed_rewards = {}
        self.observed = np.zeros((self.features_num,self.actions_num))
        self.observed_transitions = np.zeros((self.features_num,self.actions_num,self.agg_states_num))

    def phi_nonTerminal(self, s):
        hashVal = self.hashState(s)
        F_s = np.zeros(self.agg_states_num, bool)
        F_s[hashVal] = 1
        return F_s

    def featureType(self):
        return bool
    
    def location(self,s,a,ns):
        hashVal =self.hashState(s)
        hashVal_prime = self.hashState(ns)
        i,j = hashVal, hashVal_prime
        return i,j
    

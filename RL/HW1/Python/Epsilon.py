# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:48:42 2015

@author: Imanol
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:55:46 2015

@author: Imanol
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:53:57 2015

@author: Imanol
"""
import numpy as np
import posterior_sampling
import random


def Reward(s,a,H):
    r = 0
    if s==H:
        r=1
    return r

def NewState(s,a,H):
    return max(min(s+a,H),1)

#--------------------------------------------------------------------------
def policy(R,P,Rl,Pl,S,A,H):
    """
    Computes the policy for each state action pair (s,a). The policy is 
    computed as a greedy action taken over a upper confidence bound Value 
    Function.
    
    IN
    
    R: dict{int}
        Dictionary containing the sum of observed rewards for each tupke
        (t,s,a).
    P: dict{np.array}
        Dictionary of arrays of dimension |S|x1 containing the probability 
        of going to state s' given the tuple (t,s,a).
    Rl: dict{int}
        Sampled mean from posterior distribution
    Pl: dict{int}
        Sampled transition probabilities from posterior distribution
    S: list
        States
    A: list
        Actions
    H: int
        Number of states and time frame
    """
    
    # We start by creating two dictionaries that will store the value function
    # and the optimal policy
    Q = {}
    mu = {}
    c1 = H
    # We initialize Q and mu at time H-1
    # For every state-action pair the value function is equal to the sampled   
    # reward plus the sampled expected to go cost.
    # Qmax is an array which will store the maximum value of Q for each state.
    Qmax = np.zeros((c1))
    for t in xrange(H-1,-1,-1):
        new_Qmax = np.zeros((c1))
        for s in S:
            Q = []
            for a in A:
                # We call the linearprogram function which solves the MDP for
                # s,a 
                Q.append(Rl[(t,s,a)] + np.sum(Qmax*Pl[(t,s,a)]))
                                        
            new_Qmax[s-1] = np.max(Q)
            mu[(t,s)] = np.argmax(Q)*2-1
            
        Qmax = new_Qmax
                
    return mu
            
def play(mu,H,R,P,A,eps):
    
        
    s = 1
    rew = 0
    for t in xrange(H):
        samp = random.random()
        if samp<eps:
            a = random.choice(A)
        else:
            a = mu[(t,s)]
        #print 'Action',a
        r = Reward(s,a,H)
        rew+=r
        n_s = NewState(s,a,H)
        #print 'New State',n_s
        R[(t,s,a)].append(r)
        P[(t,s,a)][n_s-1]= P[(t,s,a)][n_s-1]+1
        s = n_s
    return rew
        

def PCRL(S,A,H,d,L,eps):
    
    # Make a very simple prior
    mu = 0.
    n_mu = 1.
    tau = 1.
    n_tau = 1.
    prior_ng = posterior_sampling.convert_prior(mu, n_mu, tau, n_tau)
    rew=0
    av_rew=[]
    c1 = len(S)
    prior_dir = np.ones(c1)
    R = {}
    P = {}
    time = range(H);
    for l in xrange(L):
        Rl = {}
        Pl = {}
        for t in time:
            for s in S:
                for a in A:
                    if (t,s,a) not in R:
                        R[(t,s,a)]=[]
                        P[(t,s,a)]=np.array([0 for i in xrange(c1)])
                    if len(R[(t,s,a)]) == 0:
                        Rpost = prior_ng
                        Ppost = prior_dir
                    else:
                        data = np.array(R[(t,s,a)])
                        counts = P[(t,s,a)]
                        Rpost = posterior_sampling.update_normal_ig(prior_ng, data)
                        Ppost = posterior_sampling.update_dirichlet(prior_dir, counts)

                    Rl[(t,s,a)]= posterior_sampling.sample_normal_ig(Rpost)[0]
                    Pl[(t,s,a)]= posterior_sampling.sample_dirichlet(Ppost)

        mu = policy(R,P,Rl,Pl,S,A,H)
        rew +=  play(mu,H,R,P,A,eps)
        
        if (l+1)%10==0:
            av_rew.append(rew/float(10))
            rew = 0
    return av_rew

        
if __name__ == '__main__':
    H = 9
    S = range(1,H+1)
    A = [-1,1]
    d = 1
    L = 1000
    eps = .05
    succe=[]
    for H in xrange(1,16):
        S = range(1,H+1)
        succe.append( PCRL(S,A,H,d,L,eps))
    linese = [[succe[i][j] for i in xrange(10,15)] for j in xrange(100)]
    plt.plot(linese)
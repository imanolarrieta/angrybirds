# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:55:46 2015

@author: Imanol
This code implements PSRL for the chain problem.
1-2-3-4-5-...-H
Where there is a positive reward only at the last state. 
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:53:57 2015

@author: Imanol
"""
import numpy as np
import posterior_sampling


#---------------------------------------------------------------------------
def Reward(s,a,H):
    """
    Computes the real reward which is one only if the state is the 
    last one in chain
    
    IN
    s: int
        Current State.
    a: int
        Action taken
    H: int
        Time frame and number of States
    OUT
    r: real reward
    """
    r = 0
    if s==H:
        r=1
    return r
#---------------------------------------------------------------------------
def NewState(s,a,H):
    """
    Computes the next state given the current state and the action taken.
    The next state is computed by adding the action to the current state. 
    Except for the beginning and ending of the chain where if the action 
    results in a non existent state, the agent stays in the current state.
    
    IN
    s: int
        Current State.
    a: int
        Action taken
    H: int
        Time frame and number of States
        
    OUT
    newstate: int
        New State
    """
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
    
 #---------------------------------------------------------------------------         
def play(mu,H,R,P):
    """
    Runs an episode given the optimal policy mu. The function updates
    the dictionaries R, P, Lambda registering the new visited states and 
    the corresponding rewards.
    
    IN
    mu: dict
        Dictionary of optimal actions acording to (t,s) pair.
    H: int
        Number of states or times.
    R: dict{list}
        Dictionary containing a list of observed rewards for each tuple
        (t,s,a).
    P: dict{np.array}
        Dictionary of arrays of dimension |S|x1 containing the probability 
        of going to state s' given the tuple (t,s,a).
    
    """
    
    s = 1
    rew = 0

    for t in xrange(H):
        a = mu[(t,s)]
        r = Reward(s,a,H)
        rew+=r
        n_s = NewState(s,a,H)
        R[(t,s,a)].append(r)
        P[(t,s,a)][n_s-1]= P[(t,s,a)][n_s-1]+1
        s = n_s
    return rew
        
#---------------------------------------------------------------------------            


def PSRL(S,A,H,L):
    """
    Computes the number of episodes it takes for PSRL to experience a positive 
    reward.
    
    IN
    S: list
        States
    A: list
        Actions
    H: int
        Number of states and time frame
    L: int
        Number of episodes
    OUT 
    success:  int
        Number of episodes before UCRL experiences a positive reward.
    
    """
    # Make a very simple prior
    mu = 0.
    n_mu = 1.
    tau = 1.
    n_tau = 1.
    prior_ng = posterior_sampling.convert_prior(mu, n_mu, tau, n_tau)
    
    c1 = len(S)
    prior_dir = np.ones(c1)
    R = {}
    P = {}
    time = range(H);
    av_rew = []
    rew = 0
    for l in xrange(L):
        Rl = {}
        Pl = {}
        for t in time:
            for s in S:
                for a in A:
                    if (t,s,a) not in R:
                        R[(t,s,a)]=[]
                        P[(t,s,a)]=np.array([0 for i in xrange(c1)])
                    # If we have not visited (t,s,a) we don't update our prior
                    if len(R[(t,s,a)]) == 0:
                        Rpost = prior_ng
                        Ppost = prior_dir
                    else:
                        data = np.array(R[(t,s,a)])
                        counts = P[(t,s,a)]
                        # Posterior updating
                        Rpost = posterior_sampling.update_normal_ig(prior_ng, data)
                        Ppost = posterior_sampling.update_dirichlet(prior_dir, counts)
                    # Posterior sampling
                    Rl[(t,s,a)]= posterior_sampling.sample_normal_ig(Rpost)[0]
                    Pl[(t,s,a)]= posterior_sampling.sample_dirichlet(Ppost)
        # Optimal policy
        mu = policy(R,P,Rl,Pl,S,A,H)
        #Episode
        
        rew += play(mu,H,R,P)
        
        if (l+1)%10==0:
            print rew/float(10)
            av_rew.append(rew/float(10))
            rew = 0
    return av_rew
        
        
if __name__ == '__main__':
    H = 10
    S = range(1,H+1)
    A = [-1,1]
    L = 10000
    succp = []
    succp = PSRL(S,A,H,L)
    #for H in xrange(1,15):
       # S = range(1,H+1)
        #succp.append(PSRL(S,A,H,L))
    #linesp = [[succp[i][j] for i in xrange(9,14)] for j in xrange(100)]
    #plt.plot(linesp)
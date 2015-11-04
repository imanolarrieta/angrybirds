# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:53:57 2015

@author: Imanol

This code implements UCRL for the chain problem.
1-2-3-4-5-...-H
Where there is a positive reward only at the last state. 
"""
import numpy as np
import scipy.optimize as op
import random
import matplotlib.pyplot as plt
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
def linearprogram(Qmax,c1,P,R,Pl,Rl,Lambda):
    """
    Computes the Linear Program:
    max R(s,a) + sum(P(s'|s,a)*Qmax(s'))
    Over R and P inside confidence sets.
    
    IN
    Qmax: np.array
        Array containing the maximum value of the cost to go function
        for every state from the previous iteration.
    c1: int
        Number of states
    P: np.array
        sum of times s' has been observed from (t,s,a)
    R: int
        sum of rewards observed at (t,s,a)
    Pl: int
        upper confidence bound for transition probabilities
    Rl: int 
        upper confidence bound for rewards
    Lambda: int
        number of times (t,s,a) has been observed
    
    OUT 
    
    max_value: int
        max R(s,a) + sum(P(s'|s,a)*Qmax(s'))
    """
    worse_states = list(np.argsort(Qmax+np.random.rand(c1)*0.00001))
    new_prob = np.array(P/max(float(Lambda),1.0))
    best_state = worse_states.pop()
    new_prob[best_state] = min(1,new_prob[best_state]+Pl/2.0)
    i=0
    
    while sum(new_prob)>1 and i<len(worse_states):
        state = worse_states[i]
        prob_update = max(0,1-sum(new_prob)+new_prob[state])
        new_prob[state] = prob_update
        i+=1
  
    max_value = np.sum(new_prob*Qmax) + Rl +R/max(float(Lambda),1.0)
   
    
    return max_value    
    

#---------------------------------------------------------------------------
def policy(R,P,Lambda,Rl,Pl,S,A,H):
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
    Lambda: dict{int}
        Dictionary  containing the number of times the tuple (t,s,a) has been
        visited
    Rl: dict{int}
        Upper bound for the reward for tuple (t,s,a)
    Pl: dict{int}
        Upper bound for the transition probabilities for tuple (t,s,a)
    S: list
        States
    A: list
        Actions
    H: int
        Number of states and time frame
    """
    
    # We start by creating two dictionaries that will store the value function
    # and the optimal policy
    mu = {}
    c1 = H
    # We initialize Q and mu at time H-1
    # For every state-action pair the value function is equal to the maximum
    # over the bounded rewards set given 
    # Qmax is an array which will store the maximum value of Q for each state.
    Qmax = np.zeros((c1))
    for t in xrange(H-1,-1,-1):
        new_Qmax = np.zeros((c1))
        for s in S:
            Q = []
            for a in A:
                # We call the linearprogram function which solves the MDP for
                # s,a 
                Q.append(linearprogram(Qmax,c1,P[(t,s,a)],
                                        R[(t,s,a)],Pl[(t,s,a)],
                                        Rl[(t,s,a)],Lambda[(t,s,a)]))
                                        
            new_Qmax[s-1] = np.max(Q)
            mu[(t,s)] = np.argmax(Q)*2-1
            
        Qmax = new_Qmax
                
    return mu
 #---------------------------------------------------------------------------         
def play(mu,H,R,P,Lambda):
    """
    Runs an episode given the optimal policy mu. The function updates
    the dictionaries R, P, Lambda registering the new visited states and 
    the corresponding rewards.
    
    IN
    mu: dict
        Dictionary of optimal actions acording to (t,s) pair.
    H: int
        Number of states or times.
    R: dict{int}
        Dictionary containing the sum of observed rewards for each tupke
        (t,s,a).
    P: dict{np.array}
        Dictionary of arrays of dimension |S|x1 containing the probability 
        of going to state s' given the tuple (t,s,a).
    Lambda: dict{int}
        Dictionary  containing the number of times the tuple (t,s,a) has been
        visited
    
    """
    
    s = 1
    rew = 0
    for t in xrange(H):
        a = mu[(t,s)]
        r = Reward(s,a,H)
        rew +=r
        n_s = NewState(s,a,H)
        R[(t,s,a)] = R[(t,s,a)]+r
        P[(t,s,a)][n_s-1]= P[(t,s,a)][n_s-1]+1
        Lambda[(t,s,a)] = Lambda[(t,s,a)] +1
        s = n_s
    return rew
    
#---------------------------------------------------------------------------
def UCRL(S,A,H,d,L):
    """
    Computes the number of episodes it takes for UCRL to experience a positive 
    reward.
    
    IN
    S: list
        States
    A: list
        Actions
    H: int
        Number of states and time frame
    d: double
        precision delta
    L: int
        Number of episodes
    OUT 
    success:  int
        Number of episodes before UCRL experiences a positive reward.
    
    """
    
    c1 = len(S)
    c2 = len(A)
    
    # P and R will be dictionaries that will store the experienced transitions 
    # and rewards. Lambda will store the number of times a tuple (t,s,a ) is 
    # visited.
    R = {}
    P = {}
    Lambda = {}
    time = range(H)
    av_rew = []
    rew = 0
    for l in xrange(L):
        # Rl and Pl will store the upper confidence bounds for the rewards 
        # and the transition probabilities
        Rl = {}
        Pl = {}
        for t in time:
            for s in S:
                for a in A:
                    # Initialization of the dictionaries
                    if (t,s,a) not in R:
                        R[(t,s,a)]=0.0
                        P[(t,s,a)]=np.zeros((c1))
                        Lambda[(t,s,a)]=0.0
                    # Confidence bounds
                    nl = Lambda[(t,s,a)]
                    Rl[(t,s,a)] = np.sqrt(7*np.log(2*c1*c2*H*(l+1)/float(d))/float(2*max(nl,1)))
                    Pl[(t,s,a)] = np.sqrt(14*c1*np.log(2*c1*c2*H*(l+1)/float(d))/float(max(nl,1)))
        # Optimal policy
        mu = policy(R,P,Lambda,Rl,Pl,S,A,H)
        # Episode
        rew += play(mu,H,R,P,Lambda)
        
        if (l+1)%10==0:
            av_rew.append(rew/float(10))
            rew = 0
    return av_rew

            
    

#---------------------------------------------------------------------------
        
if __name__ == '__main__':
    """
    Main Function
    """
    H = 2
    S = range(1,H+1)
    A = [-1,1]
    d = .05
    L = 1000
    succ = []
    for H in xrange(1,15):
        S = range(1,H+1)
        succ.append(UCRL(S,A,H,d,L))
    lines = [[succ[i][j] for i in xrange(14)] for j in xrange(100)]
    plt.plot(lines)
    
    

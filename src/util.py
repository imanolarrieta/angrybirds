'''
Code for simulating reinforcement learning algorithm.

original author: Percy Liang, pliang@cs.stanford.edu
Based on Stanford University's Artificial Intelligence (CS221) course material

Adapted to mini-batch (episodic) learning by Bernardo Ramos, bramos@stanford.edu
'''

import copy

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,show = False, episodicLearning=True):

    output = {}
    totalRewards = []  # The rewards we get on each trial
    levelsPassed = []
    if show: mdp.showLearning()
    for trial in range(numTrials):
        state = mdp.startState()
        if verbose: mdp.showState()
        sequence = [state]
        totalReward = 0
        maxLevel = 0
        stateHistory = []
        actionHistory = []
        rewardHistory = []
        newStateHistory = []

        for k in range(maxIterations):
            action = rl.getAction(state)
            newState,reward = mdp.succAndReward(state, action)

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)
            if not episodicLearning:
                rl.incorporateFeedback(state, action, reward, newState)
            else:
                stateHistory.append(copy.deepcopy(state))
                actionHistory.append(copy.deepcopy(action))
                rewardHistory.append(copy.deepcopy(reward))
                newStateHistory.append(copy.deepcopy(newState))

            totalReward +=  reward
            if newState==None:
                break
            maxLevel = max(maxLevel, newState.getLevel())
            if verbose: mdp.showState()
            state = newState

        if episodicLearning:   #Incorporate feedback after the episode is done
            for state, action, reward, newState in zip(stateHistory, actionHistory, rewardHistory, newStateHistory):
                rl.incorporateFeedback(state, action, reward, newState)

        if verbose:
            print("Trial %d (totalReward = %s, maxLevel = %s): %s" % (trial, totalReward,maxLevel, sequence))
        totalRewards.append(totalReward)
        levelsPassed.append(maxLevel)
        output['totalRewards'] = totalRewards
        output['levelsPassed'] = levelsPassed
    return output
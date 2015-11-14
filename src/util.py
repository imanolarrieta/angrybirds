# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,show = False):

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

        for _ in range(maxIterations):
            action = rl.getAction(state)
            newState,reward = mdp.succAndReward(state, action)

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward +=  reward
            if newState==None:
                break
            maxLevel = max(maxLevel, newState.getLevel())
            if verbose: mdp.showState()
            state = newState

        if verbose:
            print("Trial %d (totalReward = %s, maxLevel = %s): %s" % (trial, totalReward,maxLevel, sequence))
        totalRewards.append(totalReward)
        levelsPassed.append(maxLevel)
        output['totalRewards'] = totalRewards
        output['levelsPassed'] = levelsPassed
    return output
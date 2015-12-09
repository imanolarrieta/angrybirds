from QLearner import QLearningAlgorithm
from AngryBirds import AngryBirdsGame
import math
from abAPI import *
from util import *
from GameAgent import angryAgent
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
if __name__ == '__main__':

    ab = AngryBirdsMDP()
    explorationProb = 0.3
    numTrials = 50
    trial = 0
    levelsPassed=[]
    totalRewards =[]

    agent = angryAgent()
    featureExtractor =agent.custom1FeatureExtractor


    rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
                            explorationProb=explorationProb)

    while trial<numTrials:
        rl.explorationProb = explorationProb
        simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
        rl.explorationProb = 0.0
        outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
        levelsPassed = levelsPassed + outcome['levelsPassed']
        totalRewards = totalRewards + outcome['totalRewards']
        trial +=1



    print(levelsPassed)
    print(totalRewards)
    trials = range(1,numTrials+1)


    plt.plot(trials,levelsPassed,lw=2,color='blue')
    plt.scatter(trials,levelsPassed)
    plt.xlabel('Number of trials',fontsize='large')
    plt.ylabel('Number of levels passed',fontsize='large')
    plt.title('Rewards per trials', fontsize=20)
    maxLevel=0
    for i in range(numTrials):
        if levelsPassed[i]>maxLevel:
            maxLevel=levelsPassed[i]
            plt.axvline(i+1,color='r', linestyle='--')
    plt.savefig('levelsPassed.png')

    plt.figure()
    plt.plot(trials,totalRewards,lw=2,color='green')
    plt.scatter(trials,totalRewards)
    plt.xlabel('Number of trials',fontsize='large')
    plt.ylabel('Number of total rewards',fontsize='large')
    plt.title('Rewards per trials', fontsize=20)
    maxLevel=0
    for i in range(numTrials):
        if levelsPassed[i]>maxLevel:
            maxLevel=levelsPassed[i]
            plt.axvline(i+1,color='r', linestyle='--')
    plt.savefig('totalRewards.png')

# ############################################# Evaluation of Level 3
#     ab = AngryBirdsMDP(level = 3)
#     explorationProb = 0.3
#     numTrials = 50
#     agent = angryAgent()
#
#     rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
#                             explorationProb=explorationProb)
#     trainingOutcomes = simulate(ab,rl,numTrials=numTrials, maxIterations=1000, verbose=False, show=False)
#
#     explorationProb = 0
#     agent = angryAgent()
#     rl.setExplorationProb(explorationProb)
#     testOutcomes = simulate(ab,rl,numTrials=5,verbose = False, show=False)
#
#     print(trainingOutcomes)
#     print(testOutcomes)
#
#     totalRewards = trainingOutcomes['totalRewards']
#     trials = range(1,numTrials+1)
#
#     plt.figure()
#     plt.plot(trials,totalRewards,lw=2,color='green')
#     plt.scatter(trials,totalRewards)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of total rewards for level 3',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     plt.savefig('totalRewardsL3.png')
#
# ############################################# Evaluation of Level 5
#     ab = AngryBirdsMDP(level = 5)
#     explorationProb = 0.3
#     numTrials = 50
#     agent = angryAgent()
#
#     rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
#                             explorationProb=explorationProb)
#     trainingOutcomes = simulate(ab,rl,numTrials=numTrials, maxIterations=1000, verbose=False, show=False)
#
#     explorationProb = 0
#     agent = angryAgent()
#     rl.setExplorationProb(explorationProb)
#     testOutcomes = simulate(ab,rl,numTrials=5,verbose = False, show=False)
#
#     print(trainingOutcomes)
#     print(testOutcomes)
#     totalRewards = trainingOutcomes['totalRewards']
#     trials = range(1,numTrials+1)
#
#
#     plt.figure()
#     plt.plot(trials,totalRewards,lw=2,color='green')
#     plt.scatter(trials,totalRewards)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of total rewards for level 5',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     plt.savefig('totalRewardsL5.png')
#
# ############################################# Evaluation of Level 5 while training in 3
#     ab = AngryBirdsMDP(level = 7)
#     explorationProb = 0.3
#     numTrials = 50
#     agent = angryAgent()
#
#     rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
#                             explorationProb=explorationProb)
#     trainingOutcomes = simulate(ab,rl,numTrials=numTrials, maxIterations=1000, verbose=False, show=False)
#
#     explorationProb = 0
#     ab = AngryBirdsMDP(level = 7)
#     agent = angryAgent()
#     rl.setExplorationProb(explorationProb)
#     testOutcomes = simulate(ab,rl,numTrials=5,verbose = False  , show=False)
#
#     print(trainingOutcomes)
#     print(testOutcomes)
#     totalRewards = trainingOutcomes['totalRewards']
#     trials = range(1,numTrials+1)
#
#
#     plt.figure()
#     plt.plot(trials,totalRewards,lw=2,color='green')
#     plt.scatter(trials,totalRewards)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of total rewards for level 7',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     plt.savefig('totalRewardsL7.png')
#

from QLearner import QLearningAlgorithm
from AngryBirds import AngryBirdsGame
import math
from abAPI import *
from util import *
from GameAgent import angryAgent
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from LearnerRLSVI import RLSVI_wrapper
import pickle

matplotlib.use('TkAgg')
if __name__ == '__main__':


########################################################### Q-Learning original features
    ab = AngryBirdsMDP()
    explorationProb = 0.3
    numTrials = 50
    trial = 0
    levelsPassed=[]
    totalRewards =[]

    agent = angryAgent()
    featureExtractor =agent.featureExtractorXYaction


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


    levelsPassed = np.cumsum(levelsPassed)/range(1,len(levelsPassed)+1)
    totalRewards = np.cumsum(totalRewards)/range(1,len(totalRewards)+1)


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
    plt.savefig('levelsPassed_Q_original.png')

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
    plt.savefig('totalRewards_Q_original.png')

    with open('levelsPassed_Q_original','wb') as file:
        pickle.dump(levelsPassed,file)
    with open('totalRewards_Q_original','wb') as file:
        pickle.dump(totalRewards,file)

# ########################################################### Q-Learning original features
#     ab = AngryBirdsMDP()
#     explorationProb = 0.3
#     numTrials = 50
#     trial = 0
#     levelsPassed=[]
#     totalRewards =[]
#
#     agent = angryAgent()
#     featureExtractor =agent.featureExtractorXYaction
#
#
#     rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
#                             explorationProb=explorationProb)
#
#     while trial<numTrials:
#         rl.explorationProb = explorationProb
#         simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
#         rl.explorationProb = 0.0
#         outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
#         levelsPassed = levelsPassed + outcome['levelsPassed']
#         totalRewards = totalRewards + outcome['totalRewards']
#         trial +=1
#
#
#     levelsPassed = np.cumsum(levelsPassed)/range(1,len(levelsPassed)+1)
#     totalRewards = np.cumsum(totalRewards)/range(1,len(totalRewards)+1)
#
#
#     print(levelsPassed)
#     print(totalRewards)
#     trials = range(1,numTrials+1)
#
#
#     plt.plot(trials,levelsPassed,lw=2,color='blue')
#     plt.scatter(trials,levelsPassed)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of levels passed',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     maxLevel=0
#     for i in range(numTrials):
#         if levelsPassed[i]>maxLevel:
#             maxLevel=levelsPassed[i]
#             plt.axvline(i+1,color='r', linestyle='--')
#     plt.savefig('levelsPassed_Q_original_obstacles.png')
#
#     plt.figure()
#     plt.plot(trials,totalRewards,lw=2,color='green')
#     plt.scatter(trials,totalRewards)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of total rewards',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     maxLevel=0
#     for i in range(numTrials):
#         if levelsPassed[i]>maxLevel:
#             maxLevel=levelsPassed[i]
#             plt.axvline(i+1,color='r', linestyle='--')
#     plt.savefig('totalRewards_Q_original_obstacles.png')
#
#     pickle.dump(levelsPassed,'levelsPassed_Q_nested_obstacles')
#     pickle.dump(totalRewards,'totalRewards_Q_nested_obstacles')
# ########################################################### Q-Learning nested features
#     ab = AngryBirdsMDP()
#     explorationProb = 0.3
#     numTrials = 50
#     trial = 0
#     levelsPassed=[]
#     totalRewards =[]
#
#     agent = angryAgent()
#     featureExtractor =agent.nestedGridFeatureExtractor
#
#
#     rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
#                             explorationProb=explorationProb)
#
#     while trial<numTrials:
#         rl.explorationProb = explorationProb
#         simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
#         rl.explorationProb = 0.0
#         outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
#         levelsPassed = levelsPassed + outcome['levelsPassed']
#         totalRewards = totalRewards + outcome['totalRewards']
#         trial +=1
#
#
#     levelsPassed = np.cumsum(levelsPassed)/range(1,len(levelsPassed)+1)
#     totalRewards = np.cumsum(totalRewards)/range(1,len(totalRewards)+1)
#
#
#     print(levelsPassed)
#     print(totalRewards)
#     trials = range(1,numTrials+1)
#
#
#     plt.plot(trials,levelsPassed,lw=2,color='blue')
#     plt.scatter(trials,levelsPassed)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of levels passed',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     maxLevel=0
#     for i in range(numTrials):
#         if levelsPassed[i]>maxLevel:
#             maxLevel=levelsPassed[i]
#             plt.axvline(i+1,color='r', linestyle='--')
#     plt.savefig('levelsPassed_Q_nested.png')
#
#     plt.figure()
#     plt.plot(trials,totalRewards,lw=2,color='green')
#     plt.scatter(trials,totalRewards)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of total rewards',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     maxLevel=0
#     for i in range(numTrials):
#         if levelsPassed[i]>maxLevel:
#             maxLevel=levelsPassed[i]
#             plt.axvline(i+1,color='r', linestyle='--')
#     plt.savefig('totalRewards_Q_nested.png')
#
#     pickle.dump(levelsPassed,'levelsPassed_Q_nested')
#     pickle.dump(totalRewards,'totalRewards_Q_nested')
#
# ######################################################################### RLSVI
#     ab = AngryBirdsMDP()
#     explorationProb = 0.3
#     numTrials = 50
#     trial = 0
#     levelsPassed=[]
#     totalRewards =[]
#
#     agent = angryAgent()
#     featureExtractor =agent.featureExtractorXYaction
#
#
#     # rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
#     #                         explorationProb=explorationProb)
#     rl = RLSVI_wrapper(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor)
#
#     while trial<numTrials:
#         try:
#             print (trial)
#             rl.rlsvi.sigma = 10000
#             simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
#             rl.rlsvi.sigma = 1.0
#             outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
#             levelsPassed = levelsPassed + outcome['levelsPassed']
#             totalRewards = totalRewards + outcome['totalRewards']
#             trial +=1
#         except AssertionError:
#             break
#
#
#     levelsPassed = np.cumsum(levelsPassed)/range(1,len(levelsPassed)+1)
#     totalRewards = np.cumsum(totalRewards)/range(1,len(totalRewards)+1)
#
#
#     print(levelsPassed)
#     print(totalRewards)
#     trials = range(1,numTrials+1)
#
#
#     plt.plot(trials,levelsPassed,lw=2,color='blue')
#     plt.scatter(trials,levelsPassed)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of levels passed',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     maxLevel=0
#     for i in range(numTrials):
#         if levelsPassed[i]>maxLevel:
#             maxLevel=levelsPassed[i]
#             plt.axvline(i+1,color='r', linestyle='--')
#     plt.savefig('levelsPassed_RLSVI_original.png')
#
#     plt.figure()
#     plt.plot(trials,totalRewards,lw=2,color='green')
#     plt.scatter(trials,totalRewards)
#     plt.xlabel('Number of trials',fontsize='large')
#     plt.ylabel('Number of total rewards',fontsize='large')
#     plt.title('Rewards per trials', fontsize=20)
#     maxLevel=0
#     for i in range(numTrials):
#         if levelsPassed[i]>maxLevel:
#             maxLevel=levelsPassed[i]
#             plt.axvline(i+1,color='r', linestyle='--')
#     plt.savefig('totalRewards_RLSVI_original.png')
#
#     pickle.dump(levelsPassed,'levelsPassed_RLSVI_original')
#     pickle.dump(totalRewards,'totalRewards_RLSVI_original')
#

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

from QLearner import QLearningAlgorithm
from AngryBirds import AngryBirdsGame
import math
from abAPI import *
from util import *
from GameAgent import angryAgent
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from sparseLearnerRLSVI import RLSVI_wrapper as RLSVI
import pickle

matplotlib.use('TkAgg')




def evaluator(rlAlgorithm,featureExtractor,nameAlg,nameFeat,numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500):
    ab = AngryBirdsMDP()
    trial = 0
    levelsPassed=[]
    totalRewards =[]
    name = nameAlg+'_'+nameFeat+'_'+str(numFeat)

    agent = angryAgent()


    rl = rlAlgorithm(actions=lambda x: agent.getAngryBirdsActions(x,numFeat),featureExtractor=featureExtractor, epsilon = epsilon)

    if nameAlg == 'Q':
        while trial<numTrials:
            print(trial)
            rl.epsilon = epsilon
            simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
            rl.epsilon = 0.0
            outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
            levelsPassed = levelsPassed + outcome['levelsPassed']
            totalRewards = totalRewards + outcome['totalRewards']
            trial +=1

    elif nameAlg == 'LSVI':
        while trial<numTrials:
            try:
                print (trial)
                rl.makeLSVI(epsilon)
                simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                rl.makeLSVI(0.0)
                outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                levelsPassed = levelsPassed + outcome['levelsPassed']
                totalRewards = totalRewards + outcome['totalRewards']
                trial +=1
            except AssertionError:
                break
    elif nameAlg == 'RLSVI':
        while trial<numTrials:
            try:
                print (trial)
                rl.makeRSVI()
                simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                rl.makeLSVI(0.0)
                outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                levelsPassed = levelsPassed + outcome['levelsPassed']
                totalRewards = totalRewards + outcome['totalRewards']
                trial +=1
            except AssertionError:
                break

        while trial<numTrials:
            try:
                print (trial)
                rl.rlsvi.sigma = sigma
                simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                rl.rlsvi.sigma = 1.0
                outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                levelsPassed = levelsPassed + outcome['levelsPassed']
                totalRewards = totalRewards + outcome['totalRewards']
                trial +=1
            except AssertionError:
                break




    cumlevelsPassed = np.cumsum(levelsPassed)/range(1,len(levelsPassed)+1)
    cumtotalRewards = np.cumsum(totalRewards)/range(1,len(totalRewards)+1)

    with open('../results/levelsPassed_'+name,'wb') as file:
        pickle.dump(levelsPassed,file)
    with open('../results/cumlevelsPassed_'+name,'wb') as file:
        pickle.dump(cumlevelsPassed,file)
    with open('../results/totalRewards_'+name,'wb') as file:
        pickle.dump(totalRewards,file)
    with open('../results/cumtotalRewards_'+name,'wb') as file:
        pickle.dump(cumtotalRewards,file)

    print('Max Total Rewards _'+name,max(totalRewards))

    trials = range(1,len(cumlevelsPassed)+1)

    plt.figure()

    plt.plot(trials,cumlevelsPassed,lw=2,color='blue')
    plt.scatter(trials,cumlevelsPassed)
    plt.xlabel('Number of trials',fontsize='large')
    plt.ylabel('Number of levels passed',fontsize='large')
    plt.title('Rewards per trials', fontsize=20)
    maxLevel=0
    for i in range(len(cumlevelsPassed)):
        if levelsPassed[i]>maxLevel:
            maxLevel=levelsPassed[i]
            plt.axvline(i+1,color='r', linestyle='--')
    plt.savefig('../plots/levelsPassed__'+name+'.png')

    plt.figure()
    plt.plot(trials,cumtotalRewards,lw=2,color='green')
    plt.scatter(trials,cumtotalRewards)
    plt.xlabel('Number of trials',fontsize='large')
    plt.ylabel('Number of total rewards',fontsize='large')
    plt.title('Rewards per trials', fontsize=20)
    maxLevel=0
    for i in range(len(cumlevelsPassed)):
        if levelsPassed[i]>maxLevel:
            maxLevel=levelsPassed[i]
            plt.axvline(i+1,color='r', linestyle='--')
    plt.savefig('../plots/totalRewards_'+name+'.png')

def level_evaluator(level,rlAlgorithm,featureExtractor,nameAlg,nameFeat,numFeat = 64,numTrials=50, explorationProb = 0.3, sigma = 500):

    ab = AngryBirdsMDP(level = level)

    rl = rlAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,discount=ab.discount(),\
                            explorationProb=explorationProb)
    trainingOutcomes = simulate(ab,rl,numTrials=numTrials, maxIterations=1000, verbose=False, show=False)

    explorationProb = 0.0
    if nameAlg =='Q':
        rl.setExplorationProb(explorationProb)
    elif nameAlg=='LSVI':
        rl.rlsvi.epsilon = explorationProb
    else:
        rl.rlsvi.sigma = 1.0
    testOutcomes = simulate(ab,rl,numTrials=5,verbose = False, show=False)

    print(trainingOutcomes)
    print(testOutcomes)

    totalRewards = trainingOutcomes['totalRewards']
    trials = range(1,numTrials+1)

    plt.figure()
    plt.plot(trials,totalRewards,lw=2,color='green')
    plt.scatter(trials,totalRewards)
    plt.xlabel('Number of trials',fontsize='large')
    plt.ylabel('Number of total rewards for level 3',fontsize='large')
    plt.title('Rewards per trials', fontsize=20)
    plt.savefig('totalRewardsL3.png')



if __name__ == '__main__':

    agent = angryAgent()
    # evaluator(QLearningAlgorithm,agent.featureExtractorXYaction,'Q','PP',numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500)
    # evaluator(QLearningAlgorithm,agent.nestedGridFeatureExtractor,'Q','NPP',numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500)
    # evaluator(QLearningAlgorithm,agent.custom2FeatureExtractor,'Q','NPPO',numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500)
    # evaluator(RLSVI,agent.featureExtractorXYaction,'RLSVI','PP',numFeat = 64,numTrials=50, epsilon = 0.0, sigma = 500)
    # evaluator(RLSVI,agent.nestedGridFeatureExtractor,'RLSVI','NPP',numFeat = 64,numTrials=50, epsilon = 0.0, sigma = 500)
    # evaluator(RLSVI,agent.custom2FeatureExtractor,'RLSVI','NPPO',numFeat = 64,numTrials=50, epsilon = 0.0, sigma = 500)
    evaluator(RLSVI,agent.featureExtractorXYaction,'LSVI','PP',numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(RLSVI,agent.nestedGridFeatureExtractor,'LSVI','NPP',numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(RLSVI,agent.custom2FeatureExtractor,'LSVI','NPPO',numFeat = 64,numTrials=50, epsilon = 0.3, sigma = 500)


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

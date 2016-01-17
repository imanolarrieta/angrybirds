#!/usr/bin/env python
# -*- coding: utf-8 -*-

#evaluation
#This file contains the codes for generating results and graphs by running it as __main__ and uncommenting the evaluator
# functions for each feature extractor (PP, NPP, NPPO, NPPS) and algorithm (Q-Learning, RLSVI).


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
from denseLearnerRLSVI import RLSVI_wrapper as DRLSVI

import pickle

matplotlib.use('TkAgg')




def evaluator(rlAlgorithm,featureExtractor,nameAlg,nameFeat,multiple = 1.0,numTrials=50, epsilon = 0.3, sigma = 500):
    ab = AngryBirdsMDP()
    trial = 0
    levelsPassed=[]
    totalRewards =[]
    name = nameAlg+'_'+nameFeat+'_'+str(64*multiple)

    agent = angryAgent()


    rl = rlAlgorithm(actions=lambda x: agent.getAngryBirdsActions(x,multiple),featureExtractor=featureExtractor, epsilon = epsilon)

    if nameAlg == 'Q':
        while trial<numTrials:
            print(trial)
            rl.explorationProb = epsilon
            simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
            rl.explorationProb = 0.0
            outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
            levelsPassed = levelsPassed + outcome['levelsPassed']
            totalRewards = totalRewards + outcome['totalRewards']
            print(totalRewards)
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
                print(totalRewards)

                trial +=1
            except AssertionError:
                break
    elif nameAlg == 'RLSVI' or nameAlg =='DRLSVI':
        while trial<numTrials:
            try:
                print (trial)
                rl.rlsvi.sigma = 500.0
                simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                rl.rlsvi.sigma = 500.0
                outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                levelsPassed = levelsPassed + outcome['levelsPassed']
                totalRewards = totalRewards + outcome['totalRewards']
                print(totalRewards)

                trial +=1
            except AssertionError:
                break
    elif nameAlg =='DRLSVI':
        while trial<numTrials:
            try:
                print (trial)
                rl.makeRLSVI()
                simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                rl.makeLSVI(0.0)
                outcome = simulate(ab,rl,numTrials=1, maxIterations=1000, verbose=False, show=False)
                levelsPassed = levelsPassed + outcome['levelsPassed']
                totalRewards = totalRewards + outcome['totalRewards']
                print(totalRewards)

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


def movingAverage(lst,window = 10):
    ma = []
    i = 0
    while i+window<len(lst)+1:
        ma.append((np.mean(lst[i:i+window])))
        i+=1
    return ma

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

def generalize_evaluator(rlAlgorithm,featureExtractor,nameAlg,nameFeat,multiple=1.0,numTrials=50, epsilon = 0.3, sigma = 500):

    ab_train = AngryBirdsMDP(levels = [1,3,5,7])
    ab_test = AngryBirdsMDP(levels = [2,4,6,8])

    name = nameAlg+'_'+nameFeat+'_'+str(64*multiple)


    rl = rlAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=featureExtractor,\
                            epsilon=epsilon)
    trainingOutcomes = simulate(ab_train,rl,numTrials=30, maxIterations=1000, verbose=False, show=False)
    testOutcomes = simulate(ab_test,rl,numTrials=10, maxIterations=1000, verbose=False, show=False)

    with open('../results/train_'+name,'wb') as file:
        pickle.dump(trainingOutcomes,file)
    with open('../results/test_'+name,'wb') as file:
        pickle.dump(testOutcomes,file)


if __name__ == '__main__':

    agent = angryAgent()
    evaluator(QLearningAlgorithm,agent.PPFeatureExtractor,'Q','PP',multiple = 1.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(QLearningAlgorithm,agent.NPPFeatureExtractor,'Q','NPP',multiple = 1.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(QLearningAlgorithm,agent.NPPOFeatureExtractor,'Q','NPPO',multiple = 1.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(QLearningAlgorithm,agent.NPPSFeatureExtractor,'Q','NPPS',multiple = 1.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(DRLSVI,agent.NPPFeatureExtractor,'DRLSVI','NPP',multiple = 1.0,numTrials=50, epsilon = 0.0, sigma = 500)
    evaluator(RLSVI,agent.NPPFeatureExtractor,'LSVI','NPP',multiple = 1.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(QLearningAlgorithm,agent.NPPFeatureExtractor,'Q','NPP',multiple = 2.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(RLSVI,agent.NPPFeatureExtractor,'RLSVI','NPP',multiple = 2.0,numTrials=50, epsilon = 0.0, sigma = 500)
    evaluator(RLSVI,agent.NPPFeatureExtractor,'LSVI','NPP',multiple = 2.0,numTrials=50, epsilon = 0.3, sigma = 500)
    evaluator(QLearningAlgorithm,agent.NPPFeatureExtractor,'Q','NPP',multiple = 2.0,numTrials=50, epsilon = 0.3, sigma = 500)

    generalize_evaluator(RLSVI,agent.NPPFeatureExtractor,'RLSVI','NPP',multiple=1.0,numTrials=50, epsilon = 0.3, sigma = 500)
    generalize_evaluator(QLearningAlgorithm,agent.NPPFeatureExtractor,'Q','NPP',multiple=1.0,numTrials=50, epsilon = 0.3, sigma = 500)

    trials = range(1,42)

    #Comparison
    totalRewards = {'PP': pickle.load(open('../results/totalRewards_Q_PP_64.0','rb')), \
                'NPP': pickle.load(open('../results/totalRewards_Q_NPP_64.0','rb')),\
                'NPPO': pickle.load(open('../results/totalRewards_Q_NPPO_64.0','rb')),\
                'NPPS': pickle.load(open('../results/totalRewards_Q_NPPS_64.0','rb'))}
    colors = ['red','blue','green','black']

    plt.figure()
    for i , name in enumerate(totalRewards.keys()):
        rewards = [x/10000.0 for x in movingAverage(totalRewards[name],window=10)]
        trials = range(len(rewards))
        plt.plot(trials,rewards,lw=2,color=colors[i], label = name)
        plt.scatter(trials,rewards)

    plt.legend(loc='upper left')
    plt.xlabel('Number of episodes',fontsize='large')
    plt.ylabel('Average reward',fontsize='large')
    plt.title('Average rewards for NPP features', fontsize=20)
    plt.savefig('../plots/RLSVIAverageReward.png')

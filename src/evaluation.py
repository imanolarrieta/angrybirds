from QLearner import QLearningAlgorithm
from AngryBirds import AngryBirdsGame
import math
from abAPI import *
from util import *
from GameAgent import angryAgent
from matplotlib import pyplot as plt

if __name__ == '__main__':

    ab = AngryBirdsMDP()
    explorationProb = 0.3
    agent = angryAgent(explorationProb=explorationProb)
    RUN_FAST = False  # Set this to False to wait until state is steadied before taking new action (important for learning)

    rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=agent.featureExtractor,discount=ab.discount(),\
                            explorationProb=explorationProb)
    trainingOutcomes = simulate(ab,rl,numTrials=40, maxIterations=1000, verbose=True, show=False)

    explorationProb = 0
    agent = angryAgent(explorationProb=explorationProb)
    rl.setExplorationProb(explorationProb)
    testOutcomes = simulate(ab,rl,verbose = True, show=True)

    print(trainingOutcomes)
    print(testOutcomes)

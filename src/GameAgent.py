# Learning Agent for Angrybirds
# Imanol Arrieta, Bernardo Ramos, Lars Roemheld
#
# Framework for applying different learning algorithms on the Angrybirds game
# This file can be run to simulate a game with the agent playing

# TODO just pretend API exists already
from QLearner import QLearningAlgorithm
from AngryBirds import AngryBirdsGame
import math
from abAPI import *
from util import *

def API_gamestate():
    # TODO
    return None

class angryAgent:
    def __init__(self, discount=1.0, explorationProb=0.2):
        self.learner = QLearningAlgorithm(actions=self.getAngryBirdsActions, featureExtractor=self.featureExtractor, \
                                          discount=discount, explorationProb=explorationProb)

    def getAngryBirdsActions(self, state):
        """
        returns a list of allowed actions given a game state. This is where discretization of the action space
        happens. Currently ignores given state, since all actions are always possible.
        Angles: value is in radian (2pi = full circle). 0 is straight to the left, i.e. pi is straight to right
                negative values are counter-clockwise
                positive values are clockwise
        Distance: just keep it positive (normally l2-norm of mousepos-slingpos).
        :return: A list of allowed actions (angle, distance) tuples, where angle and distance are floating point numbers from the slingshot
        """
        # Current setting: 45 options, given by 5 distances (=launch power) and 9 angles.
        # TODO: test if reducing number of distances and increasing angles helps performance (intuition)
        allowedAngles = [x/10.0 * math.pi for x in range(5, 13, 1)] # 0.5pi (90deg upwards) to 1.25pi (45deg downwards)
        allowedDistances = range(20, 90+1, 20)
        return [(a, d) for a in allowedAngles for d in allowedDistances]

    def featureExtractor(self, state, action):
        """
        Returns a dictionary/counter with key/value pairs that are interpreted as feature name (can be any ID) and feature value
        This is used for function approximation [ i.e. Q(s,a)= w * featureExtractor(s, a) ]
        :param state: a gamestate (as would be passed to Q)
        :param action: a action (as would be passed to Q)
        :return: dictionary/counter that gives a (potentially sparse) feature vector
        """
        # Current GameState:
        # self.birds = {'number': len(game.getBirds()), 'positions': game.getBirdPositions()}
        # self.pigs = {'number': len(game.getPigs()), 'positions': game.getPigPositions()}
        # self.polys = {'number': len(game.getPolys()), 'features': game.getPolyFeatures()}
        # self.score = game.getScore()

        features = []
        # rounded pig position and action indicator features
        for pos in state.pigs['positions']:
            xpig = pos[0]
            ypig = pos[1]
            features.append((('pigpos', (round(xpig, -1), round(ypig, -1)), action), 1))
        return features

    def featureExtractor2(self, state, action):
        """
        Returns a dictionary/counter with key/value pairs that are interpreted as feature name (can be any ID) and feature value
        This is used for function approximation [ i.e. Q(s,a)= w * featureExtractor(s, a) ]
        :param state: a gamestate (as would be passed to Q)
        :param action: a action (as would be passed to Q)
        :return: dictionary/counter that gives a (potentially sparse) feature vector
        """
        # Current GameState:
        # self.birds = {'number': len(game.getBirds()), 'positions': game.getBirdPositions()}
        # self.pigs = {'number': len(game.getPigs()), 'positions': game.getPigPositions()}
        # self.polys = {'number': len(game.getPolys()), 'features': game.getPolyFeatures()}
        # self.score = game.getScore()

        features = []
        angle = action[0]
        distance = action[1]
        # rounded pig position and action indicator features
        for pos in state.pigs['positions']:
            xpig = pos[0]
            ypig = pos[1]
            features.append((('pigpos+action', (round(xpig, -1), round(ypig, -1)), action), 1)) #This is an indicator of the position of each pig and the action to take
            features.append((('x+action', action), xpig)) #An indicator of the x coordinate and the action taken
            features.append((('y+action', action), ypig)) #An indicator of the y coordinate and the action taken
            features.append((('xy+action', action), xpig * ypig)) #Since Q is linearly approximated, this allows for interaction effects between x and y (important for location)

        for poly in state.polys['features']:
            polyposition = poly[0]
            features.append((('polypos', (round(polyposition[0], -1), round(polyposition[1], -1)), action), 1)) #This is an indicator of the position of each polygon
        return features


    def getAction(self, state): return self.learner.getAction(state)
    def incorporateFeedback(self, state, action, reward, newState): return self.learner.incorporateFeedback(state, action, reward, newState)


if __name__=='__main__':
    ab = AngryBirdsMDP(level=3)
    explorationProb = 0.3
    agent = angryAgent(explorationProb=explorationProb)
    RUN_FAST = False  # Set this to False to wait until state is steadied before taking new action (important for learning)

    rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=agent.featureExtractor,discount=ab.discount(),\
                            explorationProb=explorationProb)
    simulate(ab,rl,numTrials=20, maxIterations=1000, verbose=True, show=False)



    # # Learn Loop, baby!
    # oldState = None
    # actions = [(-1.0, 30), (-0.5, 30), (0.0, 30), (0.5, 30), (1.0, 30)]

    # while ab.running:
    #     currentGameState = GameState(ab)
    #     currentScore = ab.getScore()
    #     if oldState: agent.incorporateFeedback(oldState, (actionAngle, actionDistance), currentScore-oldScore, currentGameState)
    #     oldState = currentGameState
    #     oldScore = currentScore
    #
    #     actionAngle, actionDistance = agent.getAction(currentGameState)
    #     ab.performAction(actionAngle, actionDistance)
    #
    #     if not RUN_FAST:
    #         ab.runUntilStatic(show=False)
    #         ab.runFrames(5, show=True)
    #     else: ab.runFrames(30, show=True)




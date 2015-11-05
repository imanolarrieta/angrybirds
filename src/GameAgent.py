# Learning Agent for Angrybirds
# Imanol Arrieta, Bernardo Ramos, Lars Roemheld
#
# Framework for applying different learning algorithms on the Angrybirds game
# This file can be run to simulate a game with the agent playing

# TODO just pretend API exists already
from QLearner import QLearningAlgorithm
from AngryBirds import AngryBirdsGame
import math

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
        allowedDistances = range(10, 90+1, 20)
        return [(a, d) for a in allowedAngles for d in allowedDistances]

    def featureExtractor(self, state, action):
        """
        Returns a dictionary/counter with key/value pairs that are interpreted as feature name (can be any ID) and feature value
        This is used for function approximation [ i.e. Q(s,a)= w * featureExtractor(s, a) ]
        :param state: a gamestate (as would be passed to Q)
        :param action: a action (as would be passed to Q)
        :return: dictionary/counter that gives a (potentially sparse) feature vector
        """
        # TODO pull features from API (Bernardo started something already?)
        return [('feature1', 5), ('feature2', -5)]

    def getAction(self, state): return self.learner.getAction(state)
    def incorporateFeedback(self, state, action, reward, newState): return self.learner.incorporateFeedback(self, state, action, reward, newState)


if __name__=='__main__':
    ab = AngryBirdsGame()
    agent = angryAgent(explorationProb=1.0)

    ab.runFrames(20,show=True)
    # Learn Loop, baby!
    oldState = None
    actions = [(-1.0, 30), (-0.5, 30), (0.0, 30), (0.5, 30), (1.0, 30)]
    for iter in range(100):
        currentGameState = API_gamestate()
        currentScore = ab.getScore()
        if oldState: agent.incorporateFeedback(oldState, (actionAngle, actionDistance), currentScore-oldScore, currentGameState)
        oldState = currentGameState
        oldScore = currentScore

        actionAngle, actionDistance = agent.getAction(currentGameState)
        ab.performAction(actionAngle, actionDistance)
        ab.runUntilStatic(show=True)
        #ab.runFrames(30, show=True)
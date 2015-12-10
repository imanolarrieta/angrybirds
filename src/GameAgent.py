# Learning Agent for Angrybirds
# Imanol Arrieta, Bernardo Ramos, Lars Roemheld
#
# Framework for applying different learning algorithms on the Angrybirds game
# This file can be run to simulate a game with the agent playing

# TODO just pretend API exists already
from QLearner import QLearningAlgorithm
from sparseLearnerRLSVI import RLSVI_wrapper
from AngryBirds import AngryBirdsGame
import math
from abAPI import *
from util import *
import math
import numpy as np
from collections import Counter

class angryAgent:
    def getAngryBirdsActions(self, state, multiple = 1.0):
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
        allowedAngles = [x/10.0/multiple * math.pi for x in range(5*multiple, 13*multiple, 1)] # 0.5pi (90deg upwards) to 1.25pi (45deg downwards)
        allowedDistances = range(20, 90+1, 20 * multiple)
        return [(a, d) for a in allowedAngles for d in allowedDistances]

    def featureExtractorXYaction(self, state, action):
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

    def featureExtractorXpigYpig(self, state, action):
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
        positions = sorted(state.pigs['positions'])
        for i,pos in enumerate(positions):
            xpig = pos[0]
            ypig = pos[1]
            features.append((('x+action'+str(i), action), xpig)) #An indicator of the x coordinate and the action taken
            features.append((('y+action'+str(i), action), ypig)) #An indicator of the y coordinate and the action taken
            features.append((('xy+action'+str(i), action), xpig * ypig)) #Since Q is linearly approximated, this allows for interaction effects between x and y (important for location)

        return features

    def polyIndicatorFeatureExtractor(self, state, action):
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


        for poly in state.polys['features']:
            polyposition = poly[0]
            features.append((('polypos', (round(polyposition[0], -1), round(polyposition[1], -1)), action), 1)) #This is an indicator of the position of each polygon
        return features

    def centroidFeatureExtractor(self, state, action):
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
        positions = sorted(state.pigs['positions'])
        if len(positions) >0 :
            meanx, meany = np.mean(positions, axis=0)
            sigmax, sigmay = np.mean(positions, axis=0)
            features.append((('centroid_x', action), meanx))
            features.append((('centroid_y', action), meany))
            features.append((('centroid_sigmax', action), sigmax))
            features.append((('centroid_sigmay', action), sigmay))

        return features

    def gridFeatureExtractor(self, state, action, size=3, shifted=False, type='pig', count=True):
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

        width = size*28.0 #This is |size| times the diameter of a pig
        features = []
        # rounded pig position and action indicator features
        if type=='pig':
            positions = sorted(state.pigs['positions'])
        elif type=='poly':
            positions = [poly[0] for poly in state.polys['features']]


        offset = width/2 if shifted else 0.0
        if not count:
            presence = {}
            for i, pos in enumerate(positions):
                try:
                    squarex = math.floor((pos[0]+offset)/width)*width #This is the position
                    squarey = math.floor((pos[1]+offset)/width)*width
                    presence[(squarex, squarey)] = 1
                except ValueError:
                    continue
        else:
            presence = Counter()
            for i, pos in enumerate(positions):
                try:
                    squarex = math.floor((pos[0]+offset)/width)*width #This is the position
                    squarey = math.floor((pos[1]+offset)/width)*width
                    presence[(squarex, squarey)] += 1
                except ValueError:
                    continue

        s = '_shifted_' if shifted else '_'
        c = 'count_' if count else 'indicator_'
        #Now add indicator functions to the features
        for squarex, squarey in presence:
            features.append(((c+type+'_gridwidth'+str(width)+''+s+'x'+str(squarex)+'_y'+str(squarey), action), presence[(squarex, squarey)])) #An indicator of the (x,y) coordinate and the action taken
        return features

    def nestedGridFeatureExtractor(self, state, action, minsize=2, shifted=False, type='pig', count=True):
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
        size = 10.5
        while size > minsize:
            features += self.gridFeatureExtractor(state, action, size=size, shifted=shifted, type=type, count=count)
            size /= 2
        return features

    def custom1FeatureExtractor(self, state, action):
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
        features += self.featureExtractorXYaction(state, action)
        features += self.centroidFeatureExtractor(state, action)
        features += self.featureExtractorXpigYpig(state, action)
        return features


    def custom2FeatureExtractor(self, state, action):
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
        features += self.nestedGridFeatureExtractor(state, action, minsize=2, shifted=False, type='pig')
        features += self.nestedGridFeatureExtractor(state, action, minsize=2, shifted=False, type='poly')
        return features



    def getAction(self, state): return self.learner.getAction(state)
    def incorporateFeedback(self, state, action, reward, newState): return self.learner.incorporateFeedback(state, action, reward, newState)


if __name__=='__main__':
    ab = AngryBirdsMDP()
    explorationProb = 0.3
    agent = angryAgent()
    RUN_FAST = False  # Set this to False to wait until state is steadied before taking new action (important for learning)

    # rl = QLearningAlgorithm(actions=agent.getAngryBirdsActions,featureExtractor=agent.nestedGridFeatureExtractor,discount=ab.discount(),\
    #                         explorationProb=explorationProb)
    rl = RLSVI_wrapper(actions=agent.getAngryBirdsActions,featureExtractor=agent.nestedGridFeatureExtractor)
    simulate(ab,rl,numTrials=20, maxIterations=1000, verbose=True, show=False, episodicLearning=True)



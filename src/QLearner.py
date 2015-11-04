# Q-Learner for Angrybirds, using function approximation
# Imanol Arrieta, Bernardo Ramos, Lars Roemheld
# Adapted from a homework assignment in Percy Liang's class CS221 at Stanford University
import collections
import random
import math

class QLearningAlgorithm():
    """
    actions: a function (!) that takes a state and returns a list of allowed actions.
    discount: a number between 0 and 1, which determines the discount factor
    featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
    explorationProb: the epsilon value indicating how frequently the policy returns a random action
    """
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    def getQ(self, state, action):
        """
        Return the Q function for the current gameState and action, computed as (linear) function approximation
        """
        score = 0.0
        for f_name, f_val in self.featureExtractor(state, action):
            score += self.weights[f_name] * f_val
        return score

    def getAction(self, state):
        """
        Epsilon-greedy algorithm: with probability |explorationProb|, take a random action. Otherwise, take action that
        maximizes expected Q
        :param state: current gameState
        :return: the chosen action
        """
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    def setExplorationProb(self, probability):
        self.explorationProb = probability

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    def incorporateFeedback(self, state, action, reward, newState):
        # goal: w := w - eta * (Q(s,a;w) - (reward + discount * V_opt(newState)) * phi(s, a)
        # Where V_opt = max_a Q(s,a)
        if newState == None:
            V_newState = 0.0
        else:
            V_newState = max(self.getQ(newState, newAction) for newAction in self.actions(newState))

        # "Gradient descent" on the Q-Learner weights
        # TODO: Note that by not running on newWeights and rather run directly on Weights we might get "stochastic" performance
        newWeights = collections.Counter(self.weights)
        phi = self.featureExtractor(state, action)
        for (f_name, f_val) in phi:
            newWeights[f_name] = self.weights[f_name] - self.getStepSize() * (\
                    self.getQ(state, action) - reward - self.discount * V_newState \
                ) * f_val

        self.weights = newWeights

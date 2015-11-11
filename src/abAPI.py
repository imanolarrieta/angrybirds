import os
import sys
import math
import time
import pygame
current_path = os.getcwd()
sys.path.insert(0, os.path.join(current_path, "../pymunk-4.0.0"))
import pymunk as pm
from characters import Bird
from level import Level

#import imp
#AngryBirds = imp.load_source('AngryBirds', '/src/AngryBirds.py')
#from src import AngryBirds

import AngryBirds

class AngryBirdsMDP:

    # Initializes the game
    def __init__(self):
        self.game = AngryBirds.AngryBirdsGame()
        self.show=True

    def showLearning(self):
        self.show = True

    def showState(self):
        self.game.runFrames(40, show=True)

    def restart(self):
        self.game.restart()

    # Return the start state.
    def startState(self):
        # TODO Initialize with different levels
        self.game = AngryBirds.AngryBirdsGame()
        return GameState(self.game)

    # Return set of actions possible from |state|.
    def actions(self, state):
        #If number of active birds > 0, return possible angles and slingshot extensions
        # LR - this is implemented in GameAgent.py already
        raise NotImplementedError("Override me")


    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # If IsEnd(state), return the empty list.
    def succAndReward(self, state, action):
        #Do we have to check first if |state| corresponds to the current state of the game? If they don't coincide, use GameState to redefine the state of self.game
        pastscore = self.game.getScore()
        angle = action[0]
        distance = action[1]
        #Run action
        self.game.performAction(angle, distance)
        self.game.runUntilStatic(self.show)
        if state.isEnd():
            if state.isWin():
                return (None,self.game.getScore())
            else:
                return (None,-1000)

        #Calculate reward
        reward = self.game.getScore() - pastscore
        #Return the next state with probability 1.
        return (GameState(self.game), reward)

    def discount(self):
        return 1


    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, reward in self.succAndReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

class GameState():

    def __init__(self, game):
        self.nbirds = game.getNumberRemainingBirds()
        self.pigs = {'number': len(game.getPigs()), 'positions': game.getPigPositions()}
        self.polys = {'number': len(game.getPolys()), 'features': game.getPolyFeatures()}

    def isEnd(self):
        return self.nbirds==0 or self.pigs['number']==0

    def isWin(self):
        return self.nbirds>0 and self.pigs['number']==0

    def isLoose(self):
        return self.nbirds==0




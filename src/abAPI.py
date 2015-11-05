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
    def __init__(self, nframes=100, showProgression=True):
        self.game = AngryBirds.AngryBirdsGame()
        self.nframes = nframes
        self.show = showProgression

    # Return the start state.
    def startState(self):
        self.game = self.game.__init__() #How to deal with the initialization of different levels?
        return GameState(self.game)

    # Return set of actions possible from |state|.
    def actions(self, state):
        #If number of active birds > 0, return possible angles and slingshot extensions
        # LR - this is implemented in GameAgent.py already
        raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action):
        #Do we have to check first if |state| corresponds to the current state of the game? If they don't coincide, use GameState to redefine the state of self.game
        pastscore = state.score
        angle = action[0]
        distance = action[1]
        #Run action
        self.game.performAction(angle, distance)
        self.game.runFrames(self.nframes, self.show)
        ###IF GAME ENDS HERE WITHOUT KILLING ALL PIGS, REWARD = -INFINITY and SuccState=None  (not implemented)
        #Calculate reward
        reward = self.game.score - pastscore
        #Return the next state with probability 1.
        return (GameState(self.game), 1, reward)

    def discount(self): raise NotImplementedError("Override me")

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
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

class GameState():

    def __init__(self, game):
        self.birds = {'number': len(game.getBirds()), 'positions': game.getBirdPositions()}
        self.pigs = {'number': len(game.getPigs()), 'positions': game.getPigPositions()}
        self.polys = {'number': len(game.getPolys()), 'features': game.getPolyFeatures()}
        self.score = game.getScore()

#
# class GameState:
#
#     """
#       A GameState specifies the full game state, including the birds, blocks,
#       pigs configurations and score changes.
#
#       GameStates are used by the Game object to capture the actual state of the game and
#       can be used by agents to reason about the game.
#
#       Much of the information in a GameState is stored in a GameStateData object.  We
#       strongly suggest that you access that data via the accessor methods below rather
#       than referring to the GameStateData object directly.
#
#       Note that in classic Pacman, Pacman is always agent 0.
#       """
#
#       ####################################################
#       # Accessor methods: use these to access state data #
#       ####################################################
#
#     def getLegalActions( self ):
#         """
#         Returns the legal actions for the player.
#         """
#         if self.isWin() or self.isLose(): return []
#
#
#         return AngryBirdRules.getLegalActions( self )
#
#     def __init__(self):
#         setupSpace()



# class GameRules:
#     """
#   These game rules manage the control flow of a game, deciding when
#   and how the game starts and ends
#   """
#     def __init__(self,quiet):
#         self.quiet = quiet
#
#     def newGame( self):
#         agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
#         initState = GameState()
#         initState.initialize( layout, len(ghostAgents) )
#         game = Game(agents, display, self, catchExceptions=catchExceptions)
#         game.state = initState
#         self.initialState = initState.deepCopy()
#         return game


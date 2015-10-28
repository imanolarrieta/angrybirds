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



class GameState:

    """
      A GameState specifies the full game state, including the birds, blocks,
      pigs configurations and score changes.

      GameStates are used by the Game object to capture the actual state of the game and
      can be used by agents to reason about the game.

      Much of the information in a GameState is stored in a GameStateData object.  We
      strongly suggest that you access that data via the accessor methods below rather
      than referring to the GameStateData object directly.

      Note that in classic Pacman, Pacman is always agent 0.
      """

      ####################################################
      # Accessor methods: use these to access state data #
      ####################################################

    def getLegalActions( self ):
        """
        Returns the legal actions for the player.
        """
        if self.isWin() or self.isLose(): return []


        return AngryBirdRules.getLegalActions( self )

    def __init__(self):
        setupSpace()



class GameRules:
    """
  These game rules manage the control flow of a game, deciding when
  and how the game starts and ends
  """
    def __init__(self,quiet):
        self.quiet = quiet

    def newGame( self):
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize( layout, len(ghostAgents) )
        game = Game(agents, display, self, catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        return game
###File description (src):

#abAPI
This API contains the definition of the Angry Birds as an approximate MDP (define states and their successors by manipulating the AngryBirdsGame class in AngryBirds.py). We also define a simplified version of the Game States by extracting all necessary information to run algorthms.

#AngryBirds
The original code was modified into an object-oriented setting, creating the class AngryBirdsGame. We define the basic methods that will let us extract all game features and manipulate the underlying game mechanics- we will now be able to set levels, get scores, extract pig positions, perform an action, run a certain number of frames, and so on.

#characters
This file defines all characteristics for pigs and birds, such as mass, life, radius, body, shape and velocity.

#evaluation
This file contains the codes for generating results and graphs by running it as __main__ and uncommenting the evaluator functions for each feature extractor (PP, NPP, NPPO, NPPS) and algorithm (Q-Learning, RLSVI).

#GameAgent
The class GameAgent defines the possible actions the bird can take, as well as all feature extractors described in the paper.

#DenseLearnerRLSVI
In this code the RLSVI algorithm is defined and adapted to our code using the RLSVI_wrapper class.

#level
This code contains the elements that define all 11 levels (number and positions of pigs and beams).

#polygon
Similar to characters, this code defines all characteristics of beams.

#QLearner
This code defines the Q-Learning algorithm based on the class’s code.

#sparseLearnerRLSVI
This code is DenseLearnerRLSVI.py except we treat sparse matrix operations at all steps. This highly improves the performance when running simulations.

#util
This file contains the basic tools we used in class to run simulations of the reinforcement learning algorithm. We adapted it so that we can set feedback incorporation after each action or after each episode.


### Codes to run and produce results

-To run as human play, run AngryBirds.py as __main__
-To produce the results (including those shown in the paper), run evaluation.py as __main__ and uncomment the evaluator functions for each feature extractor (PP, NPP, NPPO, NPPS) and algorithm (Q-Learning, RLSVI).
-To run either RL algorithm, run GameAgent as __main__ and uncommenting the appropriate function (RLSVI_wrapper for RLSVI, and QLearningAlgorithm for Q-Learning). One can also set parameters in calling these functions, for example setting show=True to see what the algorithm takes as actions.



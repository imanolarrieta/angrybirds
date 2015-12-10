'''
Simple implementation of RLSVI agent.

original author: Ian Osband, iosband@stanford.edu
On the algorithm: http://arxiv.org/abs/1402.0635

Adapted (wrapper structure) by Lars Roemheld, roemheld@stanford.edu
'''

import numpy as np

class RLSVI:
    '''
    RLSVI agent

    Important part is the memory, a list of lists.
        covs[h] = Sigma_h
        thetaMeans[h] = \overline{theta}_h
        thetaSamps[h] = \hat{theta}_h
        memory[h] = {oldFeat, reward, newFeat}
            oldFeat = A (nData x nFeat)
            reward = vector of rewards
            newFeat = array (nData x nFeat x nAction)
    with history appended row by row.
    '''
    def __init__(self, nFeat, nAction, epLen,
                 epsilon=0.0, sigma=1.0, lam=1.0, maxHist=1e6):
        self.nFeat = nFeat
        self.nAction = nAction
        self.epLen = epLen
        self.epsilon = epsilon
        self.sigma = sigma
        self.maxHist = maxHist

        # Make the computation structures
        self.covs = []
        self.thetaMeans = []
        self.thetaSamps = []
        self.memory = []
        for i in range(epLen + 1):
            self.covs.append(np.identity(nFeat) / float(lam))
            self.thetaMeans.append(np.zeros(nFeat))
            self.thetaSamps.append(np.zeros(nFeat))
            self.memory.append({'oldFeat': np.zeros([maxHist, nFeat]),
                                'rewards': np.zeros(maxHist),
                                'newFeat': np.zeros([maxHist, nAction, nFeat])})

    def update_obs(self, ep, h, oldObs, reward, newObs):
        '''
        Take in an observed transition and add it to the memory.

        Args:
            ep - int - which episode
            h - int - timestep within episode
            oldObs - nFeat x 1
            action - int
            reward - float
            newObs - nFeat x nAction

        Returns:
            NULL - update covs, update memory in place.
        '''
        if ep >= self.maxHist:
            print('****** ERROR: Memory Exceeded ******')

        # Covariance update
        u = oldObs / self.sigma
        S = self.covs[h]
        Su = np.dot(S, u)
        temp = np.outer(Su, Su)
        temp2 = np.dot(u.T, Su) # TODO LR added the ".T", is this correct?!
        self.covs[h] = S - (temp / (1 + temp2))

        # Adding the memory
        self.memory[h]['oldFeat'][ep, :] = oldObs.T  # TODO LR added the ".T", is this correct?!
        self.memory[h]['rewards'][ep] = reward
        self.memory[h]['newFeat'][ep, :, :] = newObs.T # TODO LR added the ".T", is this correct?!

        if len(self.memory[h]['oldFeat']) == len(self.memory[h]['rewards']) \
           and len(self.memory[h]['rewards']) == len(self.memory[h]['newFeat']):
            pass
        else:
            print('****** ERROR: Memory Failure ******')

    def update_policy(self, ep):
        '''
        Re-computes theta parameters via planning step.

        Args:
            ep - int - which episode are we on

        Returns:
            NULL - updates theta in place for policy
        '''
        H = self.epLen

        if len(self.memory[H - 1]['oldFeat']) == 0:
            return

        for i in range(H):
            h = H - i - 1
            A = self.memory[h]['oldFeat'][0:ep]
            nextPhi = self.memory[h]['newFeat'][0:ep, :, :]
            nextQ = np.dot(nextPhi, self.thetaSamps[h + 1])
            maxQ = nextQ.max(axis=1)
            b = self.memory[h]['rewards'][0:ep] + maxQ

            self.thetaMeans[h] = \
                self.covs[h].dot(np.dot(A.T, b)) / (self.sigma ** 2)
            self.thetaSamps[h] = \
                np.random.multivariate_normal(mean=self.thetaMeans[h],
                                              cov=self.covs[h])

    def pick_action(self, t, obs):
        '''
        The greedy policy according to thetaSamps

        Args:
            t - int - timestep within episode
            obs - nAction x nFeat - features for each action
            epsilon - float - probability of taking random action

        Returns:
            action - int - greedy with respect to thetaSamps
        '''
        # Additional epsilon-greedy. (LR)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nAction)
        else:
            qVals = np.dot(self.thetaSamps[t], obs.T)
            return qVals.argmax()

#-------------------------------------------------------------------------------

class RLSVI_wrapper:
    '''
    Wrapper class for Osband's RLVI implementation. Chiefly hacks a way around the tuple-based features used in our
    Q-Learner and game, creating rigid numpy-vector features. Additionally currently assumes that we are playing T episodes
    of length 1 timestep (i.e. we learn after every move and ignore episodes)
    '''
    def __init__(self, actions, featureExtractor, epsilon=0.0):
        self.actions = actions
        self.featureExtractor = featureExtractor
        self.currentEp = 0
        self.maxNFeatures = 500
        self.featurePos = {} # super hacky dictionary: here we store the vector position that any given feature is stored in.
        # Note that this needs to be constant across timesteps, so the dictionary needs to persist.
        self.nFeaturesSeen = 0
        self.rlsvi = RLSVI(self.maxNFeatures, len(actions(0)), epLen=1, epsilon=epsilon)
        #TODO Note: this is not robust. Calling actions(state=0) works when state is ignored, but will WAT otherwise.

    def getObsVect(self, state, action=None):
        '''
        Helper function that converts the more flexible tuple-syntax for features into the more rigid, linear-algebra
        friendly vector format
        '''
        if action is None:
            actions = self.actions(state)  #TODO Note: this is not robust. When available actions really vary by state, this may WAT.
        else:
            actions = [action]

        obsVect = np.zeros((len(actions), self.maxNFeatures))

        for (i, a) in enumerate(actions):
            feats = self.featureExtractor(state, a)
            for f in feats: # f is tuple (feature_name, feature_value)
                p = self.featurePos.get(f[0]);
                if p is None:
                    self.nFeaturesSeen += 1
                    assert self.nFeaturesSeen < self.maxNFeatures, 'RLSVI maxNFeatures is too small for actual features produced'
                    self.featurePos[f[0]] = self.nFeaturesSeen
                    p = self.nFeaturesSeen
                print(self.nFeaturesSeen)
                obsVect[i][p] = f[1]
        return obsVect

    def getAction(self, state):
        """
        Epsilon-greedy algorithm: with probability |explorationProb|, take a random action. Otherwise, take action that
        maximizes expected Q
        :param state: current gameState
        :return: the chosen action
        """

        # options = [(self.getQ(state, action), action) for action in self.actions(state)]
        # bestVal = max(options)[0]
        # return random.choice([opt[1] for opt in options if opt[0] == bestVal])
        obsVect = self.getObsVect(state)
        i_action = self.rlsvi.pick_action(0, obsVect)
        return self.actions(state)[i_action]

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    def incorporateFeedback(self, state, action, reward, newState):
        if newState == None:
            return

        obsVect_old = self.getObsVect(state, action).T
        obsVect_new = self.getObsVect(newState).T

        self.rlsvi.update_obs(self.currentEp, 0, obsVect_old, reward, obsVect_new)
        self.rlsvi.update_policy(self.currentEp)
        self.currentEp += 1



#-------------------------------------------------------------------------------
class eLSVI(RLSVI):
    '''
    epsilon-greedy LSVI agent.

    This is just RLSVI, but we don't use the noise!
    '''

    def update_policy(self, ep):
        '''
        Re-computes theta parameters via planning step.

        Args:
            ep - int - which episode are we on

        Returns:
            NULL - updates theta in place for policy
        '''
        H = self.epLen

        if len(self.memory[H - 1]['oldFeat']) == 0:
            return

        for i in range(H):
            h = H - i - 1
            A = self.memory[h]['oldFeat'][0:ep]
            nextPhi = self.memory[h]['newFeat'][0:ep, :, :]
            nextQ = np.dot(nextPhi, self.thetaSamps[h + 1])
            maxQ = nextQ.max(axis=1)
            b = self.memory[h]['rewards'][0:ep] + maxQ

            self.thetaMeans[h] = \
                self.covs[h].dot(np.dot(A.T, b)) / (self.sigma ** 2)
            self.thetaSamps[h] = self.thetaMeans[h]
#-------------------------------------------------------------------------------





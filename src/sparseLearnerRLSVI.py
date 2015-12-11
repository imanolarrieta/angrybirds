'''
Simple implementation of RLSVI agent.

original author: Ian Osband, iosband@stanford.edu
On the algorithm: http://arxiv.org/abs/1402.0635

Adapted (wrapper structure) by Lars Roemheld, roemheld@stanford.edu
Changed to sparse matrices by Bernardo Ramos, bramos@stanford.edu
'''
import scipy.sparse as sp
import numpy as np
import random

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
                 epsilon=0.0, sigma=1.0, lam=1.0, maxHist=5000):
        self.nFeat = nFeat
        self.nAction = nAction
        self.epLen = epLen
        self.epsilon = epsilon
        self.sigma = sigma
        self.maxHist = maxHist
        self.isRLSVI = (epsilon == 0.0) # sample from belief only if epsilon is 0. Setting this here allows epsilon to be changed later (to stop LSVI from exploring further)

        # Make the computation structures
        self.covs = []
        self.thetaMeans = []
        self.thetaSamps = []
        self.memory = []
        for i in range(epLen + 1):
            self.covs.append(sp.identity(nFeat) / float(lam))
            self.thetaMeans.append(sp.dok_matrix((nFeat,1)))
            self.thetaSamps.append(sp.dok_matrix((nFeat,1)))
            self.memory.append({'oldFeat': sp.dok_matrix((maxHist, nFeat)),
                                'rewards': sp.dok_matrix((maxHist,1)),
                                'newFeat': {j:sp.dok_matrix((nAction, nFeat)) for j in range(maxHist)}})

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
            print('****** ERROR: HISTORY Exceeded ******')

        # Covariance update
        u = oldObs / self.sigma
        S = self.covs[h]
        Su = S*u
        temp = Su*(Su.T)
        temp2 = (u.T*Su)[0,0] # TODO LR added the ".T", is this correct?!
        self.covs[h] = S - (temp / (1 + temp2))

        # Adding the memory
        self.memory[h]['oldFeat'][ep, :] = oldObs.T  # TODO LR added the ".T", is this correct?!
        self.memory[h]['rewards'][ep] = reward
        self.memory[h]['newFeat'][ep] = newObs.T # TODO LR added the ".T", is this correct?!

        if self.memory[h]['oldFeat']._shape[0] == self.memory[h]['rewards']._shape[0] \
           and self.memory[h]['rewards']._shape[0] == len(self.memory[h]['newFeat']):
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

        if len(self.memory[H - 1]['oldFeat']) == 0 or ep == 0:
            return
        for i in range(H):
            h = H - i - 1
            A = self.memory[h]['oldFeat'].tocsc()[0:ep,:]
            nextPhi = {j:self.memory[h]['newFeat'][j] for j in range(ep)}
            nextQ = {j: nextPhi[j]*self.thetaSamps[h + 1] for j in range(ep)}
            maxQ = sp.csc_matrix([nextQ[j].tocsr().max(axis=0).toarray()[0][0] for j in nextQ]).T
            b = self.memory[h]['rewards'][0:ep] + maxQ
            self.thetaMeans[h] = \
                self.covs[h]*A.T*b / (self.sigma ** 2)

            #(Commented code:) To sample from the normal distribution including covariances
            # self.thetaSamps[h] = \
            #     sp.csc_matrix(np.random.multivariate_normal(mean=np.array(self.thetaMeans[h].todense()).flatten(),
            #                                   cov=self.covs[h].todense())).T

            #Simulate gaussians assuming independence (i.e. taking only the diagonal terms in the covariance matrix)
            mu = np.array(self.thetaMeans[h].todense()).flatten()
            sig = np.sqrt(self.covs[h].diagonal())
            if self.isRLSVI:
                self.thetaSamps[h] = sp.csc_matrix(mu + sig*np.random.normal(size=self.nFeat)).T
            else:
                self.thetaSamps[h] = sp.csc_matrix(mu).T # If epsilon>0.0, do not sample (and use epsilon-greedy exploration)


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
            qVals = (obs*self.thetaSamps[t]).tocoo()
            if len(qVals.data)==0:
                return np.random.randint(self.nAction)
            else:
               I = qVals.data.argmax()
            return qVals.row[I]

#-------------------------------------------------------------------------------

class RLSVI_wrapper:
    '''
    Wrapper class for Osband's RLVI implementation. Chiefly hacks a way around the tuple-based features used in our
    Q-Learner and game, creating rigid numpy-vector features. Additionally currently assumes that we are playing T episodes
    of length 1 timestep (i.e. we learn after every move and ignore episodes)
    If epsilon > 0.0, this becomes LSVI, using unsampled least-squares estimates and using epsilon-greedy exploration.

    '''
    def __init__(self, actions, featureExtractor, epsilon=0.0, sigma=500.0):
        self.actions = actions
        self.featureExtractor = featureExtractor
        self.currentEp = 0
        self.maxNFeatures = 10000
        self.featurePos = {} # super hacky dictionary: here we store the vector position that any given feature is stored in.
        # Note that this needs to be constant across timesteps, so the dictionary needs to persist.
        self.nFeaturesSeen = 0
        self.rlsvi = RLSVI(self.maxNFeatures, len(actions(0)), epLen=1, epsilon=epsilon, sigma=sigma)
        #TODO Note: this is not robust. Calling actions(state=0) works when state is ignored, but will WAT otherwise.

    def makeLSVI(self, epsilon):
        self.rlsvi.isRLSVI = False
        self.rlsvi.epsilon = epsilon

    def makeRLSVI(self):
        self.rlsvi.isRLSVI = True
        self.rlsvi.epsilon = 0.0


    def getObsVect(self, state, action=None):
        '''
        Helper function that converts the more flexible tuple-syntax for features into the more rigid, linear-algebra
        friendly vector format
        '''
        if action is None:
            actions = self.actions(state)  #TODO Note: this is not robust. When available actions really vary by state, this may WAT.
        else:
            actions = [action]

        obsVect = sp.dok_matrix((len(actions), self.maxNFeatures))

        for (i, a) in enumerate(actions):
            feats = self.featureExtractor(state, a)
            for f in feats: # f is tuple (feature_name, feature_value)
                p = self.featurePos.get(f[0]);
                if p is None:
                    self.nFeaturesSeen += 1
                    assert self.nFeaturesSeen < self.maxNFeatures, 'RLVI maxNFeatures is too small for actual features produced'
                    self.featurePos[f[0]] = self.nFeaturesSeen
                    p = self.nFeaturesSeen
                obsVect[i, p] = f[1]
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



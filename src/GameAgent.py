# Learning Agent for Angrybirds
# Imanol Arrieta, Bernardo Ramos, Lars Roemheld
#
# Framework for applying different learning algorithms on the Angrybirds game
#

def getAngryBirdsActions(state):
    """
    returns a list of allowed actions given a game state. This is where discretization of the action space
    happens
    :return: A list of allowed actions (angle, distance) tuples, where angle and distance are floating point numbers from the slingshot
    """

    allowedAngles = [a / 10.0 for a in range(-16, 1, 1)]
    allowedDistances = range(0, 90, 5)
    return [(a, d) for a in allowedAngles for d in allowedDistances]

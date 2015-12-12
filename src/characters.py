#characters
#This file defines all characteristics for pigs and birds, such as mass, life, radius, body, shape and velocity.

import pymunk as pm
from pymunk import Vec2d


class Bird():
    def __init__(self, distance, angle, x, y, space):
        self.life = 100
        mass = 5
        self.radius = 12
        self.inertia = pm.moment_for_circle(mass, 0, self.radius, (0, 0))
        body = pm.Body(mass, self.inertia)
        body.position = x, y
        power = distance * 53
        impulse = power * Vec2d(1, 0)
        angle = -angle
        body.apply_impulse(impulse.rotated(angle))
        shape = pm.Circle(body, self.radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 1
        shape.collision_type = 0
        space.add(body, shape)
        self.body = body
        self.shape = shape

    def getPosition(self):
        # Get Bird position
        return self.body.position

    def getRadius(self):
        return self.radius

    def getVelocity(self):
        return min(abs(self.body.velocity[0]),abs(self.body.velocity[1]))

    def getDirection(self):
        if self.body.velocity[0]<0:
            return 'left'
        else:
            return 'right'

    def ageWhenStatic(self):
        # Ages the bird by one unit when no more movement.
        if self.getVelocity()<50 or self.getDirection()=='left':
            self.life-=1
        else:
            self.life = 100

    def dead(self):
        # Returns if the bird is dead or not
        return self.life<=0


class Pig():
    def __init__(self, x, y, space):
        self.life = 20
        mass = 5
        self.radius = 14
        inertia = pm.moment_for_circle(mass, 0, self.radius, (0, 0))
        body = pm.Body(mass, inertia)
        body.position = x, y
        shape = pm.Circle(body, self.radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 1
        shape.collision_type = 1
        space.add(body, shape)
        self.body = body
        self.shape = shape

    def getPosition(self):
        # Get Pig position
        return self.body.position

    def getRadius(self):
        # Get Radius
        return self.radius

    def getVelocity(self):
            return abs(self.body.velocity[1])
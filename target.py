import numpy as np


class Target:
    """Point target object"""

    def __init__(self, position, velocity, rcs, P=np.array([]), V=np.array([])):
        """Instantiate a target object

        Args:
            position (ndarray): A vector of the XYZ position of the target in an
            arbitrary coordinate system
            velocity (ndarray): A vector of the XYZ velocity of the target in an
            arbitrary coordinate system
            rcs (float): The radar cross section of the target (m^2) 
        """
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.rcs = rcs
        self.P = P
        self.V = V
        self.pos_state_ind = -1
        self.vel_state_ind = -1
        if P.size > 0 and self.position.size > 0:
            self.pos_state_ind = np.linalg.norm(
                self.position - P, keepdims=True, axis=1).argmin()
        if V.size > 0 and self.velocity.size > 0:
            self.vel_state_ind = np.linalg.norm(
                self.velocity - V, keepdims=True, axis=1).argmin()

    def step(self, dt):
        """Update the target motion profile

        Args:
            dt (float): Time change since last step
        """
        self.position += self.velocity*dt
        if self.P.size > 0:
            self.pos_state_ind = np.linalg.norm(
                self.position - self.P, keepdims=True, axis=1).argmin()
        if self.V.size > 0:
            self.vel_state_ind = np.linalg.norm(
                self.velocity - self.V, keepdims=True, axis=1).argmin()

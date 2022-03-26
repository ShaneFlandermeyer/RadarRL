import numpy as np

class Target:
    """Point target object"""

    def __init__(self, position, velocity, rcs):
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

    def step(self, dt):
        """Update the target motion profile

        Args:
            dt (float): Time change since last step
        """
        self.position += self.velocity*dt

import numpy as np
from abc import ABC, abstractmethod

class Interference(ABC):
    """Interference object base class"""

    def __init__(self, tx_power, states, state_ind):
        """Instantiate a new interference object

        Args:
            tx_power (float): Transmit power of the interference AT THE RADAR RECEIVER.
            Therefore, this is a very small value.
            TODO: Make this the actual transmitted power, then use scenario
            geometry to compute the power at the radar receiver. 

            states (_type_, optional): A matrix of all possible transmission
            frequencies in a discretized spectrum, where each row is a possible
            state. 
            Defaults to np.array([]).

            state_ind (int, optional): The row index of the current state in the
            states matrix.
            Defaults to 0.
        """
        self.tx_power = tx_power
        self.states = states
        self.state_ind = state_ind
        self.current_state = states[state_ind]

    @abstractmethod
    def step(self):
        """ 
        Define the frequency hopping behavior of the interference at each time step
        """
        pass


class ConstantInterference(Interference):
    """Constant interference object.

    This interferer transmits at a single frequency pattern for the entire simulation

    Args:
        Interference: Abstract interference parent class
    """

    def __init__(self, tx_power=0, states=np.array([]), state_ind=0):
        super(ConstantInterference, self).__init__(tx_power, states, state_ind)

    def step(self):
        """
        Continue transmitting at the same frequency
        """
        pass

# class PulsedInterference:
#     """
#     Interference that turns on and off every iteration, but stays in the same
#     frequency band
#     """

#     def __init__(self, inr=14, states=np.array([]), state_ind=0):
#         # ConstantInterference-to-noise ratio at the radar receiver. To simplify the
#         # scenario, this is position-independent
#         self.inr = inr
#         self.states = states
#         self.state_ind = state_ind
#         self.state = states[state_ind]

#     def step(self):
#         if np.sum(self.state) == 0:
#             self.state = self.states[self.state_ind]
#         else:
#             self.state = self.states[0]
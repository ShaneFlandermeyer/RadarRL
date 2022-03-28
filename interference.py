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

    """

    def __init__(self, tx_power=0, states=np.array([]), state_ind=0):
        super(ConstantInterference, self).__init__(tx_power, states, state_ind)

    def step(self):
        """
        Continue transmitting at the same frequency
        """
        pass


class IntermittentInterference(Interference):
    """Time-intermittent interference object.

    This interference transmits in a single pre-defined frequency band, but
    toggles on and off with a user-defined probability at every time step.
    """

    def __init__(self, tx_power=0, states=np.array([]), state_ind=0, transition_prob=0.1):
        super(IntermittentInterference, self).__init__(
            tx_power, states, state_ind)
        self.transition_prob = transition_prob
        self.original_state_ind = state_ind
        self.on = 1

    def step(self):
        """
        Transition on/off with a constant probability
        """
        if np.random.rand() < self.transition_prob:
            self.on = int(not self.on)
        # Determine the state based on the transition probability
        # If the interference is on, the state is the original state. Otherwise,
        # it is zero
        self.state_ind = self.original_state_ind * self.on
        self.current_state = self.states[self.state_ind]


class HoppingInterference():
    """Frequency-hopping interference object.

    Repeatedly hops frequency bins in a pre-defined pattern
    """

    def __init__(self, pattern, tx_power, states):
        """Create a HoppingInterference object

        Args:
            pattern: Numpy array of state indices defining the hopping pattern
            of the emitter
            tx_power: Transmit power (W)
            states: Numpy array of all possible interference states
        """
        self.pattern = pattern
        self.tx_power = tx_power
        self.states = states
        self.pattern_ind = 0
        self.state_ind = pattern[0]
        self.current_state = self.states[self.state_ind]

    def step(self):
        """Step to the next frequency in the sequence, restarting if the end is reached"""
        self.pattern_ind = (self.pattern_ind + 1) % len(self.pattern)
        self.state_ind = self.pattern[self.pattern_ind]
        self.current_state = self.states[self.state_ind]

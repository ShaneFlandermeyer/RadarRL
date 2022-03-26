import numpy as np
import scipy.constants as sc


class Radar:
    """Monostatic radar object"""

    def __init__(self, position=np.zeros((3,)), prf=1e3, center_freq=10e9,
                 tx_gain=100, tx_power=100, samp_rate=1e6, num_pulse_cpi=128, noise_fig=5, T0=270):
        self.position = np.array(position)
        self.prf = prf
        self.center_freq = center_freq
        self.tx_gain = tx_gain
        self.noise_fig = noise_fig
        self.T0 = T0
        # Assume monostatic
        self.rx_gain = tx_gain
        self.tx_power = tx_power
        self.samp_rate = samp_rate
        self.num_pulse_cpi = num_pulse_cpi
        # Derived parameters
        self.lambda0 = sc.c / center_freq
        self.max_range = sc.c/(2*prf)
        self.max_doppler = prf/2

        self.action = np.array([])

    def rx_power(self, target):
        R = np.linalg.norm(target.position - self.position)
        Pr = self.tx_power*self.tx_gain*self.rx_gain*self.lambda0**2 * \
            target.rcs/((4*sc.pi)**3*R**4)
        return Pr

    def SINR(self, target, interference, wave):
        noise_power = sc.k*self.T0*self.samp_rate*self.noise_fig
        # Interference power only contributes to SINR if it is in the same
        # frequency band
        if sum(self.action*interference.current_state) > 0:
            interference_power = interference.tx_power
        else:
            interference_power = 0
        sinr = self.rx_power(target) / (noise_power + interference_power)
        # SINR after pulse compression and coherent integration
        sinr *= self.num_pulse_cpi * (wave.pulsewidth*wave.bandwidth)
        sinr = 10*np.log10(sinr)

        return sinr


class Waveform:
    """Linear FM Waveform object"""

    def __init__(self, bandwidth, pulsewidth):
        self.bandwidth = bandwidth
        self.pulsewidth = pulsewidth

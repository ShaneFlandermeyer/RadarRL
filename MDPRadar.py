# %% [markdown]
#   # Final Project
# 

# %%
import mdptoolbox
import mdptoolbox.example
import numpy as np
import scipy.constants as sc
import itertools
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



# %% [markdown]
#   ## Radar Environment
# 

# %% [markdown]
#   The radar environment is defined by a set of possible postion states
#   $\mathcal{P}$ and a set of velocity states $\mathcal{V}$.
# 
#   $\mathcal{P} = \{\mathbf{r_1}, \mathbf{r_2}, \dots, \mathbf{r_\rho}\}$
# 
#   $\mathcal{V} = \{\mathbf{v_1}, \mathbf{v_2}, \dots, \mathbf{v_\nu}\}$
# 
#   where $\rho$ is the number of possible position states and $\nu$ is the number
#   of possible velocities.
# 
#   Each $\mathbf{r_i}, \mathbf{v_i}$ are 3-dimensional row vectors
# 
#   $\mathbf{r_i} = \left[r_x, r_y, r_z \right]$
# 
#   $\mathbf{v_i} = \left[v_x, v_y, v_z \right]$
# 
#   where x, y and z represent the cross-range, down-range, and vertical dimensions,
#   respectively.
# 

# %% [markdown]
#   For this simple script, only 1D range and velocity will be considered.
# 

# %% [markdown]
#   ## Simulation Objects
# 

# %%
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
        if sum(self.action*interference.state) > 0:
            interference_power = interference.inr*noise_power
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


class Target:
    """Point target object"""

    def __init__(self, position, velocity, rcs):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.rcs = rcs

    def step(self, dt):
        self.position += self.velocity*dt


class ConstantInterference:
    """Constant interference object"""

    def __init__(self, inr=14, states=np.array([]), state_ind=0):
        # ConstantInterference-to-noise ratio at the radar receiver. To simplify the
        # scenario, this is position-independent
        self.inr = inr
        self.states = states
        self.state_ind = state_ind
        self.state = states[state_ind]

    # Define frequency-hopping behavior
    def step(self):
        # Do nothing
        self.state_ind = self.state_ind
        self.state = self.states[self.state_ind]



# %% [markdown]
#   ### Interference Environment
# 

# %%
N = 5
channel_bw = 100e6
subband_bw = channel_bw / N
channel = np.zeros((N,))
# Matrix of all unique interference states, where each row is a unique state and
# each column is a frequency bin
Theta = np.array(list(itertools.product([0, 1], repeat=N)))
# Interfering system
interference_state = 8
comms = ConstantInterference(
    inr=10**(14/10), states=Theta, state_ind=interference_state)



# %% [markdown]
#   ## Simulation Environment

# %% [markdown]
#   ### Radar and Target Parameters

# %%
# Radar system
radar = Radar(position=np.array(0), prf=2000,
              center_freq=10e9, tx_gain=100, tx_power=1e3, num_pulse_cpi=20)
# Transmitted waveform (linear FM)
wave = Waveform(bandwidth=20e6, pulsewidth=10e-6)
# Possible actions (assume transmission over contiguous sub-bands)
actions = np.zeros((0, N))
for i in range(N):
    state = np.zeros((N,))
    state[:i+1] = 1
    actions = np.append(actions, np.array([np.roll(state, j) for j in
                                           range(N-i)]), axis=0)
# Number of position states
rho = 10
r = np.linspace(0, radar.max_range, rho)
# Number of velocity states
nu = 10
v = np.linspace(-1/2, 1/2, nu) * (radar.prf*radar.lambda0/2)
target = Target(position=[], velocity=[], rcs=0.1)

# %% [markdown]
#   ### Reward structure
# 

# %% [markdown]
#   Use the reward function defined in table 2 of Selvi2020
# 

# %%
def reward(sinr, bw):
    r = 0
    # SINR reward structure
    if sinr < 0:
        r -= 35
    elif sinr >= 0 and sinr <= 2:
        r += 1
    elif sinr > 2 and sinr <= 5:
        r += 2
    elif sinr > 5 and sinr <= 8:
        r += 3
    elif sinr > 8 and sinr <= 11:
        r += 4
    elif sinr > 11 and sinr <= 14:
        r += 5
    elif sinr > 14 and sinr <= 17:
        r += 6
    elif sinr > 17 and sinr <= 20:
        r += 8
    else:
        r += 10

    # Bandwidth reward structure
    r += 10*(bw-1)
    return r



# %% [markdown]
#   ## Train the MDP

# %%
# Number of possible states
# For this simulation, the set of possible states denotes all possible
# combinations of target position states, target velocity states, and
# interference states.
S = rho*nu*2**N
# Number of possible actions
A = actions.shape[0]
# Initialize the transition and reward matrices
# The first index is the action, the second is the initial state, and the third
# is the final state. The indexing for the states is position -> velocity ->
# interference-major. That is, for a given position, all velocities are
# enumerated before incrementing the position, and for a given velocity all
# interference states are enumerated. You can think of the position as the "most
# significant bit" and interference as the "least significant bit".
T = np.zeros((A, S, S))
R = np.zeros((A, S, S))

num_train = int(1e3)
num_test = int(1e2)
time = np.linspace(0, 1500, 25)
# Time step for the simulation
dt = time[1] - time[0]
for itrain in range(num_train):
    # Randomly select a starting position and target velocity
    target.position = np.random.choice(r)
    target.velocity = np.random.choice(v)
    # Add a gaussian perturbance to the position and velocity
    target.position += np.random.randn()
    target.velocity += np.random.randn()

    for t in time:
        # Calculate the initial state
        pos_state_i = np.digitize(target.position, r)-1
        vel_state_i = np.digitize(target.velocity, v)-1
        int_state_i = comms.state_ind
        state_i = pos_state_i*nu*(2**N) + vel_state_i*(2**N) + int_state_i
        # Randomly select a valid action
        action_index = np.random.randint(0, A)
        radar.action = actions[action_index, :]
        # Determine the bandwidth used, then update the interference, position, range, and SINR
        wave.bandwidth = subband_bw*np.sum(radar.action)
        comms.step()
        target.step(dt)
        sinr = radar.SINR(target, comms, wave)
        # Determine the new state
        pos_state_f = np.digitize(target.position, r)-1
        vel_state_f = np.digitize(target.velocity, v)-1
        int_state_f = comms.state_ind
        state_f = pos_state_f*nu*(2**N) + vel_state_f*(2**N) + int_state_f
        # Update the transition and reward matrices
        T[action_index, state_i, state_f] += 1
        R[action_index, state_i, state_f] += reward(sinr, np.sum(radar.action))
# Normalize the transition probability matrix to make it stochastic
T = np.array([normalize(T[a], axis=1, norm='l1') for a in range(A)])
# Also need to add a 1 to the diagonals of the matrices where the probability is zero
for a in range(A):
    ind = np.where(T[a].sum(axis=1) == 0)[0]
    for i in ind:
        T[a, i, i] = 1
        R[a, i, i] = 0

# Use policy iteration to determine the optimal policy
pi = mdptoolbox.mdp.PolicyIteration(T, R, 0.9)
pi.run()
# print(pi.policy)

# %% [markdown]
# ## Test the MDP

# %%
current_reward = np.zeros((time.shape[0],num_test))
current_sinr = np.zeros((time.shape[0],num_test))
for itest in range(num_test):
    # Select a NEW trajectory that was not used for training
    # Randomly select a starting position and target velocity
    target.position = np.random.choice(r)
    target.velocity = np.random.choice(v)
    # Add a gaussian perturbance to the position and velocity
    target.position += np.random.randn()
    target.velocity += np.random.randn()
    for itime in range(time.shape[0]):
        # Calculate the initial state
        pos_state_i = np.digitize(target.position, r)-1
        vel_state_i = np.digitize(target.velocity, v)-1
        int_state_i = comms.state_ind
        state_i = pos_state_i*nu*(2**N) + vel_state_i*(2**N) + int_state_i
        # Select an action from the policy
        radar.action = actions[pi.policy[state_i], :]
        # Determine the bandwidth used, then update the interference, position, range, and SINR
        wave.bandwidth = subband_bw*np.sum(radar.action)
        comms.step()
        target.step(dt)
        sinr = radar.SINR(target, comms, wave)
        current_sinr[itime,itest] = sinr
        if itime > 0:
          current_reward[itime,itest] = current_reward[itime-1,itest] + reward(sinr, np.sum(radar.action))
        # Determine the new state
        pos_state_f = np.digitize(target.position, r)-1
        vel_state_f = np.digitize(target.velocity, v)-1
        int_state_f = comms.state_ind
        state_f = pos_state_f*nu*(2**N) + vel_state_f*(2**N) + int_state_f
current_reward = np.mean(current_reward, axis=1)
current_sinr = np.mean(current_sinr,axis=1)


# %% [markdown]
# ## Visualizations
plt.figure()
plt.plot(time,current_reward, '.-')
plt.xlabel('Time (s)')
plt.ylabel('Average Cumulative Reward')

plt.figure()
plt.plot(time,current_sinr, '.-')
plt.xlabel('Time (s)')
plt.ylabel('Average SINR (dB)')
# %%

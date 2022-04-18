from gym.spaces import Discrete
import numpy as np
import functools
import itertools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.test import api_test
import matplotlib.pyplot as plt



N = 5
MOVES = np.array(list(itertools.product([0, 1], repeat=N)))
NUM_ITERS = 100


def RadarEnvironment():
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_RadarEnvironment()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_RadarEnvironment(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {'render_modes': ['human'], "name": "rps_v2"}

    def __init__(self):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        # self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = ["Radar_0", "Comms_0"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self._action_spaces = {agent: Discrete(
            2**N) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(
            2**N) for agent in self.possible_agents}
        

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Discrete(2**N)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2**N)

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if len(self.agents) == 2:
            string = ("Current state: Radar: {} , Comms: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: 0 for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}
        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.reward()
            # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[(
            #     self.state[self.agents[0]], self.state[self.agents[1]])]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >=
                          NUM_ITERS for agent in self.agents}

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 -
                                                              self.agent_name_mapping[i]]]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = 0
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def reward(self):
      # TODO: Generalize this so that each agent can have a unique reward
      # TODO: Generalize this to more than one agent
      radar_state = MOVES[self.state[self.agents[0]]]
      comms_state = MOVES[self.state[self.agents[1]]]

      r = 0
      # Number of collisions with the interference
      num_collision = np.sum(np.equal(radar_state, 1) &
                            np.equal(comms_state,1))
      # Number of sub-bands utilized by the radar
      num_subband = np.sum(radar_state)
      # Number of missed opportunities for radar transmission, where no
      # interference exists but the radar did not transmit there
      num_missed_opportunity = np.sum(
          (radar_state == 0) & (comms_state == 0))

      r += -45*num_collision
      r += 10*(num_subband-1)
      self.rewards[self.agents[0]] = r


RadarEnvironment = RadarEnvironment()

pattern = [1,2]
ind = 0
def policy(observation, agent):
    global ind
    if agent == "Radar_0":
        action = np.random.randint(0,2**N)
    elif agent == "Comms_0":
        action = pattern[ind % len(pattern)]
        ind += 1
    return action

# Run a basic simulation
RadarEnvironment.reset()
for agent in RadarEnvironment.agent_iter():
    observation, reward, done, info = RadarEnvironment.last()
    action = policy(observation, agent) if not done else None
    RadarEnvironment.step(action)
    if agent == 'Comms_0':
        RadarEnvironment.render()
        print(RadarEnvironment._cumulative_rewards)

    
    
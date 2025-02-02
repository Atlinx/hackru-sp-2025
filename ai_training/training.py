from bus_world_env import BusWorldEnv
from neural_network import NeuralNetwork
import numpy as np
import pprint

env = BusWorldEnv(render_mode="human")
state, _ = env.reset()
pprint.pp(env.get_full_state())

while True:
    # env.step(None)
    env.render()


# def decay(eps_start, eps_end, eps_decayrate, current_step):
#     #                          1            ^ (eps_decayrate * current_step)
#     # eps_start +  -----------------------
#     #               (eps_start - eps_end)
#     return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decayrate * current_step)


# env.debug_observation_sample()

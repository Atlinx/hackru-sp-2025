import pygame.freetype
from bus_world_env import BusWorldEnv
from neural_network import NeuralNetwork
import numpy as np
import pprint

# env = BusWorldEnv(
#     render_mode="human",
#     routes=[
#         {
#             "name": "A",
#             "stops": [(0, 10), (1, 48)],  # stores (stop_id, prev_dist)
#         },
#         {"name": "B", "stops": [(1, 15), (2, 15), (0, 30)]},
#         {"name": "C", "stops": [(0, 15), (1, 15)]},
#     ],
#     switch_costs=[
#         (0, 1, 15),
#         (1, 2, 15),
#         (2, 0, 15),
#     ],
#     bus_count=2,
#     stops_count=3,
#     max_students=30,
# )
env = BusWorldEnv(render_mode="human")
state, _ = env.reset()
pprint.pp(env.get_full_state())

while True:
    env.step([])
    env.render()


# def decay(eps_start, eps_end, eps_decayrate, current_step):
#     #                          1            ^ (eps_decayrate * current_step)
#     # eps_start +  -----------------------
#     #               (eps_start - eps_end)
#     return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decayrate * current_step)


# env.debug_observation_sample()

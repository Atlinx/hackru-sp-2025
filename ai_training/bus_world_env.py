import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import numpy as np
import pprint
from enum import IntEnum
from collections import defaultdict

import pygame.freetype


class BusState(IntEnum):
    ON_ROUTE = 0
    AT_STOP_UNLOADING = 1
    AT_STOP_BREAK = 2
    SWITCHING_ROUTE = 3


class BusWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        max_students=100,
        stops=3,
        routes=[
            {"name": "A", "stops": [(0, 32), (1, 48)], "length": 54},
            {"name": "B", "stops": [(1, 10), (2, 50), (0, 78)], "length": 138},
            {"name": "C", "stops": [(0, 15), (1, 20)], "length": 25},
        ],
        switch_costs=[
            (0, 1, 30),
            (1, 2, 20),
            (2, 1, 15),
        ],
        buses=5,
        window_size=512,
    ):
        """
        routes: [
            {
                name: XX
                stops: [
                    (id, dist)
                    (id, dist)
                    (id, dist)
                ]
            }
        ]
        """
        super(BusWorldEnv, self).__init__()
        # initialize state data
        max_route_length: int = max([r["length"] for r in routes])

        self.render_mode = render_mode
        self.base_routes = routes

        # [edge (a, b)] -> [routes]
        self.edges_to_base_routes = defaultdict(list)

        def add_edge_to_base_route(a, b, route):
            if a > b:
                a, b = b, a
            self.edges_to_base_routes[(a, b)].append(route)

        for route_id, route in enumerate(routes):
            route_stops = route["stops"]
            add_edge_to_base_route(
                route_stops[len(route_stops) - 1][0], route_stops[0][0], route_id
            )
            for i in range(1, len(route_stops)):
                add_edge_to_base_route(route_stops[i], route_stops[i - 1], route_id)

        self.state = {}
        self.runtime_state = {}
        self.stops_adj_mat = np.full(shape=(stops, stops), fill_value=-1)
        for a, b, cost in switch_costs:
            self.stops_adj_mat[a, b] = cost
            self.stops_adj_mat[b, a] = cost

        # A snapshot of the world of buses and students
        self.observation_space = spaces.Dict(
            {
                "routes": spaces.Tuple(
                    [
                        spaces.Dict(
                            {
                                # bus stops
                                "stops": spaces.Tuple(
                                    [
                                        spaces.Dict(
                                            {
                                                # students[destination] = count			Array of students and destination
                                                "students": spaces.Tuple(
                                                    [
                                                        spaces.Box(
                                                            low=0,
                                                            high=max_students,
                                                            dtype=np.int64,
                                                        )
                                                        for _ in routes
                                                    ]
                                                ),
                                            }
                                        )
                                        for stop_pos in stops
                                    ]
                                )
                            }
                        )
                        for name, stops, length in routes
                    ]
                ),
                # buses[] = (position, students)[]		Array of buses
                "buses": spaces.Tuple(
                    [
                        spaces.Dict(
                            {
                                # route that the bus is on (index of route)
                                "route": spaces.Discrete(len(routes)),
                                # state of the bus
                                # 0 -> on_route
                                # 1 -> at stop unloading
                                # 2 -> at stop unloading
                                # 3 -> switching_to_current_route
                                "state": spaces.Discrete(4),
                                # position of the bus
                                "position": spaces.Box(low=0, high=max_route_length),
                                # students on the bus and their destinations
                                # students[destination] = count
                                "students": spaces.Tuple(
                                    [
                                        spaces.Box(low=0, high=max_students)
                                        for _ in routes
                                    ]
                                ),
                            }
                        )
                        for _ in range(buses)
                    ]
                ),
            }
        )

        route_names = [r["name"] for r in routes]

        # Valid actions (array)
        # - SWAP -> swap bus between routes
        self.action_space = spaces.Sequence(
            spaces.Dict(
                {
                    # index of bus we want to move
                    "bus": spaces.Discrete(buses),
                    # new route the bus should be on
                    "new_route": spaces.Discrete(len(routes)),
                }
            ),
        )

        # initalize pygame
        self.window_size = window_size
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((window_size, window_size))

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}

    def _get_routes(self):
        return self.state["routes"]

    def _get_buses(self):
        return self.state["buses"]

    def get_full_state(self):
        return {
            "base_routes": self.base_routes,
            "stops_adj_mat": self.stops_adj_mat,
            "state": self.state,
            "runtime_state": self.runtime_state,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation_space.seed(seed)
        self.state = self.observation_space.sample()
        self.runtime_state = {"buses": []}
        for bus in self.state["buses"]:
            # bus: { position, route, state, students }
            # bus (runtime): { timer, stop }

            # fix positions of buses (to avoid exceeding bus route lengths)
            route = bus["route"]
            base_route = self.base_routes[route]

            if bus["position"][0] > base_route["length"]:
                bus["position"][0] %= base_route["length"]

            # update bus depending on state
            timer = 0
            curr_stop_id, curr_stop_dist = base_route["stops"][0]

            for stop_id, stop_dist in base_route["stops"]:
                if stop_dist > bus["position"][0]:
                    curr_stop_id, curr_stop_dist = stop_id, stop_dist
                    break
                match bus["state"]:
                    case BusState.AT_STOP_UNLOADING:
                        timer = self.np_random.integers(3, 5 + 1)
                        bus["position"][0] = curr_stop_dist
                        break
                    case BusState.AT_STOP_BREAK:
                        timer = self.np_random.integers(5, 20 + 1)
                        bus["position"][0] = curr_stop_dist
                        break
                    case BusState.SWITCHING_ROUTE:
                        stop_edges = list(
                            filter(lambda x: x >= 0, self.stops_adj_mat[curr_stop_id])
                        )
                        if len(stop_edges) > 0:
                            timer = stop_edges[
                                self.np_random.integers(0, len(stop_edges))
                            ]
                        else:
                            timer = 0
                        bus["position"][0] = curr_stop_dist
                        break

            # add additional data
            self.runtime_state["buses"].append(
                {"timer": timer, "stop_id": curr_stop_id}
            )

        return self._get_obs(), self._get_info()

    def step(self, action):
        # action: [{bus: 0, new_route: 2}, {bus: 2, new_route: 1}, ...]
        reward = 0
        terminated = False
        truncated = False
        # for bus, new_route in action:
        #     for bus in self.state["buses"]:
        #         # bus: { position, route, state, students }
        #         print("pos", self.state["buses"][0]["position"][0])
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            pygame.event.get()
            BLACK = (0, 0, 0)
            GREEN = (0, 255, 0)

            font = pygame.freetype.SysFont("Comic Sans MS", 16)

            # draw bg
            self.screen.fill(BLACK)
            stops_count = len(self.stops_adj_mat)
            angle_interval = 2 * np.pi / stops_count
            dist = self.window_size / 2 / 1.3

            stop_pos_list = []
            for i in range(stops_count):
                curr_angle = i * angle_interval
                center = self.window_size / 2
                pos = (
                    self.window_size / 2 + dist * np.cos(curr_angle),
                    self.window_size / 2 + dist * np.sin(curr_angle),
                )
                stop_pos_list.append(pos)

            # draw edges
            EMPTY_EDGE_COLOR = pygame.Color("#949494")
            ROUTE_COLORS = [
                pygame.Color(c)
                for c in [
                    "#eb4034",
                    "#eb9f34",
                    "#ebdc34",
                    "#a5eb34",
                    "#3aeb34",
                    "#34ebb1",
                    "#34d9eb",
                    "#3480eb",
                    "#3437eb",
                    "#9c34eb",
                    "#eb34e2",
                    "#eb3480",
                ]
            ]

            for a in range(len(self.stops_adj_mat)):
                for b in range(a + 1, len(self.stops_adj_mat)):
                    if self.stops_adj_mat[a, b] > 0:
                        pygame.draw.line(
                            self.screen,
                            EMPTY_EDGE_COLOR,
                            stop_pos_list[b],
                            stop_pos_list[a],
                            4,
                        )
                        # draw edge if it exists
                        X, Y = 0, 1
                        vec = (
                            stop_pos_list[b][X] - stop_pos_list[a][X],
                            stop_pos_list[b][Y] - stop_pos_list[a][Y],
                        )
                        perp_vec = (-vec[Y], vec[X])
                        # find all routes using this edge
                        edges = self.edges_to_base_routes[(a, b)]

            # draw points
            for i, pos in enumerate(stop_pos_list):
                pygame.draw.circle(self.screen, EMPTY_EDGE_COLOR, pos, 8)
                font.render_to(str(i), (255, 255, 255), size=16)
                self.screen.blit(text_surface, (pos[0] + 16, pos[1] + 16), rect)

            pygame.display.update()  # Update the display

    def debug_observation_sample(self):
        pprint.pp(self.observation_space.sample())

    def debug_action_sample(self):
        pprint.pp(self.action_space.sample())


if __name__ == "__main__":
    env = BusWorldEnv()
    for i in range(1000):
        env.debug_observation_sample()
        env.debug_action_sample()

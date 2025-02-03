import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import numpy as np
import pprint
from enum import IntEnum
from collections import defaultdict
import pygame_utils as pu
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
        render_mode: str = None,
        max_students: int = 100,
        stops_count: int = 6,
        routes=[
            {
                "name": "A",
                "stops": [(0, 10), (1, 48)],  # stores (stop_id, prev_dist)
            },
            {"name": "B", "stops": [(1, 15), (2, 15), (0, 30)]},
            {"name": "C", "stops": [(0, 15), (1, 15)]},
            {"name": "D", "stops": [(4, 15), (5, 15), (1, 15)]},
            {"name": "E", "stops": [(3, 15), (1, 15), (4, 15), (5, 15)]},
            {"name": "F", "stops": [(3, 15), (4, 15)]},
        ],
        switch_costs=[
            (0, 1, 15),
            (1, 2, 15),
            (2, 0, 15),
            (4, 5, 15),
            (5, 3, 15),
            (3, 4, 15),
            (3, 1, 15),
            (1, 4, 15),
            (1, 5, 15),
            (3, 2, 15),
            (5, 0, 15),
            (5, 2, 15),
        ],
        bus_count: int = 10,
        bus_speed: float = 1,
        settings={
            "bus_size": 10,  # number of students we can store on each bus
            "unload_timespan": (3, 5),
            "break_timespan": (5, 20),
            "break_on_stop_prob": 0.05,
        },
        render_delta_ms: int = 41,
        tick_delta: int = 1,
        window_size: int = 720,
    ):
        """
        routes: [
            {
                name: XX
                stops: [
                    (id, prev_dist)
                    (id, prev_dist)
                    (id, prev_dist)
                ]
            }
        ]

        base_routes: [
            {
                name: XX
                stops: [
                    (id, prev_dist, tot_dist)        tot_dist -> running total distance
                    (id, prev_dist, tot_dist)
                    (id, prev_dist, tot_dist)
                ],
                length: int
            }
        ]
        """
        super(BusWorldEnv, self).__init__()
        # initialize state data

        self.settings = settings
        self.bus_speed = bus_speed
        self.bus_tick_move_delta = bus_speed * tick_delta
        self.render_delta_ms = render_delta_ms
        self.tick_delta = tick_delta
        self.render_mode = render_mode
        self.base_routes = routes
        max_route_length: int = 0
        for route in routes:
            tot_dist = 0
            stops_set = set()
            for i in range(len(route["stops"])):
                id, prev_dist = route["stops"][i]
                stops_set.add(id)
                if i > 0:
                    tot_dist += prev_dist
                route["stops"][i] = (id, prev_dist, tot_dist)
            route["length"] = tot_dist + route["stops"][0][1]
            max_route_length = max(max_route_length, tot_dist)
            route["stops_set"] = stops_set

        # [edge (a, b)] -> [routes]
        self.edges_to_base_routes = defaultdict(set)

        def add_edge_to_base_route(a, b, route):
            if a > b:
                a, b = b, a
            self.edges_to_base_routes[(a, b)].add(route)

        for route_id, route in enumerate(routes):
            route_stops = route["stops"]
            add_edge_to_base_route(
                route_stops[len(route_stops) - 1][0], route_stops[0][0], route_id
            )
            for i in range(1, len(route_stops)):
                add_edge_to_base_route(
                    route_stops[i][0], route_stops[i - 1][0], route_id
                )

        self.state = {}
        self.runtime_state = {}
        self.stops_adj_mat = np.full(shape=(stops_count, stops_count), fill_value=-1)
        for a, b, cost in switch_costs:
            self.stops_adj_mat[a, b] = cost
            self.stops_adj_mat[b, a] = cost

        # A snapshot of the world of buses and students
        self.observation_space = spaces.Dict(
            {
                # stops[] = {
                #   students: [dest_stop1_count, dest_stop2_count, dest_stop3_count, ...]
                # }
                "stops": spaces.Tuple(
                    [
                        spaces.Dict(
                            {
                                "students": spaces.Tuple(
                                    [
                                        spaces.Discrete(max_students)
                                        for _ in range(stops_count)
                                    ]
                                ),
                            }
                        )
                        for _ in range(stops_count)
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
                                # students on the bus and their destination stops
                                # students[destination] = count
                                "students": spaces.Tuple(
                                    [
                                        spaces.Discrete(max_students)
                                        for _ in range(stops_count)
                                    ]
                                ),
                            }
                        )
                        for _ in range(bus_count)
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
                    "bus": spaces.Discrete(bus_count),
                    # new route the bus should be on
                    "new_route": spaces.Discrete(len(routes)),
                    "new_stop_index": spaces.Discrete(len(self.stops_adj_mat)),
                }
            ),
        )

        # initalize pygame
        self.window_size = window_size
        if self.render_mode == "human":
            pygame.init()
            pygame.freetype.init()
            self.screen = pygame.display.set_mode(
                (window_size, window_size), pygame.RESIZABLE
            )
            self.font = pygame.freetype.Font("helvetica-bold.ttf", 16)

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}

    def get_full_state(self):
        return {
            "bus_speed": self.bus_speed,
            "tick_delta": self.tick_delta,
            "render_delta_ms": self.render_delta_ms,
            "edges_to_base_routes": self.edges_to_base_routes,
            "base_routes": self.base_routes,
            "stops_adj_mat": self.stops_adj_mat,
            "state": self.state,
            "runtime_state": self.runtime_state,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation_space.seed(seed)
        self.state = self.observation_space.sample()
        self.runtime_state = {"buses": [], "stops": []}
        students_waiting = [0] * len(self.stops_adj_mat)
        students_destination = [0] * len(self.stops_adj_mat)
        for i, stops in enumerate(self.state["stops"]):
            stops["students"] = list(stops["students"])
            stops["students"][i] = 0  # students can't wait on stop they're already on
            students_waiting[i] = np.sum(stops["students"])
            for ci, count in enumerate(stops["students"]):
                students_destination[ci] += count
        for i, stops in enumerate(self.state["stops"]):
            self.runtime_state["stops"].append(
                {
                    "students_waiting": students_waiting[i],
                    "students_destination": students_destination[i],
                }
            )
        for bus in self.state["buses"]:
            # bus: { position, route, state, students }
            # bus (runtime): { timer, stop }

            # fix positions of buses (to avoid exceeding bus route lengths)
            route = bus["route"]
            base_route = self.base_routes[route]
            total_students = np.sum(bus["students"])
            bus["students"] = [0] * len(bus["students"])

            if bus["position"][0] > base_route["length"]:
                bus["position"][0] %= base_route["length"]

            # update bus depending on state
            timer = 0
            curr_stop_index = 0
            curr_stop_id, _, curr_stop_dist = base_route["stops"][0]

            for stop_index, (stop_id, _, stop_dist) in enumerate(base_route["stops"]):
                if stop_dist > bus["position"][0]:
                    curr_stop_index, curr_stop_id, curr_stop_dist = (
                        stop_index,
                        stop_id,
                        stop_dist,
                    )
                    break
                match bus["state"]:
                    case BusState.AT_STOP_UNLOADING:
                        timer = self.np_random.integers(
                            self.settings["unload_timespan"][0],
                            self.settings["unload_timespan"][1] + 1,
                        )
                        bus["position"][0] = curr_stop_dist
                        break
                    case BusState.AT_STOP_BREAK:
                        timer = self.np_random.integers(
                            self.settings["break_timespan"][0],
                            self.settings["break_timespan"][1] + 1,
                        )
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
            dist_left = bus["position"][0] - curr_stop_dist

            # add additional data
            self.runtime_state["buses"].append(
                {
                    "timer": timer,
                    "timer_duration": timer,
                    "stop_index": curr_stop_index,
                    "dist_left": dist_left,
                }
            )

        return self._get_obs(), self._get_info()

    def step(self, action):
        # action: [{bus: 0, new_route: 2, new_stop_index: 2}, {bus: 2, new_route: 1, new_stop_index: 2}, ...]
        reward = 0
        terminated = False
        truncated = False

        # use actions
        for bus_id, new_route_id, new_stop_index in action:
            try:
                # bus: { position, route, state, students }
                bus = self.state["buses"][bus_id]
                bus_runtime = self.runtime_state["buses"][bus_id]

                old_stop_id, _, old_stop_tot_dist = self.base_routes[bus["route"]][
                    "stops"
                ][bus_runtime["stop_index"]]
                new_stop_id, _, new_stop_tot_dist = self.base_routes[new_route_id][
                    "stops"
                ][new_stop_index]

                if abs(old_stop_tot_dist - bus["position"]) >= 0.05:
                    # bus must be at a stop in order to switch (within 0.05)
                    raise Exception(
                        "Bus must be <= 0.05 meter of stop to be considered resting and available for swap"
                    )

                distance_to_new_stop = self.stops_adj_mat[old_stop_id, new_stop_id]
                travel_time = distance_to_new_stop / self.bus_speed

                bus["position"] = new_stop_tot_dist
                bus["state"] = int(BusState.SWITCHING_ROUTE)
                bus_runtime["timer"] = travel_time
                bus_runtime["timer_duration"] = travel_time
                bus["route"] = new_route_id
                bus_runtime["stop_index"] = new_stop_index
            except:
                print(f"⚠ Invalid action: {(bus_id, new_route_id, new_stop_index)}")

        def unload(route, bus, bus_runtime, stop_id):
            nonlocal reward
            stop_runtime = self.runtime_state["stops"][stop_id]
            students_getting_off = bus["students"][stop_id]
            stop_runtime["students_destination"] -= students_getting_off
            reward += students_getting_off
            bus["students"][stop_id] = 0
            stop_students = self.state["stops"][stop_id]["students"]
            curr_bus_size = sum(bus["students"])
            # enumerate students waiting at stop_id
            for dest_id in range(len(stop_students)):
                # if student's destination index is in our route, then
                # the student will board the bus
                if (
                    dest_id in route["stops_set"]
                    and curr_bus_size < self.settings["bus_size"]
                ):
                    # try to add everyone onto the bus, but limit it by capacity of bus
                    add_amount = stop_students[dest_id]
                    remaining_space = self.settings["bus_size"] - curr_bus_size
                    if add_amount > remaining_space:
                        add_amount = remaining_space

                    if add_amount > 0:
                        curr_bus_size += add_amount
                        bus["students"][dest_id] += add_amount
                        stop_students[dest_id] -= add_amount
                        stop_runtime["students_waiting"] -= add_amount

        # simulate
        for bus_id, bus in enumerate(self.state["buses"]):
            route = self.base_routes[bus["route"]]
            bus_runtime = self.runtime_state["buses"][bus_id]
            stop_id = route["stops"][bus_runtime["stop_index"]][0]

            # decrement timer
            if bus_runtime["timer"] >= 0:
                bus_runtime["timer"] -= self.tick_delta
                if bus_runtime["timer"] <= 0:
                    bus_runtime["timer"] = 0
                    # timer ended
                    match bus["state"]:
                        case BusState.AT_STOP_UNLOADING | BusState.AT_STOP_BREAK:
                            # Finished unloading/break -> moving to next point
                            bus["state"] = int(BusState.ON_ROUTE)
                            bus_runtime["stop_index"] = (
                                bus_runtime["stop_index"] + 1
                            ) % len(route["stops"])
                            _, new_stop_prev_dist, _ = route["stops"][
                                bus_runtime["stop_index"]
                            ]
                            bus_runtime["dist_left"] = new_stop_prev_dist
                            break
                        case BusState.SWITCHING_ROUTE:
                            # Finished transporting bus to new route
                            # TODO: Actually move students off of bus, and new students onto bus
                            bus["state"] = int(BusState.AT_STOP_UNLOADING)
                            unload(route, bus, bus_runtime, stop_id)
                            break

            # move bus
            if bus["state"] == BusState.ON_ROUTE:
                bus["position"][0] += self.bus_tick_move_delta
                bus_runtime["dist_left"] -= self.bus_tick_move_delta
                if bus_runtime["dist_left"] < 0:
                    # we hit destination, snap to destination
                    _, _, dest_stop_tot_dist = route["stops"][bus_runtime["stop_index"]]
                    bus_runtime["dist_left"] = 0
                    bus["position"][0] = dest_stop_tot_dist

                    if self.np_random.random() < self.settings["break_on_stop_prob"]:
                        # go on break
                        bus["state"] = int(BusState.AT_STOP_BREAK)
                    else:
                        # unload students
                        bus["state"] = int(BusState.AT_STOP_UNLOADING)
                        unload(route, bus, bus_runtime, stop_id)

        # TODO: Return reward
        # TODO: Terminate after X cycles

        total_waiting = sum(
            [x["students_waiting"] for x in self.runtime_state["stops"]]
        )
        total_dest = sum(
            [x["students_destination"] for x in self.runtime_state["stops"]]
        )

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            pygame.event.get()
            WHITE = (255, 255, 255)
            BLACK = (0, 0, 0)
            UNUSED_LINE_COLOR = pygame.Color("#e8d5b7")
            BG_COLOR = pygame.Color("#f8ecd1")

            # draw bg
            self.screen.fill(BG_COLOR)
            stops_count = len(self.stops_adj_mat)
            angle_interval = 2 * np.pi / stops_count
            dist = self.window_size / 2 / 1.5
            global_offset = (-self.window_size / 10, 0)

            pu.render_outline_text(
                self.screen, (16, 16), self.font, "bus sim", 24, WHITE, BLACK, 2
            )
            self.font.render_to(
                self.screen,
                (16, self.window_size - 16 - 12),
                str(self.np_random_seed),
                UNUSED_LINE_COLOR,
                size=12,
            )

            stop_pos_list = []
            stop_size_list = []
            for i in range(stops_count):
                curr_angle = i * angle_interval
                center = self.window_size / 2
                pos = (
                    global_offset[0] + self.window_size / 2 + dist * np.cos(curr_angle),
                    global_offset[1] + self.window_size / 2 + dist * np.sin(curr_angle),
                )
                stop_pos_list.append(pos)
                stop_size_list.append(1)

            # draw edges
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
            STOP_RADIUS = 4
            STOP_OUTLINE = 4
            FONT_SIZE = 20
            BUS_RADIUS = 6
            BUS_OUTLINE = 4
            LINE_SPACING = 6

            # draw unused edges
            for a in range(len(self.stops_adj_mat)):
                for b in range(a + 1, len(self.stops_adj_mat)):
                    if self.stops_adj_mat[a, b] > 0:
                        # draw edge if it exists
                        routes = self.edges_to_base_routes[(a, b)]
                        stop_size_list[a] += len(routes) - 1
                        stop_size_list[b] += len(routes) - 1
                        if len(routes) == 0:
                            # draw empty edge
                            pygame.draw.line(
                                self.screen,
                                UNUSED_LINE_COLOR,
                                stop_pos_list[b],
                                stop_pos_list[a],
                                LINE_SPACING,
                            )

            # draw used edges
            for a in range(len(self.stops_adj_mat)):
                for b in range(a + 1, len(self.stops_adj_mat)):
                    if self.stops_adj_mat[a, b] > 0:
                        # draw edge if it exists
                        routes = self.edges_to_base_routes[(a, b)]
                        if len(routes) > 0:
                            # draw colored line for every route's edge
                            X, Y = 0, 1
                            vec = (
                                stop_pos_list[b][X] - stop_pos_list[a][X],
                                stop_pos_list[b][Y] - stop_pos_list[a][Y],
                            )
                            perp_vec = pu.vec_norm((-vec[Y], vec[X]))
                            shift_vec = pu.vec_mult_s(
                                perp_vec, -LINE_SPACING * (len(routes) - 1) / 2.0
                            )
                            # find all routes using this edge
                            for i, route in enumerate(routes):
                                offset_vec = pu.vec_add_v(
                                    pu.vec_mult_s(perp_vec, i * LINE_SPACING), shift_vec
                                )
                                route_b_pos = pu.vec_add_v(stop_pos_list[b], offset_vec)
                                route_a_pos = pu.vec_add_v(stop_pos_list[a], offset_vec)
                                pygame.draw.line(
                                    self.screen,
                                    ROUTE_COLORS[route],
                                    route_b_pos,
                                    route_a_pos,
                                    LINE_SPACING + 2,
                                )

            # draw buses
            for i, bus in enumerate(self.state["buses"]):
                route_id, position = bus["route"], bus["position"][0]
                bus_runtime = self.runtime_state["buses"][i]

                route_stops = self.base_routes[route_id]["stops"]

                stop_id, prev_dist, tot_dist = route_stops[bus_runtime["stop_index"]]
                prev_stop_id, _, _ = route_stops[bus_runtime["stop_index"] - 1]

                dist_from_next = bus_runtime["dist_left"]
                dist_from_next_percent = dist_from_next / prev_dist
                students_on_bus = sum(bus["students"])

                pos = pu.vec_interp(
                    stop_pos_list[stop_id],
                    stop_pos_list[prev_stop_id],
                    dist_from_next_percent,
                )
                pygame.draw.circle(
                    self.screen, ROUTE_COLORS[route_id], pos, BUS_RADIUS + BUS_OUTLINE
                )
                pygame.draw.circle(self.screen, WHITE, pos, BUS_RADIUS)
                pu.render_outline_text(
                    self.screen,
                    (
                        pos[X] + int(BUS_RADIUS),
                        pos[Y] + int(BUS_RADIUS),
                    ),
                    self.font,
                    f"{i}: {students_on_bus}/{self.settings['bus_size']}",
                    FONT_SIZE,
                    WHITE,
                    ROUTE_COLORS[route_id],
                    2,
                )

            # draw points
            for i, (pos, size) in enumerate(zip(stop_pos_list, stop_size_list)):
                radius = max(size, 2) * STOP_RADIUS
                pygame.draw.circle(self.screen, BLACK, pos, radius + STOP_OUTLINE)
                pygame.draw.circle(self.screen, WHITE, pos, radius)
                students_waiting = self.runtime_state["stops"][i][
                    "students_waiting"
                ]  # students waiting at this stop
                students_destination = self.runtime_state["stops"][i][
                    "students_destination"
                ]  # students whose destination is this stop
                pu.render_outline_text(
                    self.screen,
                    (
                        pos[X] + int(radius + STOP_RADIUS),
                        pos[Y] + int(radius + STOP_RADIUS),
                    ),
                    self.font,
                    f"{i} (w: {students_waiting}, d: {students_destination})",
                    FONT_SIZE,
                    BLACK,
                    WHITE,
                    2,
                )

            pygame.display.update()  # Update the display
            pygame.time.delay(self.render_delta_ms)

    def debug_observation_sample(self):
        pprint.pp(self.observation_space.sample())

    def debug_action_sample(self):
        pprint.pp(self.action_space.sample())


if __name__ == "__main__":
    env = BusWorldEnv()
    for i in range(1000):
        env.debug_observation_sample()
        env.debug_action_sample()

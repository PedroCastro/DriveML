"""
Original Authors: Sanjeevan Ahilan, Julian Villella, David Rusu
Modified By: Benedikt Kolbeinsson, Jiaze Sun, Pedro Castro
"""
import os, sys

import gym
import numpy as np
import math
import time

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

from ray.rllib.utils import try_import_tf
tf = try_import_tf()

MODEL_NAME = 'model'

# This action space should match the input to the action(..) function below.
ACTION_SPACE = gym.spaces.Box(low=np.array([-1.0, -1.0]),
                              high=np.array([1.0, 1.0]),
                              dtype=np.float32)

# The maximum number of lanes we expect to see in any scenario.
MAX_LANES = 5

ACTIONS = [[0, 0, 0], [0, 0, 0]] # information fo residual actions
crash_flag = False # used for training to signal a flipped car

# This observation space should match the output of observation(..) below
## Check our 1-page summary to know more about each
OBSERVATION_SPACE = gym.spaces.Dict({
    'distance_from_center': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'heading_errors': gym.spaces.Box(low=-1.0, high=1.0, shape=(10,)),
    'speed': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'steering': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'ego_lane_dist': gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
    'ego_ttc': gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
    'speed_of_closest': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'proximity': gym.spaces.Box(low=-1e10, high=1e10, shape=(6,)),
    'headings_of_cars': gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
})

def sign(x):
    return 1-(x<0)*2

def vec_2d(v):
    assert len(v) >= 2
    return np.array(v[:2])

def reset():
    global ACTIONS
    ACTIONS = [[0, 0, 0], [0, 0, 0]]

def proximity_detection(OGM):
    """
    Detects other vehicles in the vicinity of the ego vehicle
    """
    boxes = []
    boxes += [OGM[11:25, 23:27], #front left
              OGM[11:25, 27:37], #front centre
              OGM[11:25, 37:41], #front right
              OGM[25:39, 23:27], #left
              OGM[25:39, 37:41], #right
              OGM[39:53, 27:37], #back
              ]
    output = np.array([b.max()>0 for b in boxes], np.float32)
    return output

# fix taken from https://gist.github.com/davidrusu/d144a2646c9597a0d412c7e795ada548#file-nv_heading_to_ego_heading-py
def nv_heading_to_ego_heading(nv_heading):
    heading = nv_heading + 90
    if heading < 0:
        heading += 360
    return heading

def ttc_by_path(ego, waypoint_paths, neighborhood_vehicle_states, ego_closest_wp):
    # ttc stands for 'time to collision'
    # also return relative lane distance in front

    global crash_flag
    crash_flag = False
    wps_with_lane_dist = []
    for path_idx, path in enumerate(waypoint_paths):
        lane_dist = 0.
        for w1, w2 in zip(path, path[1:]):
            wps_with_lane_dist.append((w1, path_idx, lane_dist))
            lane_dist += np.linalg.norm(w2.pos - w1.pos)
        wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

    # next we compute the TTC along each of the paths
    ttc_by_path_index = [1] * len(waypoint_paths)
    lane_dist_by_path_index = [1] * len(waypoint_paths)
    headings_of_cars = [0] * len(waypoint_paths)
    ####
    speed_of_closest = 1
    wps = [path[0] for path in waypoint_paths]
    ego_closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

    for v in neighborhood_vehicle_states:
        # find all waypoints that are on the same lane as this vehicle
        wps_on_lane = \
            [(wp, path_idx, dist) for wp, path_idx, dist in wps_with_lane_dist
             if wp.lane_id == v.lane_id]

        if wps_on_lane == []:
            # this vehicle is not on a nearby lane
            continue

        # find the closest waypoint on this lane to this vehicle
        nearest_wp, path_idx, lane_dist = \
            min(wps_on_lane, key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)))


        if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
            # this vehicle is not close enough to the path, this can happen
            # if the vehicle is behind the ego, or ahead past the end of
            # the waypoints
            continue

        if ego_closest_wp.lane_index == nearest_wp.lane_index:
            if np.linalg.norm(vec_2d(ego.position) - vec_2d(v.position)) < 6:
                crash_flag = True
        relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
        if abs(relative_speed_m_per_s) < 1e-5:
            relative_speed_m_per_s = 1e-5
        dist_wp_vehicle_vector = vec_2d(v.position) - vec_2d(nearest_wp.pos)
        # take into account the position of the car instead of its nearest waypoint
        direction_vector = np.array([math.cos(math.radians(nearest_wp.heading)), math.sin(math.radians(nearest_wp.heading))]).dot(dist_wp_vehicle_vector)
        dist_to_vehicle = lane_dist + sign(direction_vector)*(np.linalg.norm(vec_2d(nearest_wp.pos) - vec_2d(v.position)))
        ttc = dist_to_vehicle / relative_speed_m_per_s
        ttc = ttc/10
        lane_dist = dist_to_vehicle/100

        if lane_dist_by_path_index[path_idx] > lane_dist:
            if nearest_wp.lane_index == v.lane_index:
                headings_of_cars[path_idx] = math.sin(math.radians(nearest_wp.relative_heading(nv_heading_to_ego_heading(v.heading))))

                ## speed
            if ego_closest_wp.lane_index == v.lane_index:
                speed_of_closest = (v.speed - ego.speed)/120



        lane_dist_by_path_index[path_idx] = min(lane_dist_by_path_index[path_idx], lane_dist)
        if ttc <= 0:
            # discard collisions that would have happened in the past
            continue
        ttc_by_path_index[path_idx] = min(ttc_by_path_index[path_idx], ttc)

    return ttc_by_path_index, lane_dist_by_path_index, headings_of_cars, speed_of_closest


## original function extended to support 5 lanes
def ego_ttc_calc(ego, ego_lane_index, ttc_by_path, lane_dist_by_path):
    #ttc, lane distance from ego perspective

    ego_ttc = [0]*5
    ego_lane_dist = [0]*5

    # current lane is centre
    ego_ttc[2] = ttc_by_path[ego_lane_index]
    ego_lane_dist[2] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path)-1
    min_lane_index = 0
    if ego_lane_index+1 > max_lane_index:
        ego_ttc[3] = 0
        ego_lane_dist[3] = 0
        ego_ttc[4] = 0
        ego_lane_dist[4] = 0
    elif ego_lane_index+2 > max_lane_index:
        ego_ttc[3] = ttc_by_path[ego_lane_index+1]
        ego_lane_dist[3] = lane_dist_by_path[ego_lane_index+1]
        ego_ttc[4] = 0
        ego_lane_dist[4] = 0
    else:
        ego_ttc[3] = ttc_by_path[ego_lane_index+1]
        ego_lane_dist[3] = lane_dist_by_path[ego_lane_index+1]
        ego_ttc[4] = ttc_by_path[ego_lane_index+2]
        ego_lane_dist[4] = lane_dist_by_path[ego_lane_index+2]

    if ego_lane_index-1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
        ego_ttc[1] = 0
        ego_lane_dist[1] = 0
    elif ego_lane_index-2 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
        ego_ttc[1] = ttc_by_path[ego_lane_index-1]
        ego_lane_dist[1] = lane_dist_by_path[ego_lane_index-1]
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index-2]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index-2]
        ego_ttc[1] = ttc_by_path[ego_lane_index-1]
        ego_lane_dist[1] = lane_dist_by_path[ego_lane_index-1]
    return ego_ttc, ego_lane_dist

def observation(env_obs, update=True):
    """
    Transform the environment's observation into something more suited for your model
    """
    ego = env_obs.ego_vehicle_state
    waypoint_paths = env_obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    # heading errors in current lane in front of vehicle
    indices = [0,1,2,3,5,8,13,21,34,50]
    closest_path = [waypoint_paths[closest_wp.lane_index][i] for i in indices]
    heading_errors = [math.sin(math.radians(wp.relative_heading(ego.heading)))
                      for wp in closest_path]

    # get number of lanes in the start of the environment because the car is spawned in the air
    global NUM_LANES
    if env_obs.ego_vehicle_state.position[2] > 0.85:
        NUM_LANES = len(waypoint_paths)

    ego_lane_index = closest_wp.lane_index

    ttc_by_p, lane_dist_by_p, headings_of_cars_all_lanes, speed_of_closest = ttc_by_path(ego, waypoint_paths, env_obs.neighborhood_vehicle_states, closest_wp)
    ego_ttc, ego_lane_dist = ego_ttc_calc(ego, ego_lane_index, ttc_by_p, lane_dist_by_p)

    ego_ttc = np.array(ego_ttc)
    ego_lane_dist = np.array(ego_lane_dist)

    # model was trained on max 3 lanes, remove extra lane observations if lane > 3 for safety reasons (can be fixed by training on +3 lane environments)
    if sum(ego_ttc>0) > 3:
        ego_ttc[0] = 0
        ego_ttc[-1] = 0
    if sum(ego_lane_dist>0) > 3:
        ego_lane_dist[0] = 0
        ego_lane_dist[-1] = 0

    proximity = proximity_detection(env_obs.occupancy_grid_map[1])

    headings_of_cars = [0] * 3
    # current lane is centre
    headings_of_cars[1] = headings_of_cars_all_lanes[ego_lane_index]
    if headings_of_cars[1] == 0:
        headings_of_cars[1] = headings_of_cars_all_lanes[ego_lane_index]
    if ego_lane_index+1 > len(headings_of_cars_all_lanes) - 1:
        headings_of_cars[2] = 0
    else:
        headings_of_cars[2] = headings_of_cars_all_lanes[ego_lane_index+1]

    if ego_lane_index-1 < 0:
        headings_of_cars[0] = 0
    else:
        headings_of_cars[0] = headings_of_cars_all_lanes[ego_lane_index-1]

    # reset steering history 
    if update:
        if env_obs.ego_vehicle_state.position[2] > 0.85:
            reset()

    closest_path = [waypoint_paths[closest_wp.lane_index][i] for i in np.arange(51)]
    headings = [math.sin(math.radians(wp.heading))
                      for wp in closest_path]
    headings = np.array(headings)

    return {
        'distance_from_center': np.array([norm_dist_from_center]),
        'heading_errors': np.array(heading_errors),
        'speed': np.array([ego.speed/120]),
        'steering': np.array([ego.steering]),
        'ego_lane_dist': np.array(ego_lane_dist),
        'ego_ttc': np.array(ego_ttc),
        'speed_of_closest': np.array([speed_of_closest]),
        'proximity': proximity,
        'headings_of_cars': np.array(headings_of_cars)
        }


def reward(env_obs, env_reward):
    """
    Here you can perform your reward shaping.

    The default reward provided by the environment is the increment in
    distance travelled. Your model will likely require a more
    sophisticated reward function
    """
    global crash_flag
    obs = observation(env_obs, False)

    center_penalty = -np.abs(obs['distance_from_center'])
    speed_reward = obs['speed']

    # penalize flip occurences (taking into account that the vehicle spawns in the air)
    if env_obs.ego_vehicle_state.speed >= 25:
        if env_obs.ego_vehicle_state.position[2] > 0.85:
            flip_penalty = -2 * env_obs.ego_vehicle_state.speed
        else:
            flip_penalty = 0
    else:
        flip_penalty = 0

    # penalise sharp turns done at high speeds
    if env_obs.ego_vehicle_state.speed > 60:
        steering_penalty = -pow((env_obs.ego_vehicle_state.speed-60)/20 * (env_obs.ego_vehicle_state.steering)*45/4, 2)
    else:
        steering_penalty = 0

    # penalise close proximity to other cars
    if crash_flag:
        crash_penalty = -5
    else:
        crash_penalty = 0

    total_reward = np.sum([1.0 * env_reward,
                           ])
    total_penalty = np.sum([0.1 * center_penalty,
                           1 * steering_penalty,
                           crash_penalty
                           ])

    if flip_penalty != 0:
        return float((-total_reward + total_penalty) / 200.0)
    else:
        return  float((total_reward + total_penalty) / 200.0)


def action(model_action):
    """
    Here you can perform you action space shaping.

    The built in action space may not be suited for all approaches, for
    example you may like to use a discrete action space.
    """

    global ACTIONS
    ACTIONS[1] = ACTIONS[0]

    acceleration, residual_steering = model_action
    if acceleration >= 0:
        throttle = acceleration
        brake = 0
    else:
        throttle = 0
        brake = -acceleration

    # add residual steering and save it for next time step
    new_steering = ACTIONS[1][2] + (np.clip(residual_steering, -1.0, 1.0) * 25.0)
    new_steering = np.clip(new_steering, -45, 45)

    ACTIONS[0] = [throttle, brake, new_steering]

    return np.array([throttle, brake, new_steering])


# #See: https://github.com/ray-project/ray/blob/b89cac976ae171d6d9b3245394e4932288fc6f11/rllib/models/tf/fcnet_v2.py#L14
class Model(FullyConnectedNetwork):
    pass

ModelCatalog.register_custom_model(MODEL_NAME, Model)

class EvaluationModel(TFModelV2):
    def __init__(self, suffix):
        super().__init__(
            obs_space=OBSERVATION_SPACE,
            action_space=ACTION_SPACE,
            num_outputs=ACTION_SPACE.shape[0],
            model_config={},
            name="evaluation-model")

        self._prep = ModelCatalog \
            .get_preprocessor_for_space(OBSERVATION_SPACE)

        self._sess = tf.Session(graph=tf.Graph())

        model_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'model'.format(suffix))

        self.base_model = tf.saved_model.load(self._sess,
                                              export_dir=model_path,
                                              tags=['serve'])

    def forward(self, obs, state, seq_lens):
        obs = self._prep.transform(obs)
        graph = self._sess.graph
        # These tensor names were found by inspecting the trained model
        # We use Tensor("split") instead of Tensor("add") to force PPO to be deterministic, CRUCIAL FOR SAFETY
        output_node = graph.get_tensor_by_name("default_policy/split:0")
        input_node = graph.get_tensor_by_name("default_policy/observation:0")
        res = self._sess.run(output_node, feed_dict={input_node: [obs]})
        action = res[0]
        return action, state


class Policy():
    def setup(self):
        self.model = EvaluationModel("normal")

    def teardown(self):
        pass

    def act(self, observation):
        action, _ = self.model.forward(observation, None, None)

        return action

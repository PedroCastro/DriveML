"""
Original Authors: Sanjeevan Ahilan, Julian Villella, David Rusu
Modified By: Benedikt Kolbeinsson, Jiaze Sun, Pedro Castro
"""
import os

import gym
from hiway.sumo_scenario import SumoScenario

from policy import Policy, OBSERVATION_SPACE, ACTION_SPACE, reward, observation, action


# Path to the scenario to test
scenario_path = '../tracks/2lane_sharp_bwd'

scenario = SumoScenario(
    scenario_root=scenario_path,
    random_social_vehicle_count=10)

env = gym.make('gym_hiway:hiway-competition-v0',
               config={
                   'sumo_scenario': scenario,
                   'headless': False,
                   'visdom': False,
                   'seed': 41,
                   'max_step_length': 10000,
                   'observation_space': OBSERVATION_SPACE,
                   'action_space': ACTION_SPACE,
                   'reward_function': reward,
                   'observation_function': observation,
                   'action_function': action,
               })


agent_policy = Policy()

agent_policy.setup()

observation = env.reset()

total_reward = 0.
for i in range(1000):
    action = agent_policy.act(observation)
    observation, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        print("simulation ended")
        break

env.close()
agent_policy.teardown()

print("Accumulated reward:", total_reward)

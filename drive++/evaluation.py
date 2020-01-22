"""
Original Authors: Benedikt Kolbeinsson, Jiaze Sun, Pedro Castro
"""
import os

import gym
from hiway.sumo_scenario import SumoScenario

from policy import Policy, OBSERVATION_SPACE, ACTION_SPACE, reward, observation, action
    

tracks = ["1lane", "1lane", "2lane_sharp_bwd", "3lane_bwd", "3lane_sharp_bwd_b", "3lane_sharp_bwd_b", "3lane_sharp_bwd_b"]
vehicles = [0, 10, 10, 10, 10, 25, 50]
seeds = [649, 343, 897, 674, 233, 877, 167]

agent_policy = Policy()
agent_policy.setup()

evaluation_reward = 0
for track, nvehicle, seed in zip(tracks, vehicles, seeds):
    # Path to the scenario to test
    scenario_path = '../tracks/{}'.format(track)
    scenario = SumoScenario(
        scenario_root=scenario_path,
        random_social_vehicle_count=nvehicle)

    env = gym.make('gym_hiway:hiway-competition-v0',
                config={
                    'sumo_scenario': scenario,
                    'headless': False,
                    'visdom': False,
                    'seed': seed,
                    'max_step_length': 10000,
                    'observation_space': OBSERVATION_SPACE,
                    'action_space': ACTION_SPACE,
                    'reward_function': reward,
                    'observation_function': observation,
                    'action_function': action,
                })
    accumulated_reward = 0
    for i in range(10):
        env_obs = env.reset()

        total_reward = 0.
        for _ in range(1000):
            pred_action = agent_policy.act(env_obs)
            env_obs, env_reward, done, _ = env.step(pred_action)
            total_reward += env_reward
            if done:
                # print("simulation ended")
                break
        accumulated_reward += total_reward

        print("Iteration {} on track {} with {} vehicles: {}".format(str(i), track, str(nvehicle), total_reward))

    evaluation_reward += accumulated_reward
    print("Total on track {} with {} vehicles: {}".format(track, str(nvehicle), accumulated_reward))
    print("##########")

    env.close()
    agent_policy.teardown()

print("Total Evaluation Reward: ",evaluation_reward)

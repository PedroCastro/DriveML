"""
__authors__ = "Sanjeevan Ahilan, Julian Villella, David Rusu"
__modified_by__= "Benedikt Kolbeinsson, Jiaze Sun, Pedro Castro"
__copyright__   = "Copyright 2019, Huawei R&D UK"

This module contains code to train an agent using the RLLib framework.
"""
import os
import random
import argparse
import shutil

import numpy as np
import ray
from ray import tune

from hiway.sumo_scenario import SumoScenario
from ray.tune import sample_from

from gym_hiway.env.competition_env import CompetitionEnv

from policy_ import MODEL_NAME, OBSERVATION_SPACE, ACTION_SPACE, observation, reward, action
from ray.tune.schedulers import PopulationBasedTraining


# Add custom metrics to your tensorboard using these callbacks
# see: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
def on_episode_start(info):
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["ego_speed"] = []


def on_episode_step(info):
    episode = info["episode"]
    single_agent_id = list(episode._agent_to_last_obs)[0]
    obs = episode.last_raw_obs_for(single_agent_id)
    episode.user_data["ego_speed"].append(obs['speed'])


def on_episode_end(info):
    episode = info["episode"]
    mean_ego_speed = np.mean(episode.user_data["ego_speed"])
    print("episode {} ended with length {} and mean ego speed {:.2f}".format(
        episode.episode_id, episode.length, mean_ego_speed))
    episode.custom_metrics["mean_ego_speed"] = mean_ego_speed

#for Population Based Training
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


def main(args):
    sumo_scenario = SumoScenario(
        scenario_root=os.path.abspath(args.scenario),
        random_social_vehicle_count=args.num_social_vehicles)

    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'tracks'))


    # tracks = ['1lane', '1lane_bwd', '1lane_sharp', '1lane_sharp_bwd',
    #           '1lane_new_a', '1lane_new_b', '1lane_new_c',
    #           '1lane_new_a_bwd', '1lane_new_b_bwd', '1lane_new_c_bwd',
    #           '2lane', '2lane_bwd', '2lane_sharp', '2lane_sharp_bwd',
    #           '2lane_new_a', '2lane_new_b', '2lane_new_c',
    #           '2lane_new_a_bwd', '2lane_new_b_bwd', '2lane_new_c_bwd',
    #           '3lane', '3lane_b', '3lane_bwd', '3lane_bwd_b', '3lane_sharp',
    #           '3lane_sharp_b', '3lane_sharp_bwd', '3lane_sharp_bwd_b',
    #           '3lane_new_a', '3lane_new_b', '3lane_new_c',
    #           '3lane_new_a_bwd', '3lane_new_b_bwd', '3lane_new_c_bwd']

    # tracks = ['2lane', '2lane_bwd', '2lane_sharp', '2lane_sharp_bwd',
    #           '2lane_new_a', '2lane_new_b', '2lane_new_c',
    #           '2lane_new_a_bwd', '2lane_new_b_bwd', '2lane_new_c_bwd']
    tracks = ['1lane', '2lane_sharp_bwd', '3lane_bwd', '3lane_sharp_bwd_b',
              '1lane_new_a', '2lane_new_b', '3lane_new_c',
              '1lane_new_c_bwd', '2lane_new_a_bwd', '3lane_new_a_bwd',
              '1lane_new_b', '2lane_new_c', '3lane_new_b_bwd', '3lane_sharp_b',
              '2lane_sharp_bwd']

    tracks_dir = [os.path.join(dataset_dir, track) for track in tracks]

    num_social_vehicles = [10, 10, 10, 50,
                           15, 15, 70,
                           20, 30, 100,
                           15, 50, 70, 25,
                           10]

    env_setting = list(zip(tracks_dir, num_social_vehicles))
    # note: ensure that len(env_setting) > args.num_workers

    #train each worker with different environmental setting
    class MultiEnv(CompetitionEnv):
        def __init__(self, env_config):
            env_config['sumo_scenario'] = SumoScenario(scenario_root=env_setting[env_config.worker_index-1][0],
                                                       random_social_vehicle_count=env_setting[env_config.worker_index-1][1])
            super(MultiEnv, self).__init__(config=env_config)


    tune_config = {
        'env': MultiEnv,
        'log_level': 'WARN',
        'num_workers': args.num_workers,
        'horizon': 1000,
        'env_config': {
            'seed': tune.randint(1000),
            'sumo_scenario': sumo_scenario,
            'headless': args.headless,
            'observation_space': OBSERVATION_SPACE,
            'action_space': ACTION_SPACE,
            'reward_function': tune.function(reward),
            'observation_function': tune.function(observation),
            'action_function': tune.function(action),
            'max_step_length': 1000,
        },
        'model':  {
            'custom_model': MODEL_NAME,
            'fcnet_activation': "relu"
        },
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end
        },
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        # "lr": 0.0,
        "lr_schedule": [[0, 1e-3],
                        [3000000, 3e-4],
                        [6000000, 1e-4],
                        [9000000, 3e-5],
                        [12000000, 1e-5]],
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 4096,
        "train_batch_size": 131072
    }

    experiment_name = 'rllib_multi_env'

    result_dir = args.result_dir
    checkpoint = None
    if args.checkpoint_num is not None:
        checkpoint = ('{dir}/checkpoint_{n}/checkpoint-{n}'
                      .format(dir=result_dir, n=args.checkpoint_num))

    log_dir = os.path.expanduser("~/ray_results")
    print(f"Checkpointing at {log_dir}")

    # for debugging e.g. with pycharm, turn on local_mode
    # ray.init(local_mode=True)

    # scheduler = pbt if args.pbt else None
    analysis = tune.run(
        'PPO',
        name=experiment_name,
        stop={'time_total_s': 60 * 60 * 40
              },
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir=log_dir,
        resume=args.resume_training,
        # restore=checkpoint,
        max_failures=1000,
        num_samples=args.num_samples,
        export_formats=['model', 'checkpoint'],
        config=tune_config,
        scheduler=scheduler,
    )

    print(analysis.dataframe().head())

    logdir = analysis.get_best_logdir('episode_reward_max')
    model_path = os.path.join(logdir, 'model')
    dest_model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "model")

    if not os.path.exists(dest_model_path):
        shutil.copytree(model_path, dest_model_path)
        print(f"wrote model to: {dest_model_path}")
    else:
        print(f"Model already exists at {dest_model_path} not overwriting")
        print(f"New model is stored at {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rllib-example')
    parser.add_argument('scenario',
                        help='Scenario to run (see scenarios/ for some samples you can use)',
                        type=str)
    parser.add_argument('--headless',
                        default=False,
                        help='run simulation in headless mode',
                        action='store_true')
    parser.add_argument('--num_samples',
                        type=int,
                        default=1,
                        help='Number of times to sample from hyperparameter space')
    parser.add_argument('--num_workers',
                        type=int,
                        default=15,
                        help='Number of workers')
    parser.add_argument('--num_social_vehicles',
                        type=int,
                        default=0,
                        help='Number of social vehicles in the environment')
    parser.add_argument('--resume_training',
                        default=False,
                        action='store_true',
                        help='Resume the last trained example')
    parser.add_argument('--result_dir',
                        type=str,
                        default='/home/ray_results',
                        help='Directory containing results')
    parser.add_argument('--checkpoint_num',
                        type=int,
                        default=None,
                        help='Checkpoint number')
    parser.add_argument('--pbt',
                        default=False,
                        help='use Population Based Training',
                        action='store_true')
    args = parser.parse_args()
    main(args)

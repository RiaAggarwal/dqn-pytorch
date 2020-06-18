import os
import pickle
from typing import List, Tuple, Dict

import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load_history(experiment_dir: str) -> List[Tuple[float, int]]:
    """
    Load the pickled history

    :param experiment_dir: The directory of the experiment results
    :return: The training history as a list with an entry per episode (reward, steps)
    """
    assert os.path.exists(experiment_dir), f"{experiment_dir} does not exist"

    file = os.path.join(experiment_dir, 'history.p')
    with open(file, 'rb') as f:
        history = pickle.load(f)
    return history


def get_experiments() -> List[Dict]:
    """
    Get all experiments in the experiments directory

    :return: List of experiments
    """
    experiments_root = os.path.join(root_dir, 'experiments')
    return [{'label': e, 'value': e} for e in os.listdir(experiments_root)]


def get_multi_index_history_df(experiments: List[str]) -> pd.DataFrame:
    """
    example:
                 baseline-1        snell-4        snell-5
              reward   step  reward   step  reward   step
        0      -20.0  430.0   -19.0  207.0   -16.0  590.0
        1      -18.0  322.0   -18.0  343.0   -19.0  361.0
        2      -17.0  423.0   -19.0  348.0   -19.0  514.0
        3      -18.0  414.0   -19.0  255.0   -18.0  538.0
        4      -20.0  364.0   -17.0  240.0   -20.0  407.0

    :param experiments:
    :return:
    """
    hist_dict = {}
    for e in experiments:
        history = load_history(os.path.join(root_dir, 'experiments', e))
        rewards = [v[0] for v in history]
        steps = [v[1] for v in history]
        hist_dict[e] = {'reward': rewards, 'step': steps}

    df = pd.DataFrame.from_dict({(i, j): hist_dict[i][j]
                                 for i in hist_dict.keys()
                                 for j in hist_dict[i].keys()},
                                orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.transpose()
    return df


def _get_history_df(experiments, selector: int):
    df = pd.DataFrame()
    for e in experiments:
        history = load_history(os.path.join(root_dir, 'experiments', e))
        rewards = [v[selector] for v in history]

        temp_df = pd.DataFrame(rewards, columns=[e])
        df = pd.concat([df, temp_df], axis=1)
    return df


def get_moving_average(df: pd.DataFrame, moving_avg_len) -> pd.DataFrame:
    if moving_avg_len <= 1:
        return df
    else:
        for column in df.columns:
            df[column] = df[column].rolling(window=moving_avg_len).mean()
    return df


def get_rewards_history_df(experiments: List[str], moving_avg_len=1) -> pd.DataFrame:
    """
    Get a dataframe of the reward after each episode for each experiment.

    :param moving_avg_len:
    :param experiments: List of experiments.
    :return: `pd.DataFrame`
    """
    df = _get_history_df(experiments, 0)
    return get_moving_average(df, moving_avg_len)


def get_steps_history_df(experiments: List[str], moving_avg_len=1) -> pd.DataFrame:
    """
    Get a dataframe of the number of steps in each episode for each experiment.

    :param moving_avg_len:
    :param experiments: List of experiments.
    :return: `pd.DataFrame`
    """
    df = _get_history_df(experiments, 1)
    return get_moving_average(df, moving_avg_len)

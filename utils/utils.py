import os
import pickle
from typing import List, Tuple

import pandas as pd

root_dir = os.path.dirname(os.path.dirname(__file__))


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


def get_experiments() -> List[str]:
    """
    Get all experiments in the experiments directory

    :return: List of experiments
    """
    experiments_root = os.path.join(root_dir, 'experiments')
    return os.listdir(experiments_root)


def get_history_dataframe(experiments: List[str]) -> pd.DataFrame:
    hist_dict = {}
    for e in experiments:
        history = load_history(os.path.join(root_dir, 'experiments', e))
        rewards = [v[0] for v in history]
        steps = [v[1] for v in history]
        hist_dict[e] = {'reward': rewards, 'step': steps}

    return pd.DataFrame(hist_dict)

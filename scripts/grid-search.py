#!/usr/bin/env python3
import argparse
from itertools import product
import math
import os
import shutil
import subprocess
from typing import List, Dict

import yaml

TEMP_DIR = os.path.join('.', 'configs', 'temp')
DEFAULT_BASE = 10
DEFAULT_STEPS = 4
COMMANDS_PER_JOB = 5


def parse_options(options_str: str) -> Dict:
    options_list = options_str.split(',')
    options_list = [s.strip() for s in options_list]
    options_dict = {s.split(' ')[0]: s.split(' ')[1:] for s in options_list}

    check_valid_option(options_dict)
    for k, v in options_dict.items():
        if len(v) < 2:
            raise parser.error(f"Option {k} must be given at least one value")
        options_dict[k] = try_convert_to_float(k, v)

    return options_dict


def try_convert_to_float(option, params, start=None, end=None):
    for idx, item in enumerate(params[start:end]):
        try:
            params[idx] = float(item)
        except ValueError:
            parser.error(f"The first two parameters of {option} must be numeric")
    return params


def check_valid_option(options_dict):
    for o in options_dict.keys():
        assert o in config.keys(), f"{o} is not a valid option"


def generate_config_files(grid, config):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    experiment_names = []
    os.mkdir(TEMP_DIR)
    for param_values in grid:
        experiment_params = dict(zip(options.keys(), param_values))
        experiment_config = config.copy()
        name = None
        for k, v in experiment_params.items():
            name = f'{k}={v}' if name is None else name + f'-{k}={v}'
            experiment_config[k] = v

        experiment_names.append(name)

        experiment_config_path = os.path.join(TEMP_DIR, name + '.yml')
        if os.path.exists(experiment_config_path):
            raise ValueError(f"duplicate experiment {name}")
        with open(experiment_config_path, 'w') as f:
            yaml.dump(experiment_config, f)

    return experiment_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Grid Search")
    parser.add_argument('-u', '--user', type=str, required=True, help="Must correspond to /data/<user>/")
    parser.add_argument('-m', '--memory', type=int, default=6, help="Memory request per experiment (default: 6)")
    parser.add_argument('-n', '--name', type=str, default='grid', help="Name to prepend to jobs")
    parser.add_argument('-o', '--options', type=str, default=None,
                        help='[option value [...][, option value [...]][, ...] '
                             'e.g. "--options width 40 80 160, height 10 20 30 40 50"')
    parser.add_argument('-f', '--file', default=os.path.join('configs', 'default.yml'), type=str,
                        help="Config file. Values outside the grid search will be read from here (default: default.yml)")
    parser.add_argument('-p', '--preview', action='store_true',
                        help="Do not run jobs. Do not delete temporary config files (view under configs/temp/)")
    args = parser.parse_args()

    with open(os.path.join('configs', 'default.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    options = parse_options(args.options)

    grid = product(*list(options.values()))
    experiment_names = generate_config_files(grid, config)

    if not args.preview:
        jobs = [experiment_names[i * COMMANDS_PER_JOB: (i + 1) * COMMANDS_PER_JOB]
                for i in range(math.ceil(len(experiment_names) / COMMANDS_PER_JOB))]
        procs = []
        for n, j in enumerate(jobs):
            names = [exp.replace('=', '.') for exp in j]
            files = [os.path.join(TEMP_DIR, exp + '.yml') for exp in j]
            cmd = 'python run-job.py -u {user} -n {names} -c {cpu} -m {memory} -f {files} --job-name {job_name}'.format(
                user=args.user, cpu=len(j) + 1, memory=len(j) * args.memory, names=' '.join(names),
                files=' '.join(files), job_name=f'{args.name}.{n}')
            print(cmd)
            subprocess.run(cmd.split(' '))
        shutil.rmtree(TEMP_DIR)

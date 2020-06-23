#!/usr/bin/env python3
import argparse
import copy
import os
import subprocess
from typing import List, Dict
import yaml


def load_experiments(files: List[str]) -> List[Dict]:
    experiments = []
    for file_path in files:
        with open(file_path, 'r') as f:
            experiments.append(yaml.load(f, Loader=yaml.FullLoader))
    return experiments


def get_options_strings(experiments: List[Dict], names: List[str]) -> List[str]:
    ex_options = []
    for ex, n in zip(experiments, names):
        options = ''
        for k, v in ex.items():
            opt = k.replace('_', '-')
            options += f' --{opt} '
            options += str(v)
        options += f' --store-dir ../experiments/{n}'
        ex_options.append(options)
    return ex_options


def load_job_template() -> Dict:
    with open('job.yml', 'r') as f:
        job = yaml.load(f, Loader=yaml.FullLoader)
    return job


def filter_git_commands(cmd_template: str) -> str:
    cmd_list = cmd_template.split(';')
    cmd_list = [s.strip() for s in cmd_list]
    cmd_list = [c for c in cmd_list if not c.startswith('git')]
    return '; '.join(cmd_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Job Runner')
    parser.add_argument('-u', '--user', type=str, required=True, help="Must correspond to /data/<user>/")
    parser.add_argument('-n', '--name', type=str, required=True, nargs='+',
                        help="Job name. Must contain only alphanumeric characters and '-'")
    parser.add_argument('-b', '--branch', default='master', type=str, help="Branch to run (default: master)")
    parser.add_argument('-f', '--file', default='default-config.yml', nargs='+', type=str,
                        help="Config file (default: default-config.yml)")
    parser.add_argument('-p', '--preview', action='store_true', help="Preview the created job file without running it")
    args = parser.parse_args()

    if not isinstance(args.file, list):
        args.file = [args.file]
    assert len(args.name) == len(args.file), "must provide an experiment name for each config file"

    experiments = load_experiments(args.file)
    experiments_options = get_options_strings(experiments, args.name)
    job = load_job_template()

    # Set job name
    job['metadata']['name'] = f'{args.user}-' + '-'.join(args.name)

    # configure the first container
    cmd_template = job['spec']['template']['spec']['containers'][0]['args'][0]
    cmd_template = cmd_template.replace('$', '')
    cmd_0 = cmd_template.format(user=args.user, branch=args.branch, options=experiments_options[0])

    job['spec']['template']['spec']['containers'][0]['args'][0] = cmd_0

    if len(args.name) > 1:
        container_spec = job['spec']['template']['spec']['containers'][0]
        cmd_template = filter_git_commands(cmd_template)
        cmd_template = '; '.join(['sleep 30', cmd_template])  # sleep the extra containers for 30s before starting
        for n, opts in enumerate(experiments_options[1:]):
            spec = copy.deepcopy(container_spec)
            cmd = cmd_template.format(user=args.user, branch=args.branch, options=opts)
            spec['args'][0] = cmd  # set commands
            spec['name'] = f"{container_spec['name']}-{n + 1}"  # set container name
            del spec['resources']['limits']['nvidia.com/gpu']  # remove gpu resources
            del spec['resources']['requests']['nvidia.com/gpu']  # remove gpu resources
            job['spec']['template']['spec']['containers'].append(spec)

    # write the yaml string
    job_yaml = yaml.dump(job, Dumper=yaml.Dumper)

    if args.preview:
        with open('final.yml', 'w') as f:
            f.write(job_yaml)
        print(job_yaml)
    else:
        with open('final.yml', 'w') as f:
            f.write(job_yaml)
        proc = subprocess.run(['kubectl', 'create', '-f', 'final.yml'])
        os.remove('final.yml')

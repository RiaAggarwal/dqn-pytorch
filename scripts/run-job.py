#!/usr/bin/env python3
import argparse
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
            if v is not None:  # boolean switches won't have arguments
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
    parser.add_argument('-c', '--cpu', type=int, default=1, help="CPU request (default: 1)")
    parser.add_argument('-g', '--gpu', type=int, default=1, help="GPU request (default: 1)")
    parser.add_argument('-m', '--memory', type=int, default=6, help="Memory request (default: 6)")
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
    cmd = cmd_template.format(user=args.user, branch=args.branch, options=experiments_options[0])

    if len(args.name) > 1:
        commands = cmd.split(';')
        del commands[-1]  # account for trailing semicolon
        commands = [s.strip() for s in commands]
        for opts in experiments_options[1:]:
            commands.append(f'python main.py {opts}')
        cmd = '; '.join(commands[:-len(args.name)]) + '; ' + ' & '.join(commands[-len(args.name):])
        # Add a process to keep the pod alive while the background jobs are running
        cmd += ' & while [ -n "$(jobs | grep Running)" ]; do sleep 10; done;'
    else:
        cmd = cmd_template.format(user=args.user, branch=args.branch, options=experiments_options[0])

    job['spec']['template']['spec']['containers'][0]['args'][0] = cmd
    job['spec']['template']['spec']['containers'][0]['resources']['limits']['memory'] = f'{round(args.memory * 1.2):d}Gi'
    job['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = args.gpu
    job['spec']['template']['spec']['containers'][0]['resources']['limits']['cpu'] = args.cpu
    job['spec']['template']['spec']['containers'][0]['resources']['requests']['memory'] = f'{args.memory}Gi'
    job['spec']['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu'] = args.gpu
    job['spec']['template']['spec']['containers'][0]['resources']['requests']['cpu'] = args.cpu

    # write the yaml string
    job_yaml = yaml.dump(job, Dumper=yaml.Dumper)

    if args.preview:
        print(job_yaml)
    else:
        with open('final.yml', 'w') as f:
            f.write(job_yaml)
        proc = subprocess.run(['kubectl', 'create', '-f', 'final.yml'])
        os.remove('final.yml')

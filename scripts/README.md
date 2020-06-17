# Utility scripts

## run-job.sh

```shell script
~/pong-underwater-rl/scripts$ ./run-job.sh -h
 [OPTION]...

Run kubernetes job

 Options:
  -u, --user        Must correspond to /data/<user>/
  -n, --name        Job name to append
  -b, --branch      Branch to run (default: master)
  -f, --file        Config file (default: experiment-config.yml)
  -p, --preview     Preview the created job file without running it
  -q, --quiet       Quiet (no output)
  -l, --log         Print log to file
  -v, --verbose     Output more information. (Items echoed to 'verbose')
  -d, --debug       Runs script in BASH debug mode (set -x)
  -h, --help        Display this help and exit
      --version     Output version information and exit
```

### Usage

First make the script executable:
```shell script
chmod +x run-job.sh
```

Create a configuration file modeled on `default-config.yml`:
```yaml
width: 160
height: 160
ball: 3.0
snell: 3.0
paddle_speed: 3.0
paddle_length: 45
learning_rate: 1e-4
update_prob: 0.2
episodes: 4000
resume: False
#checkpoint: dqn_pong_model
#history: history.p
pretrain: False
```

Preview the command using the `--preview` switch, e.g.:
```shell script
~/pong-underwater-rl/scripts$ ./run-job.sh --user username --name experiment-name --file my-experiment.yml --preview

apiVersion: batch/v1
kind: Job
metadata:
  name: username-experiment-name # specify your job name here
spec:
  template:
    spec:
      containers:
      - name: gpu-container
        image: gitlab-registry.nautilus.optiputer.net/jiacheng/docker-images:gym
        command: ["/bin/sh","-c"]
        args:
        - cd /data/username/pong-underwater-rl;
          git fetch origin master;
          git checkout master;
          git pull origin master;
          pip install -e gym-dynamic-pong;
          cd underwater_rl;
          python main.py  --width 160 --height 160 --ball 3.0 --snell 3.0 --paddle-speed 3.0 --paddle-length 45 --learning-rate 1e-4 --update-prob 0.2 --episodes 4000 --store-dir ../experiments/experiment-name;
        volumeMounts:
        - mountPath: /data
          name: data
        - mountPath: /dev/shm
          name: dshm
        resources:
          limits:
            memory: 8Gi
            nvidia.com/gpu: "1"
            cpu: "4"
          requests:
            memory: 8Gi
            nvidia.com/gpu: "1"
            cpu: "2"
      restartPolicy: Never
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: storage
        - name: dshm
          emptyDir:
            medium: Memory
  backoffLimit: 1
```

Once you are satisfied with the generated configuration, you can launch the job by removing the `--preview` switch:
```shell script
./run-job.sh --user username --name experiment-name --file my-experiment.yml
```

### Docs

Required options:
- `--user` This must correspond exactly to the name of the directory `/data/<user>` in the kubernetes cluster. 
The directory must already contain a clone of the git repository.
- `--name` The name of the experiment to run. In the cluster, results will be saved under 
`/data/<user>/pong-underwater-rl/experiments/<name>`

Other options:
- `--branch` The git branch to use. The code is automatically updated with the latest from `master` or the specified branch.
- `--file` The configuration file containing options to pass to `main.py`. The script must be updated with any new  options.
- `--preview` The job is run by automatically generating a yaml file from the template `job.yml`. With this switch the final yaml used to generate the job will be printed to console without a job being run.
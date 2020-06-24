# Utility scripts

## run-job.sh

```shell script
~/pong-underwater-rl/scripts$ ./run-job.py -h
usage: run-job.py [-h] -u USER -n NAME [NAME ...] [-c CPU] [-g GPU]
                  [-m MEMORY] [-b BRANCH] [-f FILE [FILE ...]] [-p]

Job Runner

optional arguments:
  -h, --help            show this help message and exit
  -u USER, --user USER  Must correspond to /data/<user>/
  -n NAME [NAME ...], --name NAME [NAME ...]
                        Job name. Must contain only alphanumeric characters
                        and '-'
  -c CPU, --cpu CPU     CPU request (default: 1)
  -g GPU, --gpu GPU     GPU request (default: 1)
  -m MEMORY, --memory MEMORY
                        Memory request (default: 6)
  -b BRANCH, --branch BRANCH
                        Branch to run (default: master)
  -f FILE [FILE ...], --file FILE [FILE ...]
                        Config file (default: default-config.yml)
  -p, --preview         Preview the created job file without running it

```

### Usage

First make the script executable:
```shell script
chmod +x run-job.py
```

Create a configuration file modeled on `default-config.yml`:
```yaml
width: 160
height: 160
ball: 3.0
snell: 3.0
paddle_speed: 3.0
paddle_length: 45
paddle_angle: 45
learning_rate: 1e-4
update_prob: 0.2
episodes: 4000
#resume:
#checkpoint: dqn_pong_model
#history: history.p
network: dqn_pong_model
#pretrain:
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
            cpu: "1"
          requests:
            memory: 6Gi
            nvidia.com/gpu: "1"
            cpu: "1"
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

To run multiple experiments, pass multiple arguments to `file` and `name`
```shell script
./run-job.sh --user username --name ex1 ex2 ex3 --file ex1.yml ex2.yml ex3.yml
```

### Docs

Required options:
- `--user` This must correspond exactly to the name of the directory `/data/<user>` in the kubernetes cluster. 
The directory must already contain a clone of the git repository.
- `--name` The name of the experiment(s) to run. In the cluster, results will be saved under 
`/data/<user>/pong-underwater-rl/experiments/<name>`

Other options:
- `--branch` The git branch to use. The code is automatically updated with the latest from `master` or the specified branch.
- `--file` The configuration file(s) containing options to pass to `main.py`. The script must be updated with any new  options.
- `--preview` The job is run by automatically generating a yaml file from the template `job.yml`. With this switch the final yaml used to generate the job will be printed to console without a job being run.
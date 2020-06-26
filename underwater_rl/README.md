# DQN PyTorch
This is a simple implementation of the Deep Q-learning algorithm on the Atari Pong environment.

![](/underwater_rl/assetsater_rl/assets/pong.gif)

## References
1. *Playing Atari with Deep Reinforcement Learning*, Mnih et al., 2013

## Usage

```shell script
python main.py
```

### Arguments

`--snell-visible`
- Determine visibility of the snell layer.
    - 'human' - visible when rendering only
    - 'machine' - visible to the agent and when rendering
    - 'none' - invisible

`--state`
- set the state representation
    - 'binary' - binary image (as before)
    - 'color' - color image. If this is not set, and the snell layer is set to 'machine', the agent will not be able to see the ball in the snell layer.
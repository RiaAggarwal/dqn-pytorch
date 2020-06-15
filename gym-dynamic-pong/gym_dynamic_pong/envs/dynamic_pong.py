import math
import random
import os
import time
from typing import Tuple, Dict, Any, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from shapes import Line, Rectangle, Point

EPSILON = 1e-7


def bool_array_to_rgb(array: np.ndarray) -> np.ndarray:
    assert array.dtype == np.bool, "array must be boolean"
    array = array.astype(np.uint8) * (2 ** 8 - 1)
    empty = np.zeros_like(array)
    return np.stack([array, empty, empty], axis=2)


class Paddle(Rectangle):
    def __init__(self, height, width, speed, side, max_width, max_height):
        super().__init__(height=height, width=width, max_height=max_height, max_width=max_width)
        self.side = side
        self.speed = speed

        self.set_paddle_left_right()

    def set_paddle_left_right(self):
        if self.side == 'left':
            self.x_pos = self.width / 2
        elif self.side == 'right':
            self.x_pos = self.max_width - self.width / 2
        else:
            raise ValueError("`which` must be 'left' or 'right'")

    def up(self):
        self.y_pos += self.speed

    def down(self):
        self.y_pos -= self.speed

    def get_edges(self) -> Dict[str, Line]:
        """
        Only return the field-side edge
        """
        if self.side == 'right':
            return {'left': Line((self.left_bound, self.bot_bound), (self.left_bound, self.top_bound))}
        elif self.side == 'left':
            return {'right': Line((self.right_bound, self.bot_bound), (self.right_bound, self.top_bound))}

    @property
    def y_pos(self):
        return self._y_pos

    @y_pos.setter
    def y_pos(self, value):
        if value - self.height / 2 < 0:
            self._y_pos = self.height / 2
        elif value + self.height / 2 > self.max_height:
            self._y_pos = self.max_height - self.height / 2
        else:
            self._y_pos = value

    def get_fraction_of_paddle(self, point: Point):
        """
        Computes the fractional distance from the middle of the paddle, normalized by the paddle's height.
        Asserts if the ball was not on the paddle.

        :param point: the point where the ball hit the paddle
        :return: fraction of the paddle
        """
        fraction = (point.y - self.y_pos) / self.height
        assert -0.5 <= fraction <= 0.5, "The ball was not on the paddle"
        return fraction


class Ball(Rectangle):
    def __init__(self, max_height, max_width):
        super().__init__(max_height=max_height, max_width=max_width, width=2, height=2)
        self._angle = math.pi - (random.random() - 0.5) * (math.pi / 3)

    def reset(self):
        self._angle = (random.random() - 0.5) * (math.pi / 3)
        self.x_pos = self.max_width / 2
        self.y_pos = self.max_height / 2

    @property
    def angle(self):
        """
        Angle with respect to the right horizontal
        """
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value % (2 * math.pi)

    @property
    def unit_velocity(self) -> Point:
        x = math.cos(self.angle)
        y = math.sin(self.angle)
        return Point(x, y)

    @unit_velocity.setter
    def unit_velocity(self, value: Union[Tuple[float, float], Point]):
        """
        Sets the angle parameter give a set of (x, y) coordinates
        :param value: (x, y)
        """
        if isinstance(value, tuple):
            value = Point(*value)
        assert isinstance(value, Point), f"value must be a point, not {type(value)}"
        self.angle = value.angle


class Snell(Rectangle):
    def __init__(self, width_fraction, max_height, max_width, speed):
        """
        area indicating ball speed.
        :return:
        """
        super().__init__(width=int(width_fraction * max_width), height=max_height, max_width=max_width,
                         max_height=max_height)
        self.y_pos = self.max_height / 2
        self.x_pos = self.max_width / 2
        self.speed = speed


class Canvas:
    action_meanings = {0: 'NOOP',
                       1: 'UP',
                       2: 'DOWN', }
    actions = {k: v for v, k in action_meanings.items()}

    def __init__(self,
                 paddle_l: Paddle,
                 paddle_r: Paddle,
                 ball: Ball,
                 snell: Snell,
                 ball_speed: int,
                 height: int,
                 width: int,
                 their_update_probability: float):

        assert isinstance(their_update_probability, (float, int)),\
            f"their_update_probability must be numeric, not {type(their_update_probability)}"
        assert 0 <= their_update_probability <= 1, f"{their_update_probability} outside allowed bounds [0, 1]"

        self.their_update_probability = their_update_probability

        self.width = width
        self.height = height

        # Initialize objects
        self.snell = snell
        self.default_ball_speed = ball_speed
        self.ball = ball
        self.paddle_l = paddle_l
        self.paddle_r = paddle_r

        self.we_scored = False
        self.they_scored = False

        # score
        self.our_score = 0
        self.their_score = 0

    def get_edges(self) -> Dict[Any, Dict[str, Line]]:
        """
        Gets edges of canvas and all objects as lines

        :return: ```
        {'border': {
                       'left': Line(),
                       'top': Line(),
                       'right': Line(),
                       'bottom': Line(),
                    },
         obj1: {
                   'left': Line(),
                   'top': Line(),
                   'right': Line(),
                   'bottom': Line(),
                },
        ...}
        ```
        """
        border = {
            'left'  : Line((0, 0), (0, self.height)),
            'top'   : Line((0, self.height), (self.width, self.height)),
            'right' : Line((self.width, self.height), (self.width, 0)),
            'bottom': Line((self.width, 0), (0, 0))
        }
        edges = {o: o.get_edges() for o in self.get_objects()}
        edges['border'] = border
        return edges

    def get_objects(self):
        return self.snell, self.paddle_r, self.paddle_l

    def to_numpy(self):
        out = np.zeros((self.height, self.width), dtype=np.bool)

        for sprite in (self.ball, self.paddle_l, self.paddle_r):
            out |= sprite.to_numpy()
        return out

    def score(self, who):
        """
        Increment the score and reset the ball

        :param who: 'we' or 'they'
        :return: reward
        """
        if who == 'they':
            reward = -1
            self.their_score += 1
        elif who == 'we':
            reward = 1
            self.our_score += 1
        else:
            raise ValueError(f"who must be 'we' or 'they', not {who}")

        self._reset_ball()
        return reward

    def step(self, action):
        self._move_our_paddle(action)
        self._step_their_paddle()
        return self._step_ball()

    def get_state_size(self) -> Tuple[int, int]:
        """
        Return the tuple (height, width) of the canvas dimensions
        """
        return self.height, self.width

    def _reset_ball(self):
        self.ball.reset()

    def _move_our_paddle(self, action) -> None:
        """
        Move our paddle according to the provided action
        :param action: the action code
        """
        if not isinstance(action, int):
            action = action.item()  # pops the item if the action is a single tensor
        assert action in [a for a in self.action_meanings.keys()], f"{action} is not a valid action"
        if action == self.actions['UP']:
            self.paddle_r.up()
        elif action == self.actions['DOWN']:
            self.paddle_r.down()

    def _step_ball(self, speed: Union[float, int] = None):
        """
        Move the ball to the next position according to the speed of the layer it is in.
        :param speed: used to continue the trajectory of a ball that interacted with an object
        """
        if speed is None:
            speed = self._get_ball_speed()
        new_pos = tuple(speed * v + x for x, v in zip(self.ball.pos, self.ball.unit_velocity))
        # noinspection PyTypeChecker
        trajectory = Line(self.ball.pos, new_pos)

        result = self._get_first_intersection(trajectory)
        reward = 0
        if result is None:  # No intersection
            self.ball.pos = new_pos
        else:
            reward = self._interaction_dispatcher(*result, trajectory)

        return reward

    def _interaction_dispatcher(self, obj: Union[str, Paddle, Snell], edge: str, point: Point, line: Line,
                                trajectory: Line):
        """
        Dispatch data to the appropriate method based on the interaction `obj`

        :param line: the line that the trajectory intersected
        :param trajectory: the trajectory of the ball
        :param obj: 'border' or an object in the canvas
        :param edge: 'top', 'bottom', 'left', 'right'
        :param point: the point of interaction
        """
        assert edge in ['top', 'bottom', 'left', 'right']
        reward = 0
        if obj == 'border':
            reward = self._interact_border(edge, point, trajectory)
        elif obj is self.paddle_l or obj is self.paddle_r:
            self._interact_paddle(obj, point, trajectory)
        elif obj is self.snell:
            self._refract(point, trajectory, line)

        return reward

    def _interact_paddle(self, paddle: Paddle, point: Point, trajectory: Line):
        paddle_fraction = paddle.get_fraction_of_paddle(point)
        angle = paddle_fraction * math.pi / 4
        angle = math.pi - angle if self.ball.unit_velocity.x > 0 else angle

        self.ball.angle = angle
        return self._finish_step_ball(point, trajectory)

    def _refract(self, point: Point, trajectory: Line, boundary: Line):
        if self.snell.is_in(trajectory.start):
            s0 = self.snell.speed
            s1 = self.default_ball_speed
        else:
            s0 = self.default_ball_speed
            s1 = self.snell.speed

        angle = abs(boundary.angle_to_normal(trajectory))
        if s1 > s0:  # if the second speed is faster, there is a critical angle
            critical_angle = math.asin(s0 / s1)
            if angle >= critical_angle:
                # TODO: reflect to arbitrary angle
                self._reflect(Point(-1, 1), point, trajectory)
                return
        new_angle = math.asin(s1 / s0 * math.sin(angle))
        self.ball.angle -= angle - new_angle
        return self._finish_step_ball(point, trajectory)

    def _interact_border(self, edge: str, point: Point, trajectory: Line):
        reward = 0
        if edge in ['top', 'bottom']:
            self._reflect(Point(1, -1), point, trajectory)
        elif edge == 'left':
            reward = self.score('we')
        elif edge == 'right':
            reward = self.score('they')
        else:
            raise ValueError(f'invalid edge, {edge}')

        return reward

    def _reflect(self, direction: Point, point: Point, trajectory: Line):
        """
        Multiplies the velocity of the ball by `direction`, continues the path of the ball by calculating the remaining
        speed using trajectory and point
        :param direction: velocity multiplier
        :param point: The point of interaction
        :param trajectory: The original trajectory of the ball
        """
        self.ball.unit_velocity *= direction
        return self._finish_step_ball(point, trajectory)

    def _finish_step_ball(self, point, trajectory):
        self.ball.pos = point + self.ball.unit_velocity * EPSILON
        remaining_speed = point.l2_distance(trajectory.end)
        return self._step_ball(remaining_speed)

    def _get_first_intersection(self, trajectory: Line) -> Union[Tuple[Any, str, Point], None]:
        """
        Find the first point at which the trajectory interacted with an object
        :param trajectory: the trajectory of the object
        :return: (object interacted with, object edge name, point of interaction)
        """
        first_intersection = None
        result = None
        for o, d in self.get_edges().items():
            for edge, line in d.items():
                i = trajectory.get_intersection(line)
                if i is not None:
                    if first_intersection is None:
                        result = o, edge, i, line
                    elif line.point1_before_point2(i, first_intersection):
                        result = o, edge, i, line
        return result

    def _get_ball_speed(self) -> float:
        if self.ball.is_overlapping(self.snell):
            return self.snell.speed
        else:
            return self.default_ball_speed

    def _step_their_paddle(self):
        """
        Move the opponents paddle. Override this in a subclass to change the behavior.
        """
        if random.random() < self.their_update_probability:
            if self.paddle_l.y_pos < self.ball.y_pos:
                self.paddle_l.up()
            else:
                self.paddle_l.down()


class DynamicPongEnv(gym.Env):
    metadata = {'render.modes': ['human', 'png']}

    def __init__(self,
                 max_score=20,
                 width=400,
                 height=300,
                 default_speed=3,
                 snell_speed=3,
                 our_paddle_speed=3,
                 their_paddle_speed=3,
                 our_paddle_height=45,
                 their_paddle_height=45,
                 their_update_probability=0.2,):

        for v in width, height:
            assert isinstance(v, int), "width and height must be integers"

        # configuration
        self.max_score = max_score
        self.width = width
        self.height = height
        self.default_speed = default_speed
        self.snell_speed = snell_speed
        self.our_paddle_speed = our_paddle_speed
        self.their_paddle_speed = their_paddle_speed
        self.our_paddle_height = our_paddle_height
        self.their_paddle_height = their_paddle_height
        self.their_update_probability = their_update_probability

        # initialization
        self._initialize_env()
        self.frame = None
        self.fig = None
        self.ax = None
        self.fig_handle = None
        self.frame_count = 0

        self.observation_space = spaces.Box(low=False, high=True, dtype=np.bool,
                                            shape=(self.env.get_state_size()))
        self.action_space = spaces.Discrete(3)  # initialize discrete action space with 3 actions
        self.ale = ALEInterfaceMock(self.env, self.max_score)

    def step(self, action) -> Tuple[np.ndarray, int, bool, dict]:
        """
        Move the environment to the next state according to the provided action.
        :param action:
        :return: (data, reward, episode_over, info)
        """
        reward = self.env.step(action)
        self.frame = self.env.to_numpy()
        return bool_array_to_rgb(self.frame), reward, self.episode_is_over(), {}  # {} is a generic info dictionary

    def episode_is_over(self):
        """
        :returns: True if the episode is over
        """
        if self.env.their_score == self.max_score or self.env.our_score == self.max_score:
            self.reset()
            return True
        else:
            return False

    def reset(self):
        self._initialize_env()

    def render(self, mode='human', save_dir=None):
        """
        Renders the most recent frame according to the specified mode.
        - human: render to screen using `matplotlib`
        - png: save png images to `save_dir`

        :param mode: 'human' or 'png'
        :param save_dir: directory to save images to in modes other than 'human'
        """
        if mode == 'human':
            self._display_screen()
        elif mode == 'png':
            self._save_display_images(save_dir)

    def close(self):
        self.env = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def get_action_meanings(self):
        return self.env.action_meanings

    # Display
    def _display_screen(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.fig_handle = self.ax.imshow(self.frame, cmap='gray')
            self.fig.show()
        else:
            self.fig_handle.set_data(self.frame)
        self.ax.set_title(f"{self.env.their_score}                    {self.env.our_score}")
        self.fig.canvas.draw()

    def _save_display_images(self, save_dir):
        """
        Saves the most recent frame as a png image in the directory `save_dir`. If `save_dir` does not exist, it is
        created. Another directory under `save_dir` with the timestamp of the first call of this function is created.
        Numbered frames are stored under this timestamped directory.

        :param save_dir: Directory to save the images in. It will be created if it does not exist.
        """
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H:%M', t)
        save_dir = os.path.join(save_dir, timestamp)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.fig_handle = self.ax.imshow(self.frame, cmap='gray')
        else:
            self.fig_handle.set_data(self.frame)
        self.ax.set_title(f"{self.env.their_score}                    {self.env.our_score}")
        self.fig.canvas.draw()

        path = os.path.join(save_dir, f'{self.frame_count:07d}.png')
        self.frame_count += 1
        self.fig.savefig(path)

    # Sprites
    def _initialize_env(self):
        """
        Initialize the Canvas object containing all the important interactions in the environment.
        """
        self.env = Canvas(
            self._init_paddle('left', self.their_paddle_height, self.their_paddle_speed),
            self._init_paddle('right', self.our_paddle_height, self.our_paddle_speed),
            self._init_ball(),
            Snell(0.25, self.height, self.width, self.snell_speed),
            self.default_speed,
            self.height,
            self.width,
            self.their_update_probability,
        )

    def _init_paddle(self, which_side: str, height, speed) -> Paddle:
        """
        Create a paddle object

        :param which_side: 'left' or 'right'
        :param speed: the number of units the paddle can move in a single frame
        :param height: the height of the paddle
        """
        paddle = Paddle(height, int(0.02 * self.width) + 1, speed, which_side, self.width, self.height)
        paddle.y_pos = self.height / 2
        return paddle

    def _init_ball(self) -> Ball:
        """
        Create a ball object
        """
        ball = Ball(self.height, self.width)
        ball.x_pos = self.width // 2
        ball.y_pos = self.height // 2
        return ball

    def _init_score(self):
        raise NotImplementedError()


class ALEInterfaceMock:
    """
    Object to expose the lives method. There are likely other methods that need to be exposed.
    """

    def __init__(self, env: Canvas, lives: int):
        assert lives > 0, "Number of lives must be > 0"
        self.env = env
        self._lives = lives

    def lives(self) -> int:
        """
        :returns: the minimum of the goals until we and the the goals until the opponent wins
        """
        # TODO: this env object is a copy, does not update when the original does.
        return min(self._lives - self.env.our_score, self._lives - self.env.their_score)

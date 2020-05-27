import unittest
import math

from gym_dynamic_pong.envs import DynamicPongEnv
from gym_dynamic_pong.envs.dynamic_pong import Ball
from shapes import Line, Point


@unittest.skip
class TestVideoOutputManually(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DynamicPongEnv(default_speed=10, snell_speed=8)

    def test_step_executes_up_manually(self):
        for i in range(50):
            self.env.step(1)
            self.env.render()

    def test_step_executes_down_manually(self):
        for i in range(50):
            self.env.step(1)
            self.env.render()

    def test_big_right_paddle(self):
        self.env.env.ball.angle = 0
        self.env.env.paddle_r.height = 250
        for i in range(200):
            self.env.step(0)
            self.env.render()

    def test_bounce_ball_off_the_top(self):
        self.env.env.ball.angle = math.pi / 4
        self.env.env.paddle_r.height = 250
        for i in range(600):
            self.env.step(0)
            self.env.render()

    def tearDown(self) -> None:
        self.env.close()


class TestEnvironmentBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 400
        self.height = 300
        self.default_speed = 10
        self.snell_speed = 10
        self.paddle_speed = 3
        self.paddle_height = 45
        pong_env = DynamicPongEnv(max_score=2, width=self.width,
                                  height=self.height,
                                  default_speed=self.default_speed,
                                  snell_speed=self.snell_speed,
                                  our_paddle_speed=self.paddle_speed,
                                  their_paddle_speed=self.paddle_speed,
                                  our_paddle_height=self.paddle_height,
                                  their_paddle_height=self.paddle_height, )
        self.env = pong_env
        self.env.step(0)

    def test_their_score_starts_at_zero(self):
        self.assertEqual(0, self.env.env.their_score)

    def test_our_score_starts_at_zero(self):
        self.assertEqual(0, self.env.env.our_score)

    def test_their_score_increases(self):
        self.env.env.ball.x_pos = self.width - self.default_speed + 1
        self.env.env.ball.y_pos = self.height - 2 * self.default_speed
        self.env.env.ball.angle = 0
        self.env.step(0)
        self.assertEqual(1, self.env.env.their_score)

    def test_our_score_increases(self):
        self.env.env.ball.x_pos = self.default_speed - 1
        self.env.env.ball.y_pos = self.height - 2 * self.default_speed
        self.env.env.ball.angle = math.pi
        self.env.step(0)
        self.assertEqual(1, self.env.env.our_score)

    def test_ball_does_not_get_stuck_in_1000_steps_with_defaults(self):
        self.plausible_ball_motion_tester()

    def test_ball_does_not_get_stuck_in_1000_steps_with_max_paddles(self):
        self.env.env.paddle_r.height = self.height
        self.env.env.paddle_l.height = self.height
        self.plausible_ball_motion_tester()

    def plausible_ball_motion_tester(self):
        self.env.step(0)
        x_prev = self.env.env.ball.x_pos
        y_prev = self.env.env.ball.y_pos
        for i in range(1000):
            self.env.step(0)
            x_pos = self.env.env.ball.x_pos
            y_pos = self.env.env.ball.y_pos

            # ball is too far away from any edge to interact with one and the ball is not at the default start
            if self.default_speed * 4 < x_pos < self.width - self.default_speed * 4 and \
                    self.default_speed * 4 < y_pos < self.height - self.default_speed * 4 and \
                    x_pos != 200:
                # ball is moving right and up
                if 0 < self.env.env.ball.angle < math.pi / 2:
                    self.assertGreater(x_pos, x_prev)
                    self.assertGreater(y_pos, y_prev)
                # ball is moving left and up
                elif math.pi / 2 < self.env.env.ball.angle < math.pi:
                    self.assertLess(x_pos, x_prev)
                    self.assertGreater(y_pos, y_prev)
                # ball is moving left and down
                elif math.pi < self.env.env.ball.angle < math.pi * 3 / 2:
                    self.assertLess(x_pos, x_prev)
                    self.assertLess(y_pos, y_prev)
                # ball is moving right and down
                elif math.pi * 3 / 2 < self.env.env.ball.angle < math.pi * 2:
                    self.assertGreater(x_pos, x_prev)
                    self.assertLess(y_pos, y_prev)

            x_prev = x_pos
            y_prev = y_pos

    def test_ball_hit_exact_upper_edge_and_bounces_correctly(self):
        angle = math.pi / 4
        speed = self.get_ball_speed()
        y_pos = self.height - speed * math.sin(angle)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 2)
        self.assertGreater(self.env.env.ball.x_pos, x_pos)

    def get_ball_speed(self):
        return self.snell_speed if self.env.env.snell.is_in(self.env.env.ball.pos) else self.default_speed

    def test_ball_bounced_off_top_moving_left_over_one_step(self):
        angle = math.pi * 3 / 4
        speed = self.get_ball_speed()
        y_pos = self.height - speed * math.sin(angle) / 2
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 2)

    def test_ball_bounced_off_bottom_moving_left_over_one_step(self):
        angle = math.pi * 5 / 4
        speed = self.get_ball_speed()
        y_pos = abs(speed * math.sin(angle) / 2)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 5)
        self.assertLess(self.env.env.ball.x_pos, x_pos)

    def test_ball_bounced_off_bottom_moving_right_over_one_step(self):
        angle = math.pi * 7 / 4
        speed = self.get_ball_speed()
        y_pos = abs(speed * math.sin(angle) / 2)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 5)
        self.assertGreater(self.env.env.ball.x_pos, x_pos)

    def test_ball_bounced_off_right_paddle(self):
        angle = 0
        y_pos = self.env.env.paddle_r.y_pos
        x_pos = self.width - self.default_speed / 2 - self.env.env.paddle_r.left_bound

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.x_pos = x_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 0)
        self.assertAlmostEqual(x_pos, self.env.env.ball.x_pos, 1)

    def test_ball_bounced_off_left_paddle(self):
        angle = math.pi
        y_pos = self.height / 2
        x_pos = self.env.env.paddle_l.right_bound + self.env.default_speed / 2

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.x_pos = x_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 0)
        self.assertAlmostEqual(x_pos, self.env.env.ball.x_pos, 1)

    def tearDown(self) -> None:
        self.env.close()


class TestEnvironmentBehaviorWithRefraction(TestEnvironmentBehavior):
    def setUp(self) -> None:
        self.width = 400
        self.height = 300
        self.default_speed = 10
        self.snell_speed = 8
        self.paddle_speed = 3
        self.paddle_height = 45
        pong_env = DynamicPongEnv(max_score=2, width=self.width,
                                  height=self.height,
                                  default_speed=self.default_speed,
                                  snell_speed=self.snell_speed,
                                  our_paddle_speed=self.paddle_speed,
                                  their_paddle_speed=self.paddle_speed,
                                  our_paddle_height=self.paddle_height,
                                  their_paddle_height=self.paddle_height, )
        self.env = pong_env
        self.env.step(0)


class TestBall(unittest.TestCase):
    def setUp(self) -> None:
        self.ball = Ball(100, 100)

    def test_unit_velocity_sets_angle_to_0_for_1_0(self):
        self.ball.unit_velocity = Point(1., 0.)
        self.assertAlmostEqual(0, self.ball.angle)

    def test_unit_velocity_sets_angle_to_30_for_sqrt3_1(self):
        self.ball.unit_velocity = Point(math.sqrt(3), 1)
        self.assertAlmostEqual(math.pi / 6, self.ball.angle)

    def test_unit_velocity_sets_angle_to_45_for_1_1(self):
        self.ball.unit_velocity = Point(1., 1.)
        self.assertAlmostEqual(math.pi / 4, self.ball.angle)

    def test_unit_velocity_sets_angle_to_135_for_neg1_1(self):
        self.ball.unit_velocity = Point(-1., 1.)
        self.assertAlmostEqual(math.pi * 3 / 4, self.ball.angle)

    def test_unit_velocity_sets_angle_to_150_for_neg_sqrt3_1(self):
        self.ball.unit_velocity = Point(-math.sqrt(3), 1)
        self.assertAlmostEqual(math.pi * 5 / 6, self.ball.angle)

    def test_unit_velocity_sets_angle_to_180_for_neg1_0(self):
        self.ball.unit_velocity = Point(-1., 0.)
        self.assertAlmostEqual(math.pi, self.ball.angle)

    def test_unit_velocity_sets_angle_to_225_for_neg1_neg1(self):
        self.ball.unit_velocity = Point(-1., -1.)
        self.assertAlmostEqual(math.pi * 5 / 4, self.ball.angle)

    def test_unit_velocity_sets_angle_to_315_for_1_neg1(self):
        self.ball.unit_velocity = Point(1., -1.)
        self.assertAlmostEqual(math.pi * 7 / 4, self.ball.angle)


class TestLine(unittest.TestCase):
    def test_angle_between_up_line_and_ray_from_bottom_left_is_positive(self):
        l1 = Line((0, -1), (0, 1))
        l2 = Line((-1, -1), (1, 1))
        angle = l1.angle_to_normal(l2)
        self.assertGreater(angle, 0)

    def test_angle_between_up_line_and_ray_from_top_left_is_negative(self):
        l1 = Line((0, -1), (0, 1))
        l2 = Line((-1, 1), (1, -1))
        angle = l1.angle_to_normal(l2)
        self.assertLess(angle, 0)

    def test_angle_between_down_line_and_ray_from_bottom_left_is_negative(self):
        l1 = Line((0, 1), (0, -1))
        l2 = Line((-1, -1), (1, 1))
        angle = l1.angle_to_normal(l2)
        self.assertLess(angle, 0)

    def test_angle_between_down_line_and_ray_from_top_left_is_positive(self):
        l1 = Line((0, 1), (0, -1))
        l2 = Line((-1, 1), (1, -1))
        angle = l1.angle_to_normal(l2)
        self.assertGreater(angle, 0)

    def test_10_10_20_20_and_10_20_20_10_intersect_at_15_15(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((10, 20), (20, 10))
        intersection = line1.get_intersection(line2)
        self.assertEqual(Point(15., 15.), intersection)

    def test_same_lines_do_not_intersect(self):
        line1 = Line((10, 10), (20, 20))
        line2 = line1
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_non_parallel_non_intersecting_segments_do_not_intersect_1(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((10, 20), (14, 16))
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_non_parallel_non_intersecting_segments_do_not_intersect_2(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((1, 5), (20, 5))
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_close_non_parallel_paddles_do_not_intersect(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((10.1, 11), (10.5, 11.1))
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_vertical_line(self):
        line1 = Line((10., 10.), (10., 20.))
        line2 = Line((5., 15.), (15., 15.))
        intersection = line1.get_intersection(line2)
        self.assertEqual(Point(10., 15.), intersection)

    def test_top_of_screen_trajectory(self):
        trajectory = Line((349.824288866177, 298.7636286943965), (351.9456092097367, 300.8849490379562))
        border = Line((0, 300), (400, 300))
        intersection = trajectory.get_intersection(border)
        dx = (300. - trajectory.start.y) / (
                    (trajectory.end.y - trajectory.start.y) / (trajectory.end.x - trajectory.start.x))
        self.assertEqual(Point(dx + trajectory.start.x, 300.), intersection)


class TestPoint(unittest.TestCase):
    def test_point_iter(self):
        point1 = Point(0, 1)
        v = [p for p in point1]
        self.assertEqual([0, 1], v)

    def test_point_tuple_conversion(self):
        point1 = Point(0, 1)
        self.assertEqual((0, 1), tuple(point1))


# TODO: test episode end
# TODO: test snell layer

if __name__ == '__main__':
    unittest.main()

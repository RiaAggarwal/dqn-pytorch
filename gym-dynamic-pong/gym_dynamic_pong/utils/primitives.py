import math
from typing import Tuple, Union, Dict

import numpy as np


class Point:
    def __init__(self, x: Union[float, int], y: Union[float, int] = None):
        """
        Creates point (x, y). If no argument is specified for y, x is assumed to be an angle, and x and y are populated
        as a unit vector with direction `angle = x` from the right horizontal axis.

        :param x: x position or angle
        :param y: y position or None
        """
        if y is None:
            self.x = math.cos(x)
            self.y = math.sin(x)
        else:
            self.x = x
            self.y = y

    def l1_distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    def l2_distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def dot(self, other):
        return dot(self, other)

    def perp(self, other):
        return perp(self, other)

    @property
    def l2_norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def angle(self):
        if self.x == 0.:
            if self.y > 0:
                return math.pi / 2
            elif self.y < 0:
                return math.pi * 3 / 2
            else:
                return float('nan')
        else:
            if self.x > 0:
                return math.atan(self.y / self.x)
            else:
                return math.pi + math.atan(self.y / self.x)

    def __iter__(self):
        return (p for p in (self.x, self.y))

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return Point(self.x * other.x, self.y * other.y)
        elif isinstance(other, (float, int)):
            return Point(self.x * other, self.y * other)
        else:
            raise TypeError(f"multiplication not supported for type {type(self)} and {type(other)}")

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)

    def __eq__(self, other):
        assert isinstance(other, type(self)), f"cannot compare type {type(other)} to type {type(self)}"
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


def perp(p1: Point, p2: Point) -> Union[int, float]:
    """
    Cross product of point1 and point2

    :param p1: Point()
    :param p2: Point()
    :return: cross product scalar
    """
    return p1.x * p2.y - p1.y * p2.x


def dot(p1: Point, p2: Point) -> Union[int, float]:
    """
    Dot product of point1 and point2

    :param p1: Point()
    :param p2: Point()
    :return: dot product scalar
    """
    return p1.x * p2.x + p1.y * p2.y


class Line:
    def __init__(self, start: Union[Tuple[Union[float, int], Union[float, int]], Point],
                 end: Union[Tuple[Union[float, int], Union[float, int]], Point]):
        """
        A line segment object with methods to detect intersection and to compute angles.

        :param start: Point(x, y)
        :param end: Point(x, y)
        """
        if isinstance(start, tuple):
            self.start = Point(*start)
        else:
            assert isinstance(start, Point)
            self.start = start
        if isinstance(end, tuple):
            self.end = Point(*end)
        else:
            assert isinstance(end, Point)
            self.end = end

    def get_intersection(self, other) -> Union[Point, None]:
        """
        Return the intersection point of two line segments if they intersect, None if they don't

        :param other: A Line object
        :return: a `Point` or `None`
        """
        u = self.end - self.start
        v = other.end - other.start
        w = self.start - other.start
        det = perp(u, v)

        if abs(det) == 0.:  # they are parallel
            return None

        s_i = perp(v, w) / det
        if s_i < 0 or s_i > 1:
            return None

        t_i = perp(u, w) / det
        if t_i < 0 or t_i > 1:
            return None

        return self.start + u * s_i

    def in_segment(self, point: Point) -> bool:
        """
        Determine if collinear point `other` is in segment self.

        :param point: a point collinear with self
        """
        if self.start.x != self.end.x:
            if self.start.x <= point.x <= self.end.x:
                return True
            if self.start.x >= point.x >= self.end.x:
                return True
        else:
            if self.start.y <= point.y <= self.end.y:
                return True
            if self.start.y >= point.y >= self.end.y:
                return True
        return False

    def point1_before_point2(self, point1: Point, point2: Point) -> bool:
        """
        Determines whether `point1` comes before `point2` along the direction of the line segment

        :param point1: (x1, y1)
        :param point2: (x2, y2)
        """
        if self.start.l1_distance(point1) < self.start.l1_distance(point2):
            return True
        else:
            return False

    def angle_to_normal(self, other) -> float:
        angle = math.acos(dot(self.unit_vector, other.unit_vector))
        assert 0 <= angle <= math.pi, "Angle outside of expected bounds"
        return math.pi / 2 - angle

    @property
    def angle(self):
        return (self.end - self.start).angle

    @property
    def unit_vector(self) -> Point:
        v = self.end - self.start
        return v * (1 / v.l2_norm)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Line(self.start - other, self.end - other)
        else:
            raise TypeError(f"No operation `sub` defined for type {type(self)} and {type(other)}")

    def __add__(self, other):
        if isinstance(other, Point):
            return Line(self.start + other, self.end + other)
        else:
            raise TypeError(f"No operation `add` defined for type {type(self)} and {type(other)}")

    def __len__(self):
        return self.start.l2_distance(self.end)

    def __repr__(self):
        return f"Line(({self.start.x}, {self.start.y}), ({self.end.x}, {self.end.y}))"

    def __iter__(self):
        return (p for p in (self.start, self.end))


class Circle:
    def __init__(self, center: Point, radius: float, max_width=210, max_height=160):
        self.center = center
        self.radius = radius
        self.max_width = max_width
        self.max_height = max_height

    def get_intersection(self, line: Line) -> Union[Point, None]:
        """
        Return the first intersection point of the line segment and the circle if they intersect, None if they don't

        :param line: a `primitives.Line` object
        :return: The intersection point or None
        """
        # shift the coordinate system to the center of the circle
        line -= self.center
        d = line.end - line.start
        dr = d.dot(d)

        det = line.start.perp(line.end)

        discriminant = self.radius ** 2 * dr - det ** 2
        if discriminant <= 0:  # line misses (interpret tangent as a miss)
            return None
        else:
            discriminant = math.sqrt(discriminant)

            t1_x = (det * d.y + math.copysign(1., d.y) * d.x * discriminant) / dr
            t1_y = (-det * d.x + abs(d.y) * discriminant) / dr
            t1 = Point(t1_x, t1_y)

            t2_x = (det * d.y - math.copysign(1., d.y) * d.x * discriminant) / dr
            t2_y = (-det * d.x - abs(d.y) * discriminant) / dr
            t2 = Point(t2_x, t2_y)

            t1_in_segment = line.in_segment(t1)
            t2_in_segment = line.in_segment(t2)
            if t1_in_segment and t2_in_segment:
                if line.point1_before_point2(t1, t2):
                    return t1 + self.center
                else:
                    return t2 + self.center
            elif t1_in_segment:
                return t1
            elif t2_in_segment:
                return t2
            else:
                return None


class Rectangle:
    """
    Class to help render rectangular objects in numpy
    """

    def __init__(self, **kwargs):
        """
        A rectangle object with tools to detect interaction with other objects, and to render a canvas.

        :param max_width: the canvas width
        :param max_height: the canvas height
        :param height: the height of the rectangle
        :param width: the width of the rectangle
        """
        self._x_pos = 0
        self._y_pos = 0

        self.max_width = kwargs.get('max_width', 210)
        self.max_height = kwargs.get('max_height', 160)
        self.height = kwargs.pop('height')
        self.width = kwargs.pop('width')

        for v in self.max_width, self.max_height, self.width, self.height:
            assert isinstance(v, int), "Dimensions must be integer values"

    @property
    def left_bound(self):
        return self._x_pos - self.width / 2

    @property
    def right_bound(self):
        return self._x_pos + self.width / 2

    @property
    def top_bound(self):
        return self._y_pos + self.height / 2

    @property
    def bot_bound(self):
        return self._y_pos - self.height / 2

    @property
    def x_pos(self):
        return self._x_pos

    @x_pos.setter
    def x_pos(self, value):
        if value < 0:
            self._x_pos = 0
        elif value > self.max_width:
            self._x_pos = self.max_width
        else:
            self._x_pos = value

    @property
    def y_pos(self):
        return self._y_pos

    @y_pos.setter
    def y_pos(self, value):
        if value < 0:
            self._y_pos = 0
        elif value > self.max_height:
            self._y_pos = self.max_height
        else:
            self._y_pos = value

    @property
    def pos(self):
        return Point(self._x_pos, self._y_pos)

    @pos.setter
    def pos(self, value: Union[Tuple[Union[int, float], Union[int, float]], Point]):
        self.x_pos, self.y_pos = tuple(value)

    def get_edges(self) -> Dict[str, Line]:
        """
        Edges are assigned in a clockwise fashion so that the interior of the rectangle is to the right of the ray.

        :return:
        """
        lb = self.left_bound, self.bot_bound
        lt = self.left_bound, self.top_bound
        rb = self.right_bound, self.bot_bound
        rt = self.right_bound, self.top_bound
        return {
            'left'  : Line(lb, lt),
            'top'   : Line(lt, rt),
            'right' : Line(rt, rb),
            'bottom': Line(rb, lb)
        }

    def is_overlapping(self, other) -> bool:
        """
        Determine if `self` overlaps another rectangle

        :param other: Rectangle
        :return: `True` or `False`
        """
        if self.bot_bound > other.top_bound or self.top_bound < other.bot_bound or self.right_bound < other.left_bound \
                or self.left_bound > other.right_bound:
            return False
        else:
            return True

    def is_in(self, point: Point) -> bool:
        """
        Determine if `point` is in the rectangle

        :param point: `Point(x, y)`
        :return: True or False
        """
        if self.left_bound < point.x < self.right_bound and self.bot_bound < point.y < self.top_bound:
            return True
        else:
            return False

    def to_numpy(self) -> np.ndarray:
        """
        Renders the rectangle object on the canvas given by `max_height` and `max_width`

        :return: canvas as numpy array
        """
        # TODO: consider making this sparse
        out = np.zeros((self.max_height, self.max_width), dtype=np.bool)

        top = self.max_height - round(self.top_bound)
        bottom = self.max_height - round(self.bot_bound)
        left = round(self.left_bound)
        right = round(self.right_bound)

        out[top:bottom + 1, left:right + 1] = True
        return out

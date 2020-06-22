import math
import random
from typing import Tuple, Union, Dict
from abc import ABC, abstractmethod

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

    def __len__(self):
        return self.l2_norm

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
        if isinstance(other, type(self)):
            return self.x == other.x and self.y == other.y
        else:
            raise TypeError(f"No equality comparison for type Point and {type(other)}")

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

    @angle.setter
    def angle(self, value) -> None:
        """
        Rotates the line around the start such that the angle equals `value`

        :param value: The new angle of the line
        """
        self.end = self.start + Point(value) * len(self)

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

    def __eq__(self, other):
        if isinstance(other, Line):
            return self.start == other.start and self.end == other.end
        else:
            raise TypeError(f"No equality comparison for type Line and {type(other)}")

    def __len__(self):
        return self.start.l2_distance(self.end)

    def __repr__(self):
        return f"Line(({self.start.x}, {self.start.y}), ({self.end.x}, {self.end.y}))"

    def __iter__(self):
        return (p for p in (self.start, self.end))


class Shape(ABC):
    @abstractmethod
    def get_intersection(self, line: Line) -> Union[Tuple[Line, Point], None]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def pos(self):
        raise NotImplementedError()

    @pos.setter
    @abstractmethod
    def pos(self, value: Union[Tuple[Union[int, float], Union[int, float]], Point]):
        raise NotImplementedError()

    @abstractmethod
    def _get_edges(self) -> Dict[str, Line]:
        raise NotImplementedError()

    @abstractmethod
    def is_overlapping(self, other) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def is_in(self, point: Point) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def to_numpy(self, height: int, width: int) -> np.ndarray:
        raise NotImplementedError()


class Circle(Shape):
    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    @property
    def pos(self):
        return self.center

    def _get_edges(self) -> Dict[str, Line]:
        pass

    def is_overlapping(self, other) -> bool:
        raise NotImplementedError()

    def is_in(self, point: Point) -> bool:
        raise NotImplementedError()

    def to_numpy(self, height: int, width: int) -> np.ndarray:
        raise NotImplementedError()

    def _get_intersection_point(self, line: Line) -> Union[Point, None]:
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

    def _get_tangent(self, point: Point) -> Line:
        """
        The user must verify that `point` is on the circle

        :param point: `primitives.Point`
        :return: The tangent line. (only the direction of the line is considered)
        """
        line = Line(point, point + Point(1, 1))  # line with non-zero length starting at the intersection
        angle = (point - self.center).angle % (2 * math.pi)
        line.angle = math.pi / 2 + angle  # set line perpendicular to the intersection
        return line

    def get_intersection(self, line: Line) -> Union[Tuple[Line, Point], None]:
        point = self._get_intersection_point(line)
        if point is None:
            return None
        else:
            edge = self._get_tangent(point)
            return edge, point


class Rectangle(Shape):
    """
    A rectangular shape with lots of helper methods.
    """

    def __init__(self, **kwargs):
        """
        A rectangle object with tools to detect interaction with other objects, and to render a canvas.

        :param max_width: the canvas width
        :param max_height: the canvas height
        :param height: the height of the rectangle
        :param width: the width of the rectangle
        """
        self.x = 0
        self.y = 0

        self.height = kwargs.pop('height')
        self.width = kwargs.pop('width')

    def get_intersection(self, line: Line) -> Union[Tuple[Line, Point], None]:
        """
        Get the first point of intersection between the line and the object

        :param line: `primitives.Line` object
        :return: The edge and point of intersection as a tuple or `None`
        """
        result = None
        for e in self._get_edges():
            i = e.get_intersection(line)
            if i is not None:
                if result is None:
                    result = e, i
                elif line.point1_before_point2(i, result[1]):
                    result = e, i
        return result

    @property
    def left_bound(self):
        return self.x - self.width / 2

    @property
    def right_bound(self):
        return self.x + self.width / 2

    @property
    def top_bound(self):
        return self.y + self.height / 2

    @property
    def bot_bound(self):
        return self.y - self.height / 2

    @property
    def pos(self):
        return Point(self.x, self.y)

    @pos.setter
    def pos(self, value: Union[Tuple[Union[int, float], Union[int, float]], Point]):
        self.x, self.y = tuple(value)

    @property
    def top_edge(self):
        return Line((self.left_bound, self.top_bound), (self.right_bound, self.top_bound))

    @property
    def right_edge(self):
        return Line((self.right_bound, self.top_bound), (self.right_bound, self.bot_bound))

    @property
    def bot_edge(self):
        return Line((self.right_bound, self.bot_bound), (self.left_bound, self.bot_bound))

    @property
    def left_edge(self):
        return Line((self.left_bound, self.bot_bound), (self.left_bound, self.top_bound))

    def _get_edges(self) -> Tuple[Line, Line, Line, Line]:
        """
        Edges are assigned in a clockwise fashion so that the interior of the rectangle is to the right of the ray.

        :return:
        """
        return self.left_edge, self.top_edge, self.right_edge, self.bot_edge

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

    def to_numpy(self, height: int, width: int) -> np.ndarray:
        """
        Renders the rectangle object on the canvas given by `height` and `width`

        :param height: canvas height
        :param width: canvas width
        :return: canvas as numpy array
        """
        assert isinstance(height, int), f"height must be type int, not type {type(height)}"
        assert isinstance(width, int), f"width must be type int, not type {type(width)}"
        out = np.zeros((height, width), dtype=np.bool)

        top = height - round(self.top_bound)
        if top < 0:
            top = 0
        bottom = height - round(self.bot_bound)
        if bottom > height:
            bottom = height
        left = round(self.left_bound)
        if left < 0:
            left = 0
        right = round(self.right_bound)
        if right > width:
            right = width

        out[top:bottom + 1, left:right + 1] = True
        return out
#! /usr/bin/python

import math
from cagd.vec import vec2, vec3
from cagd.polyline import polyline
from cagd.bezier import bezier_surface, bezier_patches
import cagd.utils as utils
import copy
from math import *


class spline:
    # Interpolation modes
    INTERPOLATION_GIVEN_KNOTS = 0
    INTERPOLATION_EQUIDISTANT = 1
    INTERPOLATION_CHORDAL = 2
    INTERPOLATION_CENTRIPETAL = 3
    INTERPOLATION_FOLEY = 4

    # generates a spline that interpolates the given points using the given mode
    

    def __init__(self, degree, control_points = [], knots = None, color = "black", periodic = False):
        assert (degree >= 1)
        self.degree = degree
        self.knots = knots
        self.control_points = control_points
        self.color = color
        self.periodic = periodic

    # checks if the number of knots, controlpoints and degree define a valid spline
    def validate(self):
        knots = self.knots.validate()
        points = len(self.knots) == len(self.control_points) + self.degree + 1
        return knots and points

    def evaluate(self, t):
        a, b = self.support()
        assert (a <= t <= b)
        if t == self.knots[len(self.knots) - self.degree - 1]:
            # the spline is only defined on the interval [a, b)
            # it is useful to define self(b) as lim t->b self(t)
            t = t - 0.000001
        return self.de_boor(t, 1)[0]

    # returns the interval [a, b) on which the spline is supported
    def support(self):
        return (self.knots[self.degree], self.knots[len(self.knots) - self.degree - 1])

    def __call__(self, t):
        return self.evaluate(t)

    def tangent(self, t):
        pass

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    # calculates the de_boor scheme at a given value t
    # stops when the column is only "stop" elements long
    # returns that column as a list
    def de_boor(self, t, stop):
        assert stop >= 1
        assert None not in self.knots
        degree = self.degree
        idx = self.knots.knot_index(t)
        _de_boor_tables = [self.control_points[i] for i in range(idx - degree, idx + 1)]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                alpha = (t - self.knots[idx - degree + j]) / (self.knots[idx - degree + j + degree - r + 1] - self.knots[idx - degree + j])
                _de_boor_tables[j] = (1 - alpha) * _de_boor_tables[j - 1] + alpha * _de_boor_tables[j]
            if degree - r + 1 == stop:
                return _de_boor_tables[r:]
        
    # adjusts the control points such that it represents the same function,
    # but with an added knot
    def insert_knot(self, t):
        pass

    def get_axis_aligned_bounding_box(self):
        min_vec = copy.copy(self.control_points[0])
        max_vec = copy.copy(self.control_points[0])
        for p in self.control_points:
            # print("comparing {0} to {1} and {2}".format(p, min_vec, max_vec))
            if p.x < min_vec.x:
                min_vec.x = p.x
            if p.y < min_vec.y:
                min_vec.y = p.y
            if p.x > max_vec.x:
                max_vec.x = p.x
            if p.y > max_vec.y:
                max_vec.y = p.y
        return (min_vec, max_vec)

    def draw(self, scene, num_samples):
        i = self.degree - 1
        while i < len(self.knots) - self.degree - 2:
            i += 1
            k0 = self.knots[i]
            k1 = self.knots[i + 1]
            if k0 == k1:
                continue
            p0 = self(k0)
            for j in range(1, num_samples + 1):
                t = k0 + j / num_samples * (k1 - k0)
                p1 = self(t)
                scene.draw_line(p0, p1, self.color)
                p0 = p1

    def get_polyline_from_control_points(self):
        pl = polyline()
        for p in self.control_points:
            pl.append_point(p)
        return pl

    # generates a spline that interpolates the given points using the given mode
    # kts is only used as given knots in the mode: INTERPOLATION_GIVEN_KNOTS
    # returns that spline object
    def interpolate_cubic(mode, points, kts):
        m = len(points)
        if mode == spline.INTERPOLATION_GIVEN_KNOTS:
            knts = kts
        elif mode == spline.INTERPOLATION_EQUIDISTANT:
            knts = [i for i in range(m)]
        elif mode == spline.INTERPOLATION_CHORDAL:
            knts  = [0] * m
            for i in range(1, m):
                knts[i] = knts[i - 1] + abs(points[i] - points[i - 1])
        elif mode == spline.INTERPOLATION_CENTRIPETAL:
            knts = [0] * m
            for i in range(1, m):
                knts[i] = knts[i - 1] + abs(points[i] - points[i - 1]) ** 0.5
        elif mode == spline.INTERPOLATION_FOLEY:
            dis = [0] + [abs(points[i] - points[i - 1]) for i in range(1, m)] + [0]
            ang = [0] + [math.atan2(points[i].y - points[i - 1].y, points[i].x - points[i - 1].x) for i in range(1, m)] + [0]
            ang = [min(math.pi - x, math.pi / 2) for x in ang]
            knts = [0] * m
            for i in range(1, m):
                knts[i] = knts[i - 1] + dis[i] * (
                    1 + 3 / 2 * (ang[i - 1] * dis[i - 1]) / (dis[i - 1] + dis[i]) + 3 / 2 * (ang[i] * dis[i + 1]) / (dis[i + 1] + dis[i])
                )

        else:
            raise ValueError("Invalid interpolation mode")
        knoten = knots(6 + len(knts))
        knoten.knots = [knts[0]] * 3 + knts + [knts[-1]] * 3
        return spline.interpolate_cubic_given_knots(points, knoten)

    def interpolate_cubic_given_knots(points, knts):
        assert len(points) == len(knts) - 6
        m = len(points) - 1
        alpha = [0] * (m + 1)
        beta = [0] * (m + 1)
        gamma = [0] * (m + 1)
        zero = points[0] * 0
        for i in range(2, m + 1):
            alpha[i] = (knts[i + 2] - knts[i]) / (knts[i + 3] - knts[i])
            beta[i] = (knts[i + 2] - knts[i + 1]) / (knts[i + 3] - knts[i + 1])
            gamma[i] = (knts[i + 2] - knts[i + 1]) / (knts[i + 4] - knts[i + 1])
        diag1 = [0, -1] + [(1 - beta[i]) * (1 - alpha[i]) for i in range(2, m + 1)] + [-1 + gamma[-1], 0]
        diag2 = [1, 1 + alpha[2]] + [alpha[i] * (1 - beta[i]) + beta[i] * (1 - gamma[i]) for i in range(2, m + 1)] + [2 - gamma[-1], 1]
        diag3 = [0, -alpha[2]] + [beta[i] * gamma[i] for i in range(2, m + 1)] + [-1, 0]
        res = [points[0], zero] + [points[i] for i in range(1, m)] + [zero, points[-1]]
        cps = utils.solve_tridiagonal_equation(diag1, diag2, diag3, res)
        return spline(degree=3, control_points=cps, knots=copy.deepcopy(knts))

    # generates a spline that interpolates the given points and fulfills the definition
    # of a periodic spline with equidistant knots
    # returns that spline object
    def interpolate_cubic_periodic(points):
        n = len(points)
        diag1 = [1/6] * n
        diag2 = [2/3] * n
        diag3 = [1/6] * n
        cps = utils.solve_almost_tridiagonal_equation(diag1, diag2, diag3, points)
        knts = [i for i in range(n + 1)]
        return spline(degree=3, control_points=cps, knots=copy.deepcopy(knts), periodic=True)

    # for splines of degree 3, generate a parallel spline with distance dist
    # the returned spline is off from the exact parallel by at most eps
    def generate_parallel(self, dist, eps):
        assert (self.degree == 3)
        if dist == 0:
            return self

        para_spline = None
        return para_spline

    # generates a rotational surface by rotating the spline around the z axis
    # the spline is assumed to be on the xz-plane
    # num_samples refers to the number of interpolation points in the rotational direction
    # returns a spline surface object in three dimensions
    def generate_rotation_surface(self, num_samples):
        pass


class spline_surface:
    # the two directions of the parameter space
    DIR_U = 0
    DIR_V = 1

    # creates a spline of degrees n,m
    # degree is a tuple (n,m)
    def __init__(self, degree):
        du, dv = degree
        assert (du >= 1 and dv >= 1)
        self.degree = degree
        self.periodic = (False, False)
        self.knots = (None, None)  # tuple of both knot vectors
        self.control_points = [[]]  # 2dim array of control points

    # checks if the number of knots, controlpoints and degree define a valid spline
    def validate(self):
        if len(self.control_points) == 0:
            return False
        k1, k2 = self.knots
        d1, d2 = self.degree
        knots12 = k1.validate() and k2.validate()
        p1 = len(self.control_points)
        p2 = len(self.control_points[0])
        points1 = len(k1) == p1 + d1 + 1
        points2 = len(k2) == p2 + d2 + 1
        return knots12 and points1 and points2

    def evaluate(self, u, v):
        s1, s2 = self.support()
        a, b = s1
        c, d = s2
        assert (a <= u <= b and c <= v <= d)
        if u == b:
            u = u - 0.000001
        if v == d:
            v = v - 0.000001
        t = (u, v)
        return self.de_boor(t, (1, 1))[0][0]

    # return nested tuple ((a,b), (c,d))
    # the spline is supported in (u,v) \in [a,b)x[c,d]
    def support(self):
        k1, k2 = self.knots
        d1, d2 = self.degree
        s1 = (k1[d1], k1[len(k1) - d1 - 1])
        s2 = (k2[d2], k2[len(k2) - d2 - 1])
        return (s1, s2)

    def __call__(self, u, v):
        return self.evaluate(u, v)

    # calculates the de boor scheme at t = (u,v)
    # until there are only stop = (s1, s2) elements left
    def de_boor(self, t, stop):
        d1, d2 = self.degree
        k1, k2 = self.knots
        s1, s2 = stop
        u, v = t
        m1 = len(self.control_points)
        m2 = len(self.control_points[0])

        new_rows = [None for i in range(m1)]
        for row in range(m1):
            spl = spline(d2)
            spl.knots = k2
            spl.control_points = self.control_points[row]
            new_rows[row] = spl.de_boor(v, s2)

        new_pts = [None for i in range(s2)]
        for col in range(s2):
            spl = spline(d1)
            spl.knots = k1
            ctrl_pts = [new_rows[i][col] for i in range(m1)]
            spl.control_points = ctrl_pts
            new_pts[col] = spl.de_boor(u, s1)

        return new_pts

    def insert_knot(self, direction, t):
        if direction == self.DIR_U:
            self.__insert_knot_u(t)
        elif direction == self.DIR_V:
            self.__insert_knot_v(t)
        else:
            assert (False)

    def __insert_knot_v(self, t):
        du, dv = self.degree
        pu, pv = self.periodic
        ku, kv = self.knots
        nu = len(self.control_points)
        nv = len(self.control_points[0])
        for i in range(nu):
            row = self.control_points[i]
            spl = spline(dv)
            spl.control_points = copy.copy(row)
            spl.knots = copy.deepcopy(kv)
            spl.periodic = pv
            spl.insert_knot(t)
            self.control_points[i] = spl.control_points
            self.knots = (ku, spl.knots)

    def __insert_knot_u(self, t):
        du, dv = self.degree
        pu, pv = self.periodic
        ku, kv = self.knots
        nu = len(self.control_points)
        nv = len(self.control_points[0])
        new_control_points = [[None for i in range(nv)] for j in range(nu + 1)]
        for i in range(nv):
            col = [self.control_points[j][i] for j in range(nu)]
            spl = spline(du)
            spl.control_points = col
            spl.knots = copy.deepcopy(ku)
            spl.periodic = pu
            spl.insert_knot(t)
            for j in range(nu + 1):
                new_control_points[j][i] = spl.control_points[j]
            self.knots = (spl.knots, kv)
        self.control_points = new_control_points

    # build bezier patches based on the spline with multiple knots
    # and control points sitting also as bezier points.
    def to_bezier_patches(self):
        patches = bezier_patches()
        return patches


class knots:
    # creates a knots array with n elements
    def __init__(self, n):
        self.knots = [None for i in range(n)]

    def validate(self):
        prev = None
        for k in self.knots:
            if k is None:
                return False
            if prev is not None:
                if k < prev:
                    return False
            prev = k
        return True

    def __len__(self):
        return len(self.knots)

    def __getitem__(self, i):
        return self.knots[i]

    def __setitem__(self, i, v):
        self.knots[i] = v

    def __delitem__(self, i):
        del self.knots[i]

    def __iter__(self):
        return iter(self.knots)

    def insert(self, t):
        i = 0
        while self[i] < t:
            i += 1
        self.knots.insert(i, t)

    def knot_index(self, v):
        if self.knots[0] > v or self.knots[-1] < v:
            raise ValueError("knot value out of range")
        # binary search right most index
        l, r = 0, len(self.knots) - 1
        while l < r:
            m = (l + r) // 2
            if self.knots[m] > v:
                r = m
            else:
                l = m + 1
        return r - 1

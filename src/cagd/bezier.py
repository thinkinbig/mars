#!/usr/bin/python

import math
from cagd.vec import vec2, vec3
from cagd.polyline import polyline
import copy

class bezier_curve:
    def __init__(self, degree):
        assert (degree >= 0)
        self.degree = degree
        self.control_points = [None for i in range(degree + 1)]
        self.color = "black"

    def set_control_point(self, index, val):
        assert (index >= 0 and index <= self.degree)
        self.control_points[index] = val

    def get_control_point(self, index):
        assert (index >= 0 and index <= self.degree)
        return self.control_points[index]

    #evaluates the curve at t
    def evaluate(self, t):
        return self.__de_casteljeau(t, 1)[0]

    #evaluates tangent at t
    def tangent(self, t):
        last_two_ctrl_pts = self.__de_casteljeau(t, 2)
        a = last_two_ctrl_pts[0]
        b = last_two_ctrl_pts[1]
        return b - a

    #calculates the normal at t
    def normal(self, t):
        return self.tangent(t).normalize()
    
    #syntactic sugar so bezier curve can be evaluated as curve(t)
    #instead of curve.evaluate(t)
    def __call__(self, t):
        return self.evaluate(t)

    #calculates the de-casteljeau scheme until the column only has stop elements
    def __de_casteljeau(self, t, stop):
        assert (stop >= 1)
        column = self.control_points
        while len(column) > stop:
            new_column = [None for i in range(len(column) -1)]
            for i in range(len(new_column)):
                new_column[i] = (1 - t) * column[i] + t * column[i + 1]
            column = new_column
        return column

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    #calculates the bezier representation of the derivative
    def get_derivative(self):
        d1 = self.degree
        if (d1 == 0):
            return 0
        b_c = bezier_curve(d1 - 1)
        for i in range(d1):
            b_c.set_control_point(i, d1 * (self.control_points[i + 1] - self.control_points[i]))
        return b_c

    def get_axis_aligned_bounding_box(self):
        min_vec = copy.copy(self.control_points[0])
        max_vec = copy.copy(self.control_points[0])
        for p in self.control_points:
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
        p0 = self(0)
        for i in range(1, num_samples + 1):
            t = i / num_samples
            p1 = self(t)
            scene.draw_line(p0, p1, self.color)
            p0 = p1

    def get_polyline_from_control_points(self):
        pl = polyline()
        for p in self.control_points:
            pl.append_point(p)
        return pl


class bezier_surface:
    DIR_U = 0
    DIR_V = 1

    #creates a bezier surface of degrees n,m
    #the degree parameter is a tuple (n,m)
    def __init__(self, degree):
        d1, d2 = degree
        assert (d1 >= 0 and d2 >= 0)
        self.degree = degree
        self.control_points = [[None for i in range(d2 + 1)] for j in range(d1 + 1)]
        white = (1,1,1)
        self.color = (white, white, white, white)
        self.curvature = (None, None, None, None)

    def set_control_point(self, index1, index2, val):
        assert (index1 >= 0 and index1 <= self.degree[0])
        assert (index2 >= 0 and index2 <= self.degree[1])
        self.control_points[index1][index2] = val

    def get_control_point(self, index1, index2):
        assert (index1 >= 0 and index1 <= self.degree[0])
        assert (index2 >= 0 and index2 <= self.degree[1])
        return self.control_points[index1][index2]

    def evaluate(self, t1, t2):
        return self.__de_casteljeau(t1, t2, (1, 1))[0][0]

    #sets the colors at the corners
    #c00 is the color at u=v=0, c01 is the color at u=0 v=1, etc
    #a color is a tuple (r,g,b) with values between 0 an 1
    def set_colors(self, c00, c01, c10, c11):
        self.color = (c00, c01, c10, c11)

    #sets the curvature at the corners
    #c00 is the curvature at u=v=0, c01 is the curvature at u=0 v=1, etc
    def set_curvature(self, c00, c01, c10, c11):
        self.curvature = (c00, c01, c10, c11)

    def __call__(self, t):
        t1, t2 = t
        return self.evaluate(t1, t2)

    def __de_casteljeau(self, t1, t2, stop):
        s1, s2 = stop
        d1, d2 = self.degree
        assert (s1 >= 1 and s2 >= 1)
        d1 += 1 #number of control points in each direction
        d2 += 1

        #apply the casteljeau scheme in one direction,
        #ie, reduce dimension from (d1, d2) to (s1, d2)
        column = self.control_points
        while d1 > s1:
            d1 -= 1
            new_column = [[None for i in range(d2)] for j in range(d1)]
            for i in range(d1):
                for j in range(d2):
                    new_column[i][j] = (1 - t1) * column[i][j] + t1 * column[i + 1][j]
            column = new_column

        #apply the casteljeau scheme in the other direction,
        #ie, reduce dimension from (s1, d2) to (s1, s2)
        while d2 > s2:
            d2 -= 1
            new_column = [[None for i in range(d2)] for j in range(d1)]
            for i in range(d1):
                for j in range(d2):
                    new_column[i][j] = (1 - t2) * column[i][j] + t2 * column[i][j + 1]
            column = new_column

        return column

    def normal(self, t1, t2):
        return t1.cross(t2).normalize()

    def get_derivative(self, direction):
        cps = self.control_points
        d1, d2 = self.degree
        if direction == bezier_surface.DIR_U:
            b_u = bezier_surface((d1 - 1, d2))
            for i in range(d1):
                for j in range(d2 + 1):
                    b_u.control_points[i][j] = (d1) * (cps[i + 1][j] - cps[i][j])
            return b_u
        elif direction == bezier_surface.DIR_V:
            b_v = bezier_surface((d1, d2 - 1))
            for i in range(d1 + 1):
                for j in range(d2):
                    b_v.control_points[i][j] = (d2) * (cps[i][j + 1] - cps[i][j])
            return b_v
        else:
            raise ValueError("invalid direction")

    def subdivide(self, t1, t2):
        b0,b1 = self.__subdivide_u(t1)
        b00,b01 = b0.__subdivide_v(t2)
        b10,b11 = b1.__subdivide_v(t2)
        return [b00, b01, b10, b11]

    def __subdivide_u(self, t):
        du, dv = self.degree
        left = bezier_surface((du, dv))
        right = bezier_surface((du, dv))
        for k in range(du+1):
            pts = self.__de_casteljeau(t, 0, (du-k+1, dv+1))
            left.control_points[k] = pts[0]
            right.control_points[-(k+1)] = pts[-1]
        return (left, right)

    def __subdivide_v(self, t):
        du, dv = self.degree
        left = bezier_surface((du, dv))
        right = bezier_surface((du, dv))
        for k in range(dv+1):
            pts = self.__de_casteljeau(0, t, (du+1, dv-k+1))
            for i in range(du+1):
                left.control_points[i][k] = pts[i][0]
                right.control_points[i][-(k+1)] = pts[i][-1]
        return (left, right)


class bezier_patches:
    CURVATURE_GAUSSIAN = 0
    CURVATURE_AVERAGE = 1
    CURVATURE_PRINCIPAL_MAX = 2 #Maximale Hauptkruemmung
    CURVATURE_PRINCIPAL_MIN = 3 #Minimale Hauptkruemmung
    COLOR_MAP_LINEAR = 4
    COLOR_MAP_CUT = 5
    COLOR_MAP_CLASSIFICATION = 6

    def __init__(self):
        self.patches = []

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, p):
        return self.patches[p]

    def __setitem__(self, i, p):
        self.patches[i] = p

    def __delitem__(self, p):
        del self.patches[p]

    def __iter__(self):
        return iter(self.patches)

    def append(self, p):
        self.patches.append(p)

    #refines patches by subdividing each patch into four new patches
    #there are 4^num times more patches after calling this function
    def refine(self, num):
        for i in range(num):
            new_patches = bezier_patches()
            for p in self:
                new = p.subdivide(0.5, 0.5)
                for n in new:
                    new_patches.append(n)
            self.patches = new_patches
    
    
    def get_derivative(self, patch):
        d1, d2 = patch.degree
        b_u_surface = patch.get_derivative(bezier_surface.DIR_U)
        b_v_surface = patch.get_derivative(bezier_surface.DIR_V)
        b_uu_surface = b_u_surface.get_derivative(bezier_surface.DIR_U)
        b_uv_surface = b_u_surface.get_derivative(bezier_surface.DIR_V)
        b_vv_surface = b_v_surface.get_derivative(bezier_surface.DIR_V)
        b_u, b_v, b_uu, b_uv, b_vv = [], [], [], [], []
        for i in range(2):
            for j in range(2):
                b_u.append(b_u_surface.control_points[-i][-j])
                b_v.append(b_v_surface.control_points[-i][-j])
                b_uu.append(b_uu_surface.control_points[-i][-j])
                b_uv.append(b_uv_surface.control_points[-i][-j])
                b_vv.append(b_vv_surface.control_points[-i][-j])
        return b_u, b_v, b_uu, b_uv, b_vv

    def gauss_and_middle_curvature(self, patch):
        K, H = [], []
        b_u, b_v, b_uu, b_uv, b_vv = self.get_derivative(patch)
        for i in range(4):
            print(b_u[i], b_v[i])
            normal = patch.normal(b_u[i], b_v[i])
            E = b_u[i].dot(b_u[i])
            F = b_u[i].dot(b_v[i])
            G = b_v[i].dot(b_v[i])

            e = b_uu[i].dot(normal)
            f = b_uv[i].dot(normal)
            g = b_vv[i].dot(normal)
            K.append((e * g - f ** 2) / (E * G - F ** 2))
            H.append((e * G + g * E - 2 * f * F) / ((E * G - F ** 2) * 2))
        return K, H

    def main_curvature(self, K, H):
        K_max, K_min = [], []
        for i in range(4):
            K_max.append(H[i] + math.sqrt(H[i] ** 2 - K[i]))
            K_min.append(H[i] - math.sqrt(H[i] ** 2 - K[i]))
        return K_max, K_min

    def visualize_curvature(self, curvature_mode, color_map):
        #calculate curvatures at each corner point
        #set colors according to color map
        def h(x, K, H , K_max, K_min):
            if x < 0.25:
                return (0, 4 * x, 1)
            elif x < 0.5:
                return (0, 1, 2 - 4 * x)
            elif x < 0.75:
                return (4 * x - 2, 1, 0)
            else:
                return (1, 4 - 4 * x, 0)
        def f_cut(x, K, H, K_max, K_min):
            if x < 0:
                return 0
            elif x <= 1:
                return x
            else:
                return 1
        def f_linear(x, K, H, K_max, K_min):
            res = []
            for i in range(4):
                k0, h0, k_max, k_min = K[i], H[i], K_max[i], K_min[i]
                # find the maximum and minimum curvature
                max_curvature = max(k0, h0, k_max, k_min)
                min_curvature = min(k0, h0, k_max, k_min)
                res.append((x - min_curvature) / (max_curvature - min_curvature))
            return res
        def f_classification(x, K, H, K_max, K_min):
            if x < 0:
                return 0
            elif x == 0:
                return 0.5
            else:
                return 1
        if color_map == bezier_patches.COLOR_MAP_CUT:
            func = f_cut
        elif color_map == bezier_patches.COLOR_MAP_LINEAR:
            func = f_linear
        else:
            func = f_classification
        for patch in self.patches:
            K, H = self.gauss_and_middle_curvature(patch)
            K_max, K_min = self.main_curvature(K, H)
            if curvature_mode == bezier_patches.CURVATURE_GAUSSIAN:
                curvature = K
            elif curvature_mode == bezier_patches.CURVATURE_AVERAGE:
                curvature = H
            elif curvature_mode == bezier_patches.CURVATURE_PRINCIPAL_MAX:
                curvature = K_max
            elif curvature_mode == bezier_patches.CURVATURE_PRINCIPAL_MIN:
                curvature = K_min
            else:
                raise ValueError("Invalid curvature mode")
            patch.set_curvature(*curvature)
            patch.set_colors(*[h(func(x, K, H, K_max, K_min)) for x in curvature])
    def export_off(self):
        def export_point(p):
            return str(p.x) + " " + str(p.y) + " " + str(p.z)

        def export_colors(c):
            s = ""
            for x in c:
                s += str(x)
                s += " "
            s += "1" #opacity
            return s

        s = "CBEZ333\n"
        for patch in self:
            #coordinates
            for row in patch.control_points:
                for p in row:
                    s += export_point(p)
                    s += "\n"

            #colors
            s += export_colors(patch.color[0])
            s += "\n"
            s += export_colors(patch.color[2])
            s += "\n"
            s += export_colors(patch.color[1])
            s += "\n"
            s += export_colors(patch.color[3])
            s += "\n"
            s += "\n"

        return s

    def export_standard_off(self):
        def export_numbers(degree):
            d1, d2 = degree
            n_v = (d1 + 1)*(d2 + 1)
            n_f = d1*d2
            n_e = d1*(d2 + 1) + d2*(d1 + 1)
            patch_num = len(self)
            return str(n_v*patch_num) + " " + str(n_f*patch_num) + " " + str(n_e*patch_num) + "\n"

        def export_vertex(v):
            return str(v.x) + " " + str(v.y) + " " + str(v.z) + "\n"

        def export_face(f_vs):
            s = "4  "
            for f_v in f_vs:
                s += str(f_v)
                s += " "
            return s

        def avg_color(f_cs):
            avg_f_c = [0, 0, 0]
            for f_c in f_cs:
                avg_f_c[0] += f_c[0]
                avg_f_c[1] += f_c[1]
                avg_f_c[2] += f_c[2]
            return (round(255*avg_f_c[0]/4),
                    round(255*avg_f_c[1]/4),
                    round(255*avg_f_c[2]/4))

        def export_color(f_c):
            return " " + str(f_c[0]) + " " + str(f_c[1]) + " " + str(f_c[2]) + "\n"

        s = "OFF\n\n"
        s += export_numbers(self[0].degree)

        for patch in self:
            for v_row in patch.control_points:
                for vertex in v_row:
                    s += export_vertex(vertex)
        start_v = 0
        for patch in self:
            cps = patch.control_points
            row_num = len(cps)
            col_num = len(cps[0])
            for row in range(row_num - 1):
                for col in range(col_num - 1):
                    first_v = start_v + row*col_num + col
                    f_vertices = (first_v, first_v + 1, first_v + 1 + col_num, first_v + col_num)
                    s += export_face(f_vertices)
                    s += export_color(avg_color(patch.color))
            start_v += 16
        return s

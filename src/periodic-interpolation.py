#!/usr/bin/python

from math import pi, cos, sin
import cagd.scene_2d as scene
from cagd.vec import vec2
from cagd.spline import spline, knots
from cagd.polyline import polyline

#returns a list of num_samples points that are uniformly distributed on the unit circle
def unit_circle_points(num_samples):
    angle = 2 * pi / num_samples
    return [vec2(cos(x * angle), sin(x * angle)) for x in range(num_samples)]

#calculates the deviation between the given spline and a unit circle
def calculate_circle_deviation(spline):
    sample_number = 888
    a, b = spline.support()
    intervall = b - a
    delta = intervall / sample_number
    t = [a + i * delta for i in range(sample_number)]
    s = [spline.evaluate(t[i]) for i in range(sample_number)]
    err = [(s[i].x ** 2 + s[i].y ** 2) ** 0.5 - 1 for i in range(sample_number)]
    avg_err = sum(err) / sample_number
    abs_err = [abs(err[i]) for i in range(sample_number)]
    max_err = max(abs_err)
    print("average error: ", avg_err)
    print("maximal error: ", max_err)





#interpolate 6 points with a periodic spline to create the number "8"
pts = [vec2( 0, 2.5), vec2(-1, 1), vec2( 1,-1), vec2( 0,-2.5), vec2(-1,-1), vec2(1,1)]
pts_line = polyline()
pts_line.points = pts
pts_line.set_color("red")
s = spline.interpolate_cubic_periodic(pts)
p = s.get_polyline_from_control_points()
p.set_color("blue")
sc = scene.scene()
sc.set_resolution(900)
sc.add_element(s)
sc.add_element(p)
sc.add_element(pts_line)

#generate a spline that approximates the unit circle
n = 8
circle_pts = unit_circle_points(n)
circle = spline.interpolate_cubic_periodic(points=circle_pts)
sc.add_element(circle)
calculate_circle_deviation(circle)
sc.write_image()
sc.show()


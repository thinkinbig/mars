#!/usr/bin/python
import os
import math
from cube import CubeEdges, CubeVertexDirections, CubeVertices, CubeEdgeFlags, CubeTriangles
from off import triangle_list
from vec3 import vec3, vec_from_list

ISOVAL = 1  # iso-value
SUBDIV = 32  # resolution

SPHERE = 0
OCTAHEDRON = 1
CUBE = 2
TORUS = 3



# lookup table for edge intersections
vertices = {}
class cubic:
    def __init__(self, mode):
        self.vertices = [vec_from_list(v) for v in CubeVertices]
        self.edges = CubeEdges
        self.vertex_directions = CubeVertexDirections
        self.edgesflags = CubeEdgeFlags
        self.triTables = CubeTriangles
        self.evaluate = []
        if mode == SPHERE:
            func = sphere
        elif mode == OCTAHEDRON:
            func = octahedron
        elif mode == CUBE:
            func = cube_func
        elif mode == TORUS:
            func = torus
        else:
            raise ValueError("Invalid mode")
        self.function = func

    def evaluate_value(self):
        for i in range(len(self.vertices)):
            self.evaluate.append(self.function(self.vertices[i]) - ISOVAL)

    def add_interpolation(self, edge_index, p1, p2, axis, val1, val2):
        if (val1 <= 0 and val2 >= 0) or (val1 >= 0 and val2 <= 0):
            if axis == 0:
                x = interpolate_value(p1.x, p2.x, val1, val2)
                y = p1.y
                z = p1.z
            elif axis == 1:
                x = p1.x
                y = interpolate_value(p1.y, p2.y, val1, val2)
                z = p1.z
            elif axis == 2:
                x = p1.x
                y = p1.y
                z = interpolate_value(p1.z, p2.z, val1, val2)
            else:
                raise ValueError("Invalid axis")
            vertices[edge_index] = vec3(x, y, z)
            print("vertices[{}] = {}".format(edge_index, vertices[edge_index]))
        else:
            raise ValueError("Invalid interpolation")


def sphere(vec):
    return abs(vec)

def octahedron(vec):
    return abs(vec.x) + abs(vec.y) + abs(vec.z)

def cube_func(vec):
    return max(max(abs(vec.x), abs(vec.y)), abs(vec.z))

def torus(vec):
    x = vec.x
    y = vec.y
    z = vec.z
    c = x*x + y*y + z*z + .7*.7 - .2*.2
    d = 4 * .7*.7 * ( x*x + y*y)
    return c*c - d

# interpolate only happens when isosurface crosses edge
def interpolate_value(p1, p2, val1, val2):
    assert (val1 <= 0 and val2 >= 0) or (val1 >= 0 and val2 <= 0)
    return p2 - (p2 - p1) * val2 / (val2 - val1)

def march_cubes(mode):
    cube = cubic(mode=mode)
    # evaluate function at each vertex
    cube.evaluate_value()
    cube_index = 0
    for i in range(8):
        if cube.evaluate[i] <= 0:
            cube_index |= 1 << i
    # get edges that are intersected by isosurface
    edge = cube.edgesflags[cube_index]
    # generate 12-bit number that represents which edges are intersected
    for i in range(12):
        if (edge & (1 << i)) != 0:
            direction = cube.vertex_directions[i]
            if (direction[0]) != 0:
                axis = 0
            elif (direction[1]) != 0:
                axis = 1
            elif (direction[2]) != 0:
                axis = 2
            cube.add_interpolation(i,
                                   cube.vertices[cube.edges[i][0]],
                                   cube.vertices[cube.edges[i][1]],
                                   axis,
                                   cube.evaluate[cube.edges[i][0]],
                                   cube.evaluate[cube.edges[i][1]])
    # generate triangles
    triangles = triangle_list()
    tri_list = cube.triTables[cube_index]
    for m in range(len(tri_list)):
        if tri_list[m] == -1:
            break
        triangles.append(vertices[tri_list[m]])
    

    # write triangles to file
    path = "marschall_cubes.off"
    f = open(path, 'w')
    f.close()
march_cubes(mode=TORUS)


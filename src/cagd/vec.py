#!/usr/bin/python
import math
class vec3:
    def __init__(self, a, b, c):
        self.x = a
        self.y = b
        self.z = c
    
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return not self == other
    
    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

    def normalize(self):
        assert abs(self) != 0
        return vec3(self.x / abs(self), self.y / abs(self), self.z / abs(self))

    def __abs__(self):
        return math.sqrt(self.dot(self))

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __copy__(self):
        return vec3(self.x, self.y, self.z)


class vec2:
    def __init__(self, a, b):
        self.x = a
        self.y = b
    
    def zero():
        return vec2(0,0)
    
    def rotate_90(self):
        return vec2(-self.y, self.x)
    
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def __add__(self, other):
        return vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return vec2(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def cross(self, other):
        return self.x * other.y - self.y * other.x
    
    def normalize(self):
        return vec2(self.x / abs(self), self.y / abs(self))

    def __abs__(self):
        return math.sqrt(self.dot(self))

    def __hash__(self):
        return hash((self.x, self.y))

    def __copy__(self):
        return vec2(self.x, self.y)

    def __truediv__(self, other):
        return vec2(self.x / other, self.y / other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

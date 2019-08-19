import numpy

def rad(dgr):
    return dgr * (numpy.pi/180)

def dgr(rad):
    return rad * (180/numpy.pi)

def normalize(vector):
    return vector/numpy.linalg.norm(vector)

def angle(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    return numpy.arccos(numpy.dot(v1, v2)/(numpy.linalg.norm(v1)*numpy.linalg.norm(v2)))

def translate(pos):
    return numpy.array([[1, 0, 0, pos[0]],
                        [0, 1, 0, pos[1]],
                        [0, 0, 1, pos[2]],
                        [0, 0, 0, 1]], dtype=numpy.float32)

def rotate(angle, axis, pt = None):
    rads = rad(angle)
    v = normalize(axis)
    c = numpy.cos(rads)
    omc = 1-c
    s = numpy.sin(rads)
    r = numpy.array([[v[0]*v[0]*omc + c,   v[0]*v[1]*omc - v[2]*s, v[0]*v[2]*omc + v[1]*s, 0],
                        [v[0]*v[1]*omc + v[2]*s, v[1]*v[1]*omc + c,   v[1]*v[2]*omc - v[0]*s, 0],
                        [v[0]*v[2]*omc - v[1]*s, v[1]*v[2]*omc + v[0]*s, v[2]*v[2]*omc + c,   0],
                        [0, 0, 0, 1]], dtype=numpy.float32)
    if pt == None:
        return r
    pt = numpy.array(pt)
    return translate(pt)@r@translate(-pt)

def scale(factors, pt = None):
    s = numpy.array([[factors[0], 0, 0, 0],
                        [0, factors[1], 0, 0],
                        [0, 0, factors[2], 0],
                        [0, 0, 0,          1]],dtype=numpy.float32)
    if pt == None:
        return s
    pt = numpy.array(pt)
    return translate(pt)@s@translate(-pt)

def lookAt(eye, at, up):
    n = normalize(at-eye)
    u = normalize(numpy.cross(n[0:3], up[0:3]))
    v = normalize(numpy.cross(u[0:3], n[0:3]))

    rotate = numpy.array([[u[0], v[0], -n[0], 0],
                          [u[1], v[1], -n[1], 0],
                          [u[2], v[2], -n[2], 0],
                          [0,    0,    0,    1]], dtype=numpy.float32).transpose()
    return rotate@translate(-eye)

def camera(pos, dir, up):
    n = normalize(dir[0:3])
    u = normalize(numpy.cross(n[0:3], up[0:3]))
    v = normalize(numpy.cross(u[0:3], n[0:3]))

    rotate = numpy.array([[u[0], u[1], u[2], 0],
                          [v[0], v[1], v[2], 0],
                          [-n[0], -n[1], -n[2], 0],
                          [0,    0,    0,    1]], dtype=numpy.float32)
    return rotate@translate(-pos)

def frustum(left, right, top, bottom, near, far):
    rl = right-left
    tb = top-bottom
    fn = far-near
    
    return numpy.array([[2*near/rl, 0,         (right+left)/rl, 0],
                        [0,         2*near/tb, (top+bottom)/tb, 0],
                        [0,         0,        -(far+near)/fn,   -(2*far*near)/fn],
                        [0,         0,        -1,               0]], dtype=numpy.float32)

def ortho(left, right, top, bottom, near, far):
    rl = right-left
    tb = top-bottom
    fn = far-near
    
    return numpy.array([[2/rl, 0,    0,    -(right+left)/rl],
                        [0,    2/tb, 0,    -(top+bottom)/tb],
                        [0,    0,   -2/fn, -(far+near)/fn  ],
                        [0,    0,    0,     1              ]], dtype=numpy.float32)

def perspective(fovy, aspect, near, far):
    ymax = near*numpy.tan(.5*rad(fovy))
    ymin = -ymax
    xmin = ymin*aspect
    xmax = ymax*aspect
    return frustum(xmin, xmax, ymax, ymin, near, far)


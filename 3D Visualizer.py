import pygame
from pygame.locals import *
from MV import *
from ctypes import c_void_p
from OpenGL.GL import *
from OpenGL.GL import shaders
from CalcParser import *
from os import system
from PIL import Image
import time
from multiprocessing import Pool
from functools import partial
from itertools import product

vertex_shader_graph = '''
#version 420

in vec3 vpos_modelspace;
in vec3 vnorm_modelspace;
uniform mat4 mvp;
smooth out vec3 worldPos;
smooth out vec3 worldNormal;

void main(){
    gl_Position = mvp * vec4(vpos_modelspace, 1.0);
    worldPos = vpos_modelspace;
    worldNormal = vnorm_modelspace;
}
'''

fragment_shader_graph = '''
#version 420

subroutine vec4 ObjectType();
subroutine uniform ObjectType culled;

out vec4 fragcolor;
uniform bool cutoff;
uniform float zcutoff;
uniform vec2 zbounds;

uniform vec3 lightPos;
uniform mat4 msrb;
uniform mat4 msr;
uniform vec3 camPos;
smooth in vec3 worldPos;
smooth in vec3 worldNormal;
vec3 ambientColor = vec3(0.0, .8, 0.0);
vec3 diffuseColor = vec3(0.0, 1.0, 0.0);
vec3 specColor = vec3(1.0, 1.0, 1.0);
float shininess = 50.0;

void main(){
    fragcolor = culled();
}

vec4 blinnphong(){
    vec3 P = worldPos;
    vec3 N = worldNormal;
    vec3 L = lightPos;
    vec3 lightDir = normalize(L - P);
    float lamb = max(dot(lightDir, N),0.0);
    float spec = 0.0;
    if(lamb > 0.0){
        vec3 viewDir = normalize(camPos-P);
        vec3 halfway = normalize(lightDir + viewDir);
        spec = pow(max(dot(halfway, N), 0.0), shininess);
    }
    return vec4(ambientColor+lamb*diffuseColor+spec*specColor, 1.0);
}

subroutine(ObjectType)
vec4 culled_object(){
    float margin = .02*(zbounds.x-zbounds.y);
    if(worldPos.z < zcutoff+margin && worldPos.z > zcutoff-margin)
        return vec4(1.0, 1.0, 1.0, 1.0);
    else if(cutoff)
        return vec4(0.0, 0.0, 0.0, 0.0);
    return blinnphong();
}

subroutine(ObjectType)
vec4 border(){
    return vec4(1.0, 1.0, 1.0, 1.0);
}
'''

vertex_shader_billboard = '''
#version 420

in vec3 vpos_modelspace;
in vec2 billboard_size;
uniform mat4 mv;
out vec2 size;

void main(){
    size = billboard_size;
    gl_Position = mv * vec4(vpos_modelspace, 1.0);
}
'''

geom_shader_billboard = '''
#version 420

subroutine void BillboardType();
subroutine uniform BillboardType type;

layout(points) in;
layout(triangle_strip, max_vertices=18) out;

uniform mat4 proj_matrix;

in vec2 size[];

out vec2 UV;
out vec4 color;
flat out int textured;

void main(){
    type();
}

subroutine(BillboardType)
void indices(){
    vec4 p = gl_in[0].gl_Position;
    gl_Position = proj_matrix * vec4(p.x - .5*size[0].x*(p.z/500), p.y - .5*size[0].y*(p.z/500), p.zw);
    UV = vec2(0.0, 0.0);
    textured = 1;
    EmitVertex();

    gl_Position = proj_matrix * vec4(p.x - .5*size[0].x*(p.z/500), p.y + .5*size[0].y*(p.z/500), p.zw);
    UV = vec2(0.0, 1.0);
    textured = 1;
    EmitVertex();

    gl_Position = proj_matrix * vec4(p.x + .5*size[0].x*(p.z/500), p.y - .5*size[0].y*(p.z/500), p.zw);
    UV = vec2(1.0, 0.0);
    textured = 1;
    EmitVertex();

    gl_Position = proj_matrix * vec4(p.x + .5*size[0].x*(p.z/500), p.y + .5*size[0].y*(p.z/500), p.zw);
    UV = vec2(1.0, 1.0);
    textured = 1;
    EmitVertex();
   
    EndPrimitive();
}

float ptsize = 4;

subroutine(BillboardType)
void raycast(){
    vec4 p = gl_in[0].gl_Position;

    gl_Position = proj_matrix * (p + vec4(-.5*ptsize*(p.z/500), -.5*ptsize*(p.z/500), 0.0, 0.0));
    color = vec4(1.0, 0.0, 0.0, 1.0);
    textured = 0;
    EmitVertex();

    gl_Position = proj_matrix * (p + vec4(-.5*ptsize*(p.z/500), .5*ptsize*(p.z/500), 0.0, 0.0));
    color = vec4(1.0, 0.0, 0.0, 1.0);
    textured = 0;
    EmitVertex();

    gl_Position = proj_matrix * (p + vec4(.5*ptsize*(p.z/500), -.5*ptsize*(p.z/500), 0.0, 0.0));
    color = vec4(1.0, 0.0, 0.0, 1.0);
    textured = 0;
    EmitVertex();

    gl_Position = proj_matrix * (p + vec4(.5*ptsize*(p.z/500), .5*ptsize*(p.z/500), 0.0, 0.0));
    color = vec4(1.0, 0.0, 0.0, 1.0);
    textured = 0;
    EmitVertex();

    EndPrimitive();

    gl_Position = proj_matrix * (vec4(p.x - .5*size[0].x*(p.z/500), p.y - .5*size[0].y*(p.z/500), p.zw) + vec4(.5*size[0].x*(p.z/500), -.5*size[0].y*(p.z/500), 0.0, 0.0));
    UV = vec2(0.0, 0.0);
    textured = 1;
    EmitVertex();

    gl_Position = proj_matrix * (vec4(p.x - .5*size[0].x*(p.z/500), p.y + .5*size[0].y*(p.z/500), p.zw) + vec4(.5*size[0].x*(p.z/500), -.5*size[0].y*(p.z/500), 0.0, 0.0));
    UV = vec2(0.0, 1.0);
    textured = 1;
    EmitVertex();

    gl_Position = proj_matrix * (vec4(p.x + .5*size[0].x*(p.z/500), p.y - .5*size[0].y*(p.z/500), p.zw) + vec4(.5*size[0].x*(p.z/500), -.5*size[0].y*(p.z/500), 0.0, 0.0));
    UV = vec2(1.0, 0.0);
    textured = 1;
    EmitVertex();

    gl_Position = proj_matrix * (vec4(p.x + .5*size[0].x*(p.z/500), p.y + .5*size[0].y*(p.z/500), p.zw) + vec4(.5*size[0].x*(p.z/500), -.5*size[0].y*(p.z/500), 0.0, 0.0));
    UV = vec2(1.0, 1.0);
    textured = 1;
    EmitVertex();
   
    EndPrimitive();
}

'''

fragment_shader_billboard = '''
#version 420

in vec2 UV;
in vec4 color;
flat in int textured;
out vec4 frag_color;

uniform sampler2D tex;

void main(){
    if(textured == 1)
	    frag_color = texture(tex, -UV);
    else
        frag_color = color;
}
'''

vertex_shader_normals = '''
#version 420

in vec3 vpos_modelspace;
in vec3 vnorm_modelspace;
uniform mat4 msr;
uniform mat4 bounds_matrix;

out Vertex{
    vec4 worldNormal;
    vec4 color;
} vertex_out;

void main(){
    gl_Position = msr*bounds_matrix*vec4(vpos_modelspace, 1.0);
    vertex_out.worldNormal = msr*vec4(vnorm_modelspace, 1.0);
    vertex_out.color = vec4(0.0, 0.0, 1.0, 1.0);
}
'''

geom_shader_normals = '''
#version 420

in Vertex{
    vec4 worldNormal;
    vec4 color;
} vertex_out[];

out vec4 color;
uniform mat4 vp;

layout(points) in;
layout(line_strip, max_vertices=2) out;

void main(){
    vec3 P = gl_in[0].gl_Position.xyz;
    vec3 N = vertex_out[0].worldNormal.xyz;
    gl_Position = vp * vec4(P, 1.0);
    color = vertex_out[0].color;
    EmitVertex();
    gl_Position = vp * vec4(P+N, 1.0);
    color = vertex_out[0].color;
    EmitVertex();
    EndPrimitive();
}
'''

fragment_shader_normals = '''
#version 420

in vec4 color;
out vec4 frag_color;

void main(){
    frag_color = color;
}
'''

screensize = (800, 600)
DEBUG_NORMALS = False

def pt(function, p):
    x = p[0]
    y = p[1]
    return numpy.array((x, y, eval(function)), dtype=numpy.float32)

def nrm(gradients, i):
    return normalize(numpy.cross(gradients[i+1]-gradients[i], gradients[i+3]-gradients[i+2]))

def invnrm(inv_bounds_matrix, i):
    return normalize((inv_bounds_matrix@numpy.array([i[0], i[1], i[2], 1], dtype=numpy.float32))[0:3])

def main():
    pygame.init()
    labelfont = pygame.font.SysFont('arial', 12)

    while True:
        print('''
        Enter a function to plot.
        Syntax list: 
            sin(angle), cos(angle), tan(angle)
            sinh(angle), cosh(angle), tanh(angle)
            sqrt(val), pow(val, power), log(val, exp)
            deg(radians), rad(degrees)
            e = 2.71828
            pi = 3.14159
            ''')
        
        #function = input('Function: ')
        #function = 'pow(x,2)+pow(y,2)'lwl
        function = 'cos(x)+sin(y)'
        #function = 'cos(x/y+1)'
        if not check_valid(function):
            system('cls')
            print('Invalid characters used.')
            continue
        var = vars(function)
        if len(var) != 2:
            continue
        function = replace(function)
        try:
            compile(function, '<string>', 'eval')
        except Exception:
            system('cls')
            print('Syntax Error.')
            continue
        break  

    system('cls')

    area = 50
    stepsize = .2

    pool = Pool(2)
    func = partial(pt, function)

    print('Collecting points...')
    t0 = time.time()
    single_verts = pool.map(func, ((x,y) for x in numpy.arange(0, area, stepsize) for y in numpy.arange(0, area, stepsize)))
    t1 = time.time()
    print('Took',t1-t0,'seconds')

    print('Finding bounds...')
    t0 = time.time()
    l = [z for x, y, z in single_verts]
    zbounds = [max(l), min(l)]
    t1 = time.time()
    print('Took',t1-t0,'seconds')

    bounds_matrix = numpy.eye(4, 4, dtype=numpy.float32)
    if zbounds[0]-zbounds[1] > area:
        bounds_matrix = scale([1,1,50/(zbounds[0]-zbounds[1])])

    print('Calculating normals...')
    t0 = time.time()
    gradients = pool.map(func, ((x, y) for i in numpy.arange(0, area, stepsize) for j in numpy.arange(0, area, stepsize) for x, y in [[i-.1, j], [i+.1, j], [i, j-.1], [i, j+.1]]))
    func = partial(nrm, gradients)
    single_normals = pool.map(func, numpy.arange(0, len(gradients), 4))
    if zbounds[0]-zbounds[1] > area:
        inv_bounds_matrix = numpy.linalg.inv(bounds_matrix)
        func = partial(invnrm, inv_bounds_matrix)
        pool.map(func, single_normals)
    t1 = time.time()
    print('Took',t1-t0,'seconds')

    pool.close()
    pool.join()

    print('Calculating mesh...')
    t0 = time.time()
    verts = [vert for x in range(0, int(area/stepsize)-1) for y in range(0, int(area/stepsize)-1) 
                  for vert in [single_verts[x +   y*int(area/stepsize)], single_verts[x+1 +   y*int(area/stepsize)], single_verts[x +   (y+1)*int(area/stepsize)], single_verts[x+1 + y*int(area/stepsize)], single_verts[x + (y+1)*int(area/stepsize)], single_verts[x+1 + (y+1)*int(area/stepsize)]]]
    normals = [normal for x in range(0, int(area/stepsize)-1) for y in range(0, int(area/stepsize)-1) 
                      for normal in [single_normals[x +   y*int(area/stepsize)], single_normals[x+1 +   y*int(area/stepsize)], single_normals[x +   (y+1)*int(area/stepsize)], single_normals[x+1 + y*int(area/stepsize)], single_normals[x + (y+1)*int(area/stepsize)], single_normals[x+1 + (y+1)*int(area/stepsize)]]]
    t1 = time.time()
    print('Took',t1-t0,'seconds')

    centerpt = numpy.array([area/2, area/2, zbounds[1]+(zbounds[0]-zbounds[1])/2, 1],dtype=numpy.float32)

    axis = [(0,0,zbounds[1]), (area,0,zbounds[1]), 
            (0,0,zbounds[1]), (0,area,zbounds[1]), 
            (0,0,zbounds[1]), (0,0,zbounds[0]),
            (area,area,zbounds[1]), (area,0,zbounds[1]),
            (50,50,zbounds[1]), (0,area,zbounds[1]),
            (50,50,zbounds[1]), (area,area,zbounds[0]),
            (0,0,zbounds[0]), (area,0,zbounds[0]),
            (0,0,zbounds[0]), (0,area,zbounds[0]),
            (area,area,zbounds[0]), (area,0,zbounds[0]),
            (area,area,zbounds[0]), (0,area,zbounds[0]),
            (area,0,zbounds[1]), (area,0,zbounds[0]),
            (0,area,zbounds[1]), (0,area,zbounds[0])]

    print('Creating axes...')
    t0 = time.time()
    zlabelpts = [(0, 0, z, z) for z in numpy.arange(zbounds[1], zbounds[0], numpy.power(10,int(numpy.log10(zbounds[0]-zbounds[1]))))]+[(0, 0, zbounds[0], zbounds[0])]
    xlabelpts = [(x, 0, zbounds[1], x) for x in numpy.arange(0, area+1, 10)]
    ylabelpts = [(0, y, zbounds[1], y) for y in numpy.arange(0, area+1, 10)]
    labelpts = numpy.array(xlabelpts+ylabelpts[1:]+zlabelpts[1:],dtype=numpy.float32)

    numverts = len(verts)
    numaxisverts = len(axis)
    verts.extend(axis)

    single_verts = numpy.array(single_verts,dtype=numpy.float32)
    verts = numpy.array(verts,dtype=numpy.float32)
    single_normals = numpy.array(single_normals,dtype=numpy.float32)
    normals = numpy.array(normals,dtype=numpy.float32)

    text = [labelfont.render(str(int(pt[3])), 1, (255, 255, 255)) for pt in labelpts]
    billboard_size = numpy.array([t.get_size() for t in text],dtype=numpy.float32)

    labelpts = numpy.array([(x, y, z) for x, y, z, label in labelpts],dtype=numpy.float32)
    t1 = time.time()
    print('Took',t1-t0,'seconds')

    print('Hit Enter to continue')
    input()

    cam_pos = numpy.array([-40, -45, 75, 1], dtype=numpy.float32)
    cam_dir = numpy.array([1, 1.1, -.9, 1], dtype=numpy.float32)
    up = numpy.array([0, 0, 1, 1], dtype=numpy.float32)
    proj_matrix = perspective(45, screensize[0]/screensize[1], .1, 1000)

    camera_matrix = lookAt(cam_pos[0:3], centerpt[0:3], up[0:3])
    model_matrix = numpy.eye(4, 4, dtype=numpy.float32)
    rot_matrix = numpy.eye(4, 4, dtype=numpy.float32)
    scale_matrix = numpy.eye(4, 4, dtype=numpy.float32)

    lightPos = numpy.array([100, 100, 100],dtype=numpy.float32)

    system('cls')
    print('''
        3D Plot Visualizer

        ___Controls___
        Movement: WASD
        Toggle Screen Lock: L
        Toggle Hide Geometry: H
        Move Highlight Plane: Mouse Scroll
        ''')
    canvas = pygame.display.set_mode(screensize, DOUBLEBUF|OPENGL)
    pygame.display.set_caption('3D Function Visualizer')
    glClearColor(.1, .1, .1, 1)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    VERTEXSHADER = shaders.compileShader(vertex_shader_graph, GL_VERTEX_SHADER)
    FRAGMENTSHADER = shaders.compileShader(fragment_shader_graph, GL_FRAGMENT_SHADER)
    program_graph = shaders.compileProgram(VERTEXSHADER, FRAGMENTSHADER)
    VERTEXSHADER = shaders.compileShader(vertex_shader_billboard, GL_VERTEX_SHADER)
    FRAGMENTSHADER = shaders.compileShader(fragment_shader_billboard, GL_FRAGMENT_SHADER)
    GEOMSHADER = shaders.compileShader(geom_shader_billboard, GL_GEOMETRY_SHADER)
    program_billboard = shaders.compileProgram(VERTEXSHADER, GEOMSHADER, FRAGMENTSHADER)
    if DEBUG_NORMALS:
        VERTEXSHADER = shaders.compileShader(vertex_shader_normals, GL_VERTEX_SHADER)
        FRAGMENTSHADER = shaders.compileShader(fragment_shader_normals, GL_FRAGMENT_SHADER)
        GEOMSHADER = shaders.compileShader(geom_shader_normals, GL_GEOMETRY_SHADER)
        program_normals = shaders.compileProgram(VERTEXSHADER, GEOMSHADER, FRAGMENTSHADER)

    vpos_loc_graph = glGetAttribLocation(program_graph, 'vpos_modelspace')
    vnorm_loc_graph = glGetAttribLocation(program_graph, 'vnorm_modelspace')
    mvp_loc_graph = glGetUniformLocation(program_graph, 'mvp')
    msrb_loc_graph = glGetUniformLocation(program_graph, 'msrb')
    msr_loc_graph = glGetUniformLocation(program_graph, 'msr')
    zcutoff_loc_graph = glGetUniformLocation(program_graph, 'zcutoff')
    cutoff_loc_graph = glGetUniformLocation(program_graph, 'cutoff')
    zbounds_loc_graph = glGetUniformLocation(program_graph, 'zbounds')
    campos_loc_graph = glGetUniformLocation(program_graph, 'camPos')
    lightpos_loc_graph = glGetUniformLocation(program_graph,'lightPos')
    sub_cull_index_graph = glGetSubroutineIndex(program_graph, GL_FRAGMENT_SHADER, 'culled_object')
    sub_border_index_graph = glGetSubroutineIndex(program_graph, GL_FRAGMENT_SHADER, 'border')

    vpos_loc_billboard = glGetAttribLocation(program_billboard, 'vpos_modelspace')
    proj_mat_loc_billboard = glGetUniformLocation(program_billboard, 'proj_matrix')
    tex_loc_billboard = glGetUniformLocation(program_billboard, 'tex')
    mv_loc_billboard = glGetUniformLocation(program_billboard, 'mv')
    size_loc_billboard = glGetAttribLocation(program_billboard, 'billboard_size')
    sub_ind_index_billboard = glGetSubroutineIndex(program_billboard, GL_GEOMETRY_SHADER, 'indices')
    sub_rc_index_billboard = glGetSubroutineIndex(program_billboard, GL_GEOMETRY_SHADER, 'raycast')

    if DEBUG_NORMALS:
        vpos_loc_normals = glGetAttribLocation(program_normals, 'vpos_modelspace')
        vnorm_loc_normals = glGetAttribLocation(program_normals, 'vnorm_modelspace')
        vp_loc_normals = glGetUniformLocation(program_normals, 'vp')
        msr_loc_normals = glGetUniformLocation(program_normals, 'msr')
        bounds_loc_normals = glGetUniformLocation(program_normals, 'bounds_matrix')

    glUseProgram(program_graph)

    vao_graph = glGenVertexArrays(1)
    glBindVertexArray(vao_graph)

    vbo_pos_graph = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_graph)
    glBufferData(GL_ARRAY_BUFFER, verts.size * verts.itemsize, verts, GL_STATIC_DRAW)

    glVertexAttribPointer(vpos_loc_graph, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(vpos_loc_graph)

    vbo_norm_graph = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_norm_graph)
    glBufferData(GL_ARRAY_BUFFER, normals.size*normals.itemsize, normals, GL_STATIC_DRAW)

    glVertexAttribPointer(vnorm_loc_graph, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(vnorm_loc_graph)

    glUniform2fv(zbounds_loc_graph, 1, zbounds)
    glUniform3fv(lightpos_loc_graph, 1, lightPos)

    glUseProgram(program_billboard)

    vao_billboard = glGenVertexArrays(1)
    glBindVertexArray(vao_billboard)

    vbo_pos_billboard = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_billboard)
    glBufferData(GL_ARRAY_BUFFER, (labelpts.size+3)*labelpts.itemsize, labelpts, GL_STATIC_DRAW)

    glVertexAttribPointer(vpos_loc_billboard, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(vpos_loc_billboard)

    vbo_size_billboard = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_size_billboard)
    glBufferData(GL_ARRAY_BUFFER, (billboard_size.size+2)*billboard_size.itemsize, billboard_size, GL_STATIC_DRAW)

    glVertexAttribPointer(size_loc_billboard, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(size_loc_billboard)

    text_texture = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, text_texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glUniform1i(tex_loc_billboard, 0)

    if DEBUG_NORMALS:
        glUseProgram(program_normals)

        vao_normals = glGenVertexArrays(1)
        glBindVertexArray(vao_normals)

        vbo_pos_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_normals)
        glBufferData(GL_ARRAY_BUFFER, single_verts.size*single_verts.itemsize, single_verts, GL_STATIC_DRAW)

        glVertexAttribPointer(vpos_loc_normals, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(vpos_loc_normals)

        vbo_norm_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_norm_normals)
        glBufferData(GL_ARRAY_BUFFER, single_normals.size*single_normals.itemsize, single_normals, GL_STATIC_DRAW)

        glVertexAttribPointer(vnorm_loc_normals, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(vnorm_loc_normals)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    zoom = 0
    regspeed = 10 # movement per second
    sprintspeed = 50
    delta = 0
    mousedelta = [0,0]
    FPS = 60 # desired fps
    clock = pygame.time.Clock()
    mouse_toggle = False
    zcutoff = zbounds[1]-.02*(zbounds[0]-zbounds[1])
    cutoff = False
    arrow = Image.open('arrow.png')
    arrow = arrow.convert('RGBA')
    arrow_data = arrow.getdata()
    arrow_data = [(255, 255, 255, 0) if color[0] == 255 and color[1] == 255 and color[2] == 255 else (0, 255, 0, 255) for color in arrow_data]
    arrow.putdata(arrow_data)
    arrow = arrow.rotate(180)

    def get_frag_depth(pos):
        return glReadPixels(pos[0], pos[1], 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)

    world_pos_text = None
    running = True
    edown = False
    grab = False
    while(running):
        delta = clock.tick(FPS) / 1000
        speed = regspeed
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    mouse_toggle = not mouse_toggle
                    pygame.mouse.set_visible(not mouse_toggle)
                    pygame.mouse.set_pos(screensize[0]/2, screensize[1]/2)
                if event.key == pygame.K_h:
                    cutoff = not cutoff
                if event.key == pygame.K_e:
                    edown = True
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_e:
                    edown = False
            if event.type == pygame.MOUSEMOTION:
                if mouse_toggle:
                    mousedelta = [pygame.mouse.get_pos()[0] - screensize[0]/2, pygame.mouse.get_pos()[1] - screensize[1]/2]
                    cam_dir = rotate(-speed*delta*mousedelta[0], up[0:3])@cam_dir
                    if (angle(up[0:3], cam_dir[0:3]) > .1 and mousedelta[1] < 0) or (angle(-up[0:3], cam_dir[0:3]) > .1 and mousedelta[1] > 0):
                        cam_dir = rotate(speed*delta*mousedelta[1], numpy.cross(up[0:3], cam_dir[0:3]))@cam_dir
                    pygame.mouse.set_pos(screensize[0]/2, screensize[1]/2)
                if grab:
                    mousedelta = [pygame.mouse.get_pos()[0] - screensize[0]/2, pygame.mouse.get_pos()[1] - screensize[1]/2]
                    rot_matrix = rotate(speed*delta*mousedelta[0], up[0:3], centerpt[0:3])@rot_matrix
                    pygame.mouse.set_pos(screensize[0]/2, screensize[1]/2)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    if edown:
                        scale_matrix = scale([1.1, 1.1, 1.1], centerpt[0:3])@scale_matrix
                    else:
                        if zcutoff <= zbounds[0]+(.02*(zbounds[0]-zbounds[1])):
                            zcutoff += (.02*(zbounds[0]-zbounds[1]))
                    
                if event.button == 5:
                    if edown:
                        scale_matrix = scale([.9, .9, .9], centerpt[0:3])@scale_matrix
                    else:
                        if zcutoff >= zbounds[1]-(.02*(zbounds[0]-zbounds[1])):
                            zcutoff -= (.02*(zbounds[0]-zbounds[1]))
                if event.button == 2 and not mouse_toggle:
                    grab = True
                    pygame.mouse.set_pos(screensize[0]/2, screensize[1]/2)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    grab = False
                    
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT]:
            speed = sprintspeed
        if mouse_toggle:
            if keys[pygame.K_w]:
                cam_pos = translate(speed*delta*normalize(cam_dir[0:3]))@cam_pos
            if keys[pygame.K_s]:
                cam_pos = translate(speed*delta*normalize(-cam_dir[0:3]))@cam_pos
            if keys[pygame.K_a]:
                cam_pos = translate(speed*delta*normalize(numpy.cross(up[0:3], cam_dir[0:3])))@cam_pos
            if keys[pygame.K_d]:
                cam_pos = translate(speed*delta*normalize(numpy.cross(cam_dir[0:3], up[0:3])))@cam_pos

        camera_matrix = camera(cam_pos[0:3], cam_dir[0:3], up[0:3])
        msr = scale_matrix@rot_matrix@model_matrix
        msrb = bounds_matrix@msr
        mv = camera_matrix@msrb
        mvp = proj_matrix@mv
        vp = proj_matrix@camera_matrix

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # Begin First Pass
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(program_graph)
        glBindVertexArray(vao_graph)
        glUniformMatrix4fv(mvp_loc_graph, 1, GL_FALSE, mvp.transpose())
        glUniformMatrix4fv(msrb_loc_graph, 1, GL_FALSE, msrb.transpose())
        glUniformMatrix4fv(msr_loc_graph, 1, GL_FALSE, msr.transpose())
        glUniform3fv(campos_loc_graph, 1, cam_pos[0:3])
        glUniform1f(zcutoff_loc_graph, zcutoff)
        glUniform1i(cutoff_loc_graph, cutoff)
    
        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, sub_cull_index_graph)
        # Render the graph
        glDrawArrays(GL_TRIANGLES, 0, int(numverts))
    
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mpos = pygame.mouse.get_pos()
                    d = 2*get_frag_depth([mpos[0],screensize[1]-mpos[1]])-1
                    mpos = [2*(mpos[0]/screensize[0])-1, 1-2*(mpos[1]/screensize[1])]
                    world_pos = numpy.linalg.inv(mvp)@numpy.array([mpos[0], mpos[1], d, 1], dtype=numpy.float32)
                    world_pos = numpy.array([world_pos[0] / world_pos[3], world_pos[1] / world_pos[3], world_pos[2] / world_pos[3]],dtype=numpy.float32)
                    if world_pos[0] >= 0 and world_pos[0] <= area and world_pos[1] >= 0 and world_pos[1] and world_pos[2] >= zbounds[1] and world_pos[2] <= zbounds[0]:
                        world_pos_text = labelfont.render('('+'%.2f'%world_pos[0]+', '+'%.2f'%world_pos[1]+', '+'%.2f'%world_pos[2]+')',1,(255,255,255))
                        rc_billboard_size = numpy.array(world_pos_text.get_size(), dtype=numpy.float32)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_billboard)
                        glBufferSubData(GL_ARRAY_BUFFER, labelpts.size*labelpts.itemsize, world_pos.size*world_pos.itemsize, world_pos)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_size_billboard)
                        glBufferSubData(GL_ARRAY_BUFFER, billboard_size.size*billboard_size.itemsize, rc_billboard_size.size*rc_billboard_size.itemsize, rc_billboard_size)

        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, sub_border_index_graph)
        #Render the bound lines
        glDrawArrays(GL_LINES, int(numverts), int(numaxisverts))
        # End First Pass

        # Begin Second Pass
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(program_billboard)
        glUniformSubroutinesuiv(GL_GEOMETRY_SHADER, 1, sub_ind_index_billboard)
        glBindVertexArray(vao_billboard)
        glUniformMatrix4fv(proj_mat_loc_billboard, 1, GL_FALSE, proj_matrix.transpose())
        glUniformMatrix4fv(mv_loc_billboard, 1, GL_FALSE, mv.transpose())
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, text_texture)
        # Render the number billboards
        for i in range(0, len(text)):
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text[i].get_width(), text[i].get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, pygame.image.tostring(text[i], 'RGBA', True))
            glDrawArrays(GL_POINTS, i, 1)
        # Render the raycast point
        if world_pos_text != None:
            glClear(GL_DEPTH_BUFFER_BIT)
            glBlendFunc(GL_SRC_ALPHA, GL_SRC_COLOR)
            glUniformSubroutinesuiv(GL_GEOMETRY_SHADER, 1, sub_rc_index_billboard)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, world_pos_text.get_width(), world_pos_text.get_height(), 0, GL_RGBA,GL_UNSIGNED_BYTE, pygame.image.tostring(world_pos_text, 'RGBA', True))
            glDrawArrays(GL_POINTS, len(text), 1)
        # End Second Pass

        if DEBUG_NORMALS:
            # Begin Third Pass
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(program_normals)
            glBindVertexArray(vao_normals)
            glUniformMatrix4fv(vp_loc_normals, 1, GL_FALSE, vp.transpose())
            glUniformMatrix4fv(msr_loc_normals, 1, GL_FALSE, msr.transpose())
            glUniformMatrix4fv(bounds_loc_normals, 1, GL_FALSE, bounds_matrix.transpose())
            glDrawArrays(GL_POINTS, 0, single_verts.size)
            # End Third Pass

        # Render compass
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        center_pos = mv@centerpt
        rot_zvec = numpy.array([0,0,-1],dtype=numpy.float32)
        view_angle = angle(center_pos[0:3], rot_zvec)
        render_compass = dgr(view_angle) > (24+2600/numpy.linalg.norm(center_pos))

        if render_compass:
            rot_upvec = numpy.array([0,1],dtype=numpy.float32)
            rot_angle = numpy.arctan2(rot_upvec[0]*center_pos[1] - center_pos[0]*rot_upvec[1], rot_upvec@center_pos[0:2])
            rot_arrow = arrow.rotate(-dgr(rot_angle)).tobytes()
            glRasterPos2f(-(2*arrow.size[0]/screensize[0])/2,-(2*arrow.size[0]/screensize[0])/2)
            glDrawPixels(arrow.width, arrow.height, GL_RGBA, GL_UNSIGNED_BYTE, rot_arrow)
            glRasterPos2f(0,0)
    
        glBindVertexArray(0)
        glUseProgram(0)
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
import numpy as np
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as mpl3D
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import widgets
from matplotlib import pyplot as plt
from ipywidgets import widgets, interact

from calib3d import Point2D, Point3D

import sys
np.set_printoptions(precision=5, suppress=True)

# Set colormap to 'gray'
mpl.rc('image', cmap='gray')
WORLD_LIMIT = 9.0



class Cube():
    def __init__(self, offset=(0,0,0), size=1, color="green"):
        self.color = color
        x,y,z = offset
        s = size
        # cube bottom base
        self.A = Point3D(x  , y  , z  )
        self.B = Point3D(x+s, y  , z  )
        self.C = Point3D(x+s, y+s, z  )
        self.D = Point3D(x  , y+s, z  )

        # cube top base
        self.E = Point3D(x  , y  , z+s)
        self.F = Point3D(x+s, y  , z+s)
        self.G = Point3D(x+s, y+s, z+s)
        self.H = Point3D(x  , y+s, z+s)

    @property
    def bottom_base(self):
        return [self.A, self.B, self.C, self.D]

    @property
    def top_base(self):
        return [self.E, self.F, self.G, self.H]

    @property
    def corners(self):
        return self.bottom_base + self.top_base

    @property
    def edges(self):
        bottom_base_edges = [[X, Y] for X, Y in zip(self.bottom_base[0:], self.bottom_base[1:]+[self.A])]
        top_base_edges = [[X, Y] for X, Y in zip(self.top_base[0:], self.top_base[1:]+[self.E])]
        vertical_edges = [[X, Y] for X, Y in zip(self.top_base, self.bottom_base)]
        return bottom_base_edges + top_base_edges + vertical_edges


def create_3D_world(fig, subplot=111, world_limit=WORLD_LIMIT, z_down=False, linewidth=4):
    ax = fig.add_subplot(subplot , projection="3d")
    ax.set_aspect('auto')
    if world_limit is not None:
        ax.set_xlim3d([0, world_limit])
        ax.set_ylim3d([0, world_limit])
        ax.set_zlim3d([-world_limit, 0] if z_down else [0, world_limit])
    L = 10**np.floor(np.log10(world_limit)) # arrow length
    o = 10**np.floor(np.log10(world_limit)-1)  # offset to display text
    if z_down:
        ax.invert_yaxis()
        ax.invert_zaxis()
    ax.plot([0, L], [0, 0], [0, 0], color="red", linewidth=linewidth)
    ax.plot([0, 0], [0, L], [0, 0], color="red", linewidth=linewidth)
    ax.plot([0, 0], [0, 0], [0, L], color="red", linewidth=linewidth)
    ax.text(L, -o, -o, "x", color="red")
    ax.text(-o, L, -o, "y", color="red")
    ax.text(o, -o, L, "z", color="red")
    return ax

def create_2D_world(fig, subplot=111, world_limit=1000, linewidth=4):
    ax = fig.add_subplot(subplot)
    L = 10**np.floor(np.log10(world_limit))  # arrow length
    o = 1  # offset to display text
    ax.plot([0, L], [0, 0], color="blue", linewidth=linewidth)
    ax.plot([0, 0], [0, L], color="blue", linewidth=linewidth)
    ax.text(L, -o, "x", color="blue")
    ax.text(-o, L, "y", color="blue")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return ax


def build_R2(a,b,c):
    return np.array([[np.cos(c),np.sin(c),0],[-np.sin(c),np.cos(c),0],[0,0,1]])@np.array([[np.cos(b),0,-np.sin(b)],[0,1,0],[np.sin(b),0,np.cos(b)]])@np.array([[1,0,0],[0,np.cos(a),np.sin(a)],[0,-np.sin(a),np.cos(a)]])

def build_T2(a,b,c,d):
    return -d@np.array([[a],[b],[c]])

def draw_camera_3D(ax3D, calib, L=0.5, color="blue", name="camera", linewidth=1, skip_fov=False):
    C = -np.transpose(calib.R)@calib.T
    x, y, z = C.flatten()
    ax3D.text(x, y, z, name)
    ax3D.plot([x, x], [y, y], [0, z], linestyle="--", color=color, linewidth=0.5)
    ax3D.plot([0, x], [y, y], [0, 0], linestyle="--", color=color, linewidth=0.5)
    ax3D.plot([x, x], [0, y], [0, 0], linestyle="--", color=color, linewidth=0.5)
    ax3D.scatter(x, y, z, color=color)

    vector = calib.T-np.array([[L, 0, 0]]).T
    x1, y1, z1 = -(np.transpose(calib.R)@vector).flatten()
    ax3D.plot([x, x1], [y, y1], [z, z1], color=color, linewidth=linewidth)
    ax3D.text(x1,y1,z1, "x", color=color)

    vector = calib.T-np.array([[0, L, 0]]).T
    x1, y1, z1 = -(np.transpose(calib.R)@vector).flatten()
    ax3D.plot([x, x1], [y, y1], [z, z1], color=color, linewidth=linewidth)
    ax3D.text(x1,y1,z1, "y", color=color)

    vector = calib.T-np.array([[0, 0, L]]).T
    x1, y1, z1 = -(np.transpose(calib.R)@vector).flatten()
    ax3D.plot([x, x1], [y, y1], [z, z1], color=color, linewidth=linewidth)
    ax3D.text(x1,y1,z1, "z", color=color)
    
    if skip_fov:
        return
    
    w, h = calib.width, calib.height
    M = np.linalg.pinv(np.vstack((np.hstack((calib.R, calib.T)), np.array([[0,0,0,1]]))))
    proj = lambda x, y: Point3D(M @ Point3D((calib.Kinv @ Point2D(x, y).H)*L/2).H)
    C = -calib.R.T@calib.T
    vtx = [np.array([C, proj(x1, y1), proj(x2, y2), C]) for x1, y1, x2, y2 in zip(
        [0, 0, w, w, 0], [0, h, h, 0, 0], [0, w, w, 0, 0], [h, h, 0, 0, 0])]
    facecolors = "lightblue" if color == "blue" else "lightgray"
    for v in vtx:
        ax3D.add_collection3d(mpl3D.art3d.Poly3DCollection([v.T[0].T], facecolors=facecolors, edgecolors="black", linewidths=linewidth))

def draw_camera_2D(ax2D, calib, color="blue", linewidth=4):
    w, h = calib.width, calib.height

    # draw camera view
    ax2D.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color=color)
    ax2D.text(0, -0.3, "Camera view", color=color)
    ax2D.set_xlim([-w/10, 11*w/10])
    ax2D.set_ylim([11*h/10, -h/10])

    # draw origin of the world (depends on calib)
    res = 10
    O = calib.project_3D_to_2D(Point3D(0,0,0))
    for e, name in zip([Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)], ["x", "y", "z"]):
        points = np.stack([np.linspace(0, e.x, res), np.linspace(0, e.y, res), np.linspace(0, e.z, res)])
        for i in range(points.shape[1]-1):
            p1 = calib.project_3D_to_2D(Point3D(points[:,i]))
            p2 = calib.project_3D_to_2D(Point3D(points[:,i+1]))
            ax2D.plot([p1.x, p2.x], [p1.y, p2.y], color="red", linewidth=linewidth)
        point2D = calib.project_3D_to_2D(e)
        ax2D.text(point2D.x, point2D.y, name, color="red")

def draw_court_3D(ax3D, color="blue"):
    # draw court rectangle
    ax3D.plot([0, 2800, 2800, 0, 0], [0, 0, 1500, 1500, 0], [0, 0, 0, 0, 0], color=color)
    # draw middle line
    ax3D.plot([1400, 1400], [0, 1500], [0, 0], color=color)
    # draw center circle
    T = np.linspace(0,2*np.pi,100)
    ax3D.plot([1400+180*np.sin(t) for t in T], [750+180*np.cos(t) for t in T], [0 for t in T], color=color)
    # draw basket rectangles
    ax3D.plot([0,580,580,0,0],[750-490/2, 750-490/2, 750+490/2, 750+490/2, 750-490/2], [0,0,0,0,0], color=color)
    ax3D.plot([2800, 2800-580, 2800-580, 2800, 2800], [750-490/2, 750-490/2, 750+490/2, 750+490/2, 750-490/2], [0,0,0,0,0], color=color)
    ax3D.plot([580+180*np.sin(t) for t in T], [750+180*np.cos(t) for t in T], [0 for t in T], color=color)
    ax3D.plot([2800-580+180*np.sin(t) for t in T], [750+180*np.cos(t) for t in T], [0 for t in T], color=color)


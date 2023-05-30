import pygame,random
import numpy as np
from math import *
import avatar_point as ap

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
G_38 = (38,38,38)
G_37 = (36,36,38)
G_35 = (34,34,36)
BLUE = (0,100,255)
BLUEE = (0,100,200)

SCALE = 100
WIDTH, HEIGHT = 600, 680

circle_pos = [WIDTH/2, HEIGHT/2]

projection_matrix = np.matrix([
	[1, 0, 0],
	[0, 1, 0],
])

PLUS = 0

face = ap.face
chin = ap.chin
horn = ap.horn
right_horn = ap.right_horn
left_horn = ap.left_horn
top_head = ap.top_head
left_head = ap.left_head
right_head = ap.right_head
left_eye = ap.left_eye_happy# normal eye
right_eye = ap.right_eye_happy
mouth = ap.mouth

class Particles:
    def __init__(self):
        self.particles = []
    
    def show(self,screen):
        if self.particles:
            for particle in self.particles:
                particle[0][1] += particle[2][1] # mengubah posisi y
                particle[0][0] += particle[2][0] # mengubah posisi x
                particle[1] -= 0.15
                particle[3] -= 0.18
                particle[2][1] += 0.1
                pygame.draw.circle(screen,(148,245,255),particle[0],int(particle[3]))
                pygame.draw.circle(screen,(255,255,255),particle[0],int(particle[1]))
                if particle [1]<=0:
                    self.particles.remove(particle)
        
    def add_particles(self):
        pos_x = WIDTH/2 + random.randint(-40,40)
        pos_y = HEIGHT/2 + random.randint(0,20)
        radius = random.randint(10,20)
        radius2 = radius + 4
        direction_y = 0
        direction_x = 0
        particle_circle = [[pos_x,pos_y],radius,[direction_x,direction_y],radius2]
        self.particles.append(particle_circle)
		
def rotate_z(angle):
	rotation_z = np.matrix([
		[cos(angle), -sin(angle), 0],
		[sin(angle), cos(angle), 0],
		[0, 0, 1],
	])
	return rotation_z

def rotate_y(angle):
	rotation_y = np.matrix([
		[cos(angle), 0, sin(angle)],
		[0, 1, 0],
		[-sin(angle), 0, cos(angle)],
	])
	return rotation_y

def rotate_x(angle):
	rotation_x = np.matrix([
		[1, 0, 0],
		[0, cos(angle), -sin(angle)],
		[0, sin(angle), cos(angle)],
	])
	return rotation_x
	
def connect_point(screen,listpoint,color,x_angle,y_angle,z_angle):
	points = []
	for i in listpoint:
		points.append(np.array(i))#Convert List To Array/Matrix
	distance = 3
	projected_points = []
	i =0
	for point in points:
		rotate_2d = np.dot(rotate_z(z_angle), point.reshape((3, 1)))
		rotate_2d = np.dot(rotate_y(y_angle), rotate_2d)
		rotate_2d = np.dot(rotate_x(x_angle), rotate_2d)

		z = 1/(distance-rotate_2d[2][0])

		projected2d = np.dot(projection_matrix, rotate_2d)
		x = int(projected2d[0][0] * SCALE) + circle_pos[0]
		y = int(projected2d[1][0] * SCALE) + circle_pos[1]
		projected_points.append([x,y])
		# pygame.draw.circle(screen, RED, (x,y), 4)
		i+=1
	pygame.draw.polygon(screen,color,projected_points)


def update(screen,x_angle,y_angle,z_angle):
	connect_point(screen,left_head,G_37,x_angle,y_angle,z_angle)
	connect_point(screen,right_head,G_37,x_angle,y_angle,z_angle)
	connect_point(screen,top_head,G_37,x_angle,y_angle,z_angle)
	connect_point(screen,chin,G_38,x_angle,y_angle,z_angle)
	connect_point(screen,face,G_38,x_angle,y_angle,z_angle)
	connect_point(screen,horn,BLUE,x_angle,y_angle,z_angle)
	connect_point(screen,left_horn,BLUEE,x_angle,y_angle,z_angle)
	connect_point(screen,right_horn,BLUEE,x_angle,y_angle,z_angle)
	connect_point(screen,left_eye,WHITE,x_angle,y_angle,z_angle)
	connect_point(screen,right_eye,WHITE,x_angle,y_angle,z_angle)
	connect_point(screen,mouth,WHITE,x_angle,y_angle,z_angle)
	
x_angle=y_angle=z_angle=0



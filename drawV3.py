from numpy.lib.polynomial import poly
import pygame
import math
import numpy as np
from main import *
import poly_3d
import avatar_point as ap
import cv2
from argparse import ArgumentParser
# from object_3d import *
# This File Is Conneted With face.py

landmarks = []
position = [0,0,0]
pos_histories = [[0], [0], [0]]
X_ROT_SCALE = 3
rot_histories = [[0], [0], [0]]
AVERAGE_DISTANCE = 100
hoy = [0,4,5,6,8,9,15,17,18,19,20,37,40,41,47,49,50,51,52,67]
running = False

def drawInPygame(screen,list_point,font):
    py = sum(pos_histories[1]) / len(pos_histories[1])
    particles = poly_3d.Particles()
    keys=pygame.key.get_pressed()
    for list in list_point:
        
        points = list['points']
        rotation = list['rotation']
        particles.add_particles()
        left_eyebrow = points[17:22]
        right_eyebrow = points[22:27]
        nose = points[27:36]
        left_eye = points[36:42]
        right_eye = points[42:48]
        landmarks.append(points)
        particles.add_particles()
        
        if len(rotation)!= 0:
            screen.fill((0 ,177 , 64))
            eye_dis = math.sqrt((right_eye[2][1] - left_eye[1][1]) ** 2 + (right_eye[2][0] - left_eye[1][0]) ** 2)
            rot_z = rotation[2]
            text = font.render('rotz: ' + str(round(math.degrees(rot_z), 2)), screen, (255, 250,255))
            screen.blit(text, (10 , 10 ))
            if round(math.degrees(rot_z), 2) > 12:
                poly_3d.PLUS = 40
            elif round(math.degrees(rot_z), 2) < -12:
                poly_3d.PLUS = -40
            else:
                poly_3d.PLUS = 0

            left_eyebrow_length = math.sqrt((left_eyebrow[-1][0] - left_eyebrow[0][0]) ** 2 + (left_eyebrow[-1][1] - left_eyebrow[0][1]) ** 2) / eye_dis
            right_eyebrow_length = math.sqrt((right_eyebrow[-1][0] - right_eyebrow[0][0]) ** 2 + (right_eyebrow[-1][1] - right_eyebrow[0][1]) ** 2) / eye_dis
            eyebrow_implications = left_eyebrow_length - right_eyebrow_length
            rot_y = rotation[1]
            text = font.render('roty: ' + str(round(math.degrees(rot_y), 2)), screen, (255, 250,255))
            screen.blit(text, (10 , 30 ))

            nose_length = nose[6][1] - nose[3][1]
            position = [nose[6][0] / 100 - 0.5, nose[6][1] - 0.5, eye_dis / AVERAGE_DISTANCE]
            rot_x = rotation[0]
            text = font.render('rotx: ' + str(round(math.degrees(rot_x), 2)), screen, (255,250, 255))
            screen.blit(text, (10 , 50 ))
            for i, rot in enumerate([-rot_x, -rot_y, rot_z]):
                rot_histories[i].append(rot)
                rot_histories[i] = rot_histories[i][-20:]

            poly_3d.x_angle = max(sum(rot_histories[0]) / len(rot_histories[0]), - math.radians(20))
            poly_3d.y_angle = sum(rot_histories[1]) / len(rot_histories[1])
            poly_3d.z_angle = sum(rot_histories[2]) / len(rot_histories[2])
            
            for i, p in enumerate(position):
                pos_histories[i].append(position[i])
                pos_histories[i] = pos_histories[i][-10:]

            mouth_height = math.sqrt((points[4][0] - points[5][0]) ** 2 + (points[4][1] - points[5][1]) ** 2) / eye_dis
            mouth_width = math.sqrt((points[15][0] - points[47][0]) ** 2 + (points[15][1] - points[47][1]) ** 2) / eye_dis -0.479

            poly_3d.mouth[0][1] = poly_3d.mouth[1][1] - mouth_height * 0.275 + 0.01
            poly_3d.mouth[2][1] = poly_3d.mouth[1][1] + mouth_height * 0.7 + 0.033
            poly_3d.mouth[1][0] = poly_3d.mouth[0][0] - 0.175 - mouth_width * 0.5
            poly_3d.mouth[3][0] = poly_3d.mouth[0][0] + 0.175 + mouth_width * 0.5


            particles.show(screen)
            poly_3d.update(screen,poly_3d.x_angle,poly_3d.y_angle,poly_3d.z_angle)
            pygame.display.update()
            

def runPygame(screen,clock,font):
    running = True

    list = main(True)
    # list_point = initOpencv(running)
    while running:
        drawInPygame(screen,list,font)
        clock.tick(60)
        pygame.display.update()
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

def initializePygame():
    pygame.init()
    pygame.font.init()
    my_font = pygame.font.Font('freesansbold.ttf', 22)
    pygame.display.set_caption('game base')

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((600, 680),0,32)
    runPygame(screen,clock,my_font)

initializePygame()
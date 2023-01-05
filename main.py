"""
Main program to run the detection
"""

from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np

# for TCP connection with unity
import socket
from collections import deque
from platform import system

# face detection and facial landmark
from facial_landmark import FaceMeshDetector

# pose estimation and stablization
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

face_points = [127,93,132,58,172,136,150,176,152,400,365,379,397,288,361,323,356
,46,53,52,65,55,285,295,282,283,276,168,197,5,4,98,97,2,326,327,33,160,158,133,144,153
,362,385,387,263,373,380,61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178]

def main(debug=True):

    list_points = []
    list_points_last = []
    cap = cv2.VideoCapture(0)

    # Facemesh
    detector = FaceMeshDetector()

    # get a sample frame for pose estimation img
    success, img = cap.read()

    # Emotion List
    emotions = ["Angry","Happy","Neutral","Sad","Surprise"]

    # Pose estimation related
    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for eyes
    eyes_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for mouth_dist
    mouth_dist_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    )



    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # first and two steps
        img_facemesh, faces, emotion_id = detector.findFaceMesh(img)

        # flip the input image so that it matches the facemesh stuff
        img = cv2.flip(img, 1)
        rot = []
        # kern_25 = np.ones((35,35),np.float32)/625.0
        # kern_output = cv2.filter2D(img,-1,kern_25)

        # if there is any face detected
        if faces:
            list_points = []
            # only get the first face
            for id,lm in enumerate(image_points):
                image_points[id, 0] = faces[0][id][0]
                image_points[id, 1] = faces[0][id][1]
                for num in face_points:
                    if id == num:
                        list_points.append([lm[0],lm[1]])

            # The third step: pose estimation
            # pose: [[rvec], [tvec]]
            pose = pose_estimator.solve_pose_by_all_points(image_points)
            # print(image_points)
            x_left, y_left, x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(img, faces[0], Eyes.LEFT)
            x_right, y_right, x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(img, faces[0], Eyes.RIGHT)

            ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

            pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mouth_distance = FacialFeatures.mouth_distance(image_points)

            # print("left eye: %d, %d, %.2f, %.2f" % (x_left, y_left, x_ratio_left, y_ratio_left))
            # print("right eye: %d, %d, %.2f, %.2f" % (x_right, y_right, x_ratio_right, y_ratio_right))

            # print("rvec (y) = (%f): " % (pose[0][1]))
            # print("rvec (x, y, z) = (%f, %f, %f): " % (pose[0][0], pose[0][1], pose[0][2]))
            # print("tvec (x, y, z) = (%f, %f, %f): " % (pose[1][0], pose[1][1], pose[1][2]))

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()

            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])

            steady_pose = np.reshape(steady_pose, (-1, 3))

            # stabilize the eyes value
            steady_pose_eye = []
            for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                ps_stb.update([value])
                steady_pose_eye.append(ps_stb.state[0])

            mouth_dist_stabilizer.update([mouth_distance])
            steady_mouth_dist = mouth_dist_stabilizer.state[0]

            # print("rvec (x, y, z) = (%f, %f, %f): " % (steady_pose[0][0], steady_pose[0][1], steady_pose[0][2]))
            # print("tvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[1][0], steady_pose[1][1], steady_pose[1][2]))

            # calculate the roll/ pitch/ yaw
            # roll: +ve when the axis pointing upward
            # pitch: +ve when we look upward
            # yaw: +ve when we look left
            roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90) # Rot z
            pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)# Rot x
            yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90) # Rot y
            
            rot_x = steady_pose[0][0]+3
            rot_y = steady_pose[0][2]
            rot_z = steady_pose[0][1]
            
            x_ = rot_x*2
            rot_x -= x_
            y_ = rot_y*2
            rot_y -= y_
            z_ = rot_z*2
            rot_z -= z_

            rot = [rot_x,rot_y,rot_z]

            # pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=(255, 128, 128))

            # pose_estimator.draw_axis(img, pose[0], pose[1])

            pose_estimator.draw_axes(img, steady_pose[0], steady_pose[1])
            pose_estimator.draw_mouth(img,list_points)
            pose_estimator.draw_info_text(img,emotions[emotion_id])
            

        else:
            # reset our pose estimator
            pose_estimator = PoseEstimator((img_facemesh.shape[0], img_facemesh.shape[1]))

        if debug:
            cv2.imshow('Facial landmark', img)
            # cv2.imshow('Facial landmark', img_facemesh)


        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        yield {'points':list_points,'rotation':rot,"emotion":emotion_id}



if __name__ == "__main__":

    # demo code
    main()

"""
Taken from https://github.com/lincolnhard/head-pose-estimation
Lightly modified to expose drawing helpers
"""
import cv2
import dlib
import numpy as np
from imutils import face_utils

# Assume moderate lens distortion
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

cube_points = np.float32([[10.0, 10.0, 10.0],
                          [10.0, 10.0, -10.0],
                          [10.0, -10.0, -10.0],
                          [10.0, -10.0, 10.0],
                          [-10.0, 10.0, 10.0],
                          [-10.0, 10.0, -10.0],
                          [-10.0, -10.0, -10.0],
                          [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_cam_matrix(img_size):
    focal_length = img_size[1]
    camera_center = (img_size[1] / 2, img_size[0] / 2)
    return np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype=np.float32)


def get_head_pose(shape, cam_matrix):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return rotation_vec, translation_vec, euler_angle

def draw_head_pose_features(frame, rotation_vec, translation_vec, euler_angle, shape, cam_matrix):
    draw_head_pose_box(frame, rotation_vec, translation_vec, cam_matrix)
    draw_axis(frame, rotation_vec, translation_vec, cam_matrix)
    draw_landmarks(frame, shape)
    draw_angle(frame, euler_angle)

def draw_head_pose_box(frame, rotation_vec, translation_vec, cam_matrix):

    points_2d, _ = cv2.projectPoints(cube_points, rotation_vec, translation_vec, cam_matrix,
                                     dist_coeffs)

    points_2d = tuple(map(tuple, points_2d.reshape(8, 2)))

    for start, end in line_pairs:
        cv2.line(frame, points_2d[start], points_2d[end], (0, 255, 255), 20)

def draw_angle(frame, euler_angle):
    cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=20)
    cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=20)
    cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=20)

def draw_axis(img, r, t, camera_matrix):
    points = np.float32(
        [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

    axisPoints, _ = cv2.projectPoints(
        points, r, t, camera_matrix, dist_coeffs)

    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[0].ravel()), (255, 0, 0), 20)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[1].ravel()), (0, 255, 0), 20)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[2].ravel()), (0, 0, 255), 20)


def draw_landmarks(frame, shape):
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    while cap.isOpened():
        ret, frame = cap.read()
        cam_matrix = get_cam_matrix(frame.shape())
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)
                rotation, translation, euler_angle = get_head_pose(shape, cam_matrix)
                draw_head_pose_box(frame, shape, rotation, translation, euler_angle, cam_matrix)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()

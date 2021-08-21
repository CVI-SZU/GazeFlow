import cv2
import numpy as np
import scipy.io as sio


def get_annotation_dict_in_day(annotation_fp):
    annotation_dict = {}
    with open(annotation_fp, 'r') as f:
        annotation = f.readlines()
    for idx, line in enumerate(annotation):
        line = line.strip()
        line = line.split(' ')
        img_name = f'{idx + 1:04}.jpg'
        annotation_dict[img_name] = line
    return annotation_dict


def get_cameraMatrix(calibration_fp):
    cameraCalib = sio.loadmat(calibration_fp)
    return cameraCalib['cameraMatrix']


def get_headpose(annotation):
    headpose_hr = np.array(annotation[29:32], np.float32)
    headpose_ht = np.array(annotation[32:35], np.float32)
    headpose_ht = headpose_ht.reshape(3, 1)
    hR = cv2.Rodrigues(headpose_hr)[0]
    return headpose_hr, headpose_ht, hR


def get_facemodel(facemodel_path='6 points-based face model.mat'):
    face_model = sio.loadmat(facemodel_path)
    return face_model['model']


def calculate_Fc(face_model, headpose_ht, hR):
    Fc = np.matmul(hR, face_model)
    Fc = Fc + headpose_ht
    return Fc


def calculate_eye_position(Fc):
    eye_center_right = 0.5 * (Fc[:, 0] + Fc[:, 1])
    eye_center_left = 0.5 * (Fc[:, 2] + Fc[:, 3])
    return eye_center_right, eye_center_left


def get_gaze_target(annotation):
    gaze_target = np.asarray([float(i) for i in annotation[26:29]])
    return gaze_target


def get_theta_phi(gaze, headpose):
    # convert the gaze direction in the camera cooridnate system to the angle 
    # in the polar coordinate system
    gaze_theta = np.arcsin((-1) * gaze[1])  # vertical gaze angle
    gaze_phi = np.arctan2((-1) * gaze[0], (-1) * gaze[2])  # horizontal gaze angle

    # save as above, conver head pose to the polar coordinate system
    M = cv2.Rodrigues(headpose)[0]
    Zv = M[:, 2]
    headpose_theta = np.arcsin(Zv[1])  # vertical head pose angle
    headpose_phi = np.arctan2(Zv[0], Zv[2])  # horizontal head pose angle
    #     return gaze_theta, gaze_phi, headpose_theta, headpose_phi
    return np.array([gaze_theta, gaze_phi]), np.array([headpose_theta, headpose_phi])


def read_img(img_file, CLAHE=True, clahe=None):
    img_ori = cv2.imread(img_file)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

    if CLAHE:
        if not clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # hist equalization
        b, g, r = cv2.split(img_ori)
        b_eq_hist = clahe.apply(b)
        g_eq_hist = clahe.apply(g)
        r_eq_hist = clahe.apply(r)
        img = cv2.merge((b_eq_hist, g_eq_hist, r_eq_hist))
    else:
        img = img_ori
    return img


def normalizeImg(inputImg, target_3D, hR, gc, roiSize, cameraMatrix):
    '''
        inorder to overcome the difference of the camera internal parameters 
    and the the distance from face center to the camera optical
    '''
    # new virtual camera
    focal_new = 960.0
    #    distance_new=300.0 for 448*448
    distance_new = 600.0

    distance = np.linalg.norm(target_3D)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0.0, roiSize[0] / 2], [0.0, focal_new, roiSize[1] / 2], [0.0, 0.0, 1.0]],
                       np.float32)
    scaleMat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]], np.float32)
    hRx = hR[:, 0]
    forward = target_3D / distance
    down = np.cross(forward, hRx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    rotMat = np.c_[right, down, forward]
    rotMat = rotMat.transpose()
    warpMat = np.dot(np.dot(cam_new, scaleMat), np.dot(rotMat, np.linalg.inv(cameraMatrix)))
    img_warped = cv2.warpPerspective(inputImg, warpMat, roiSize)
    # img_warped = np.transpose(img_warped, (1,0,2))

    # rotation normalization
    cnvMat = np.dot(scaleMat, rotMat)
    hRnew = np.dot(cnvMat, hR)
    hrnew = cv2.Rodrigues(hRnew)[0]
    htnew = np.dot(cnvMat, target_3D)

    # gaze vector normalization
    gcnew = np.dot(cnvMat, gc)
    gvnew = gcnew - htnew
    gvnew = gvnew / np.linalg.norm(gvnew)

    return img_warped, hrnew, gvnew


def get_eyes_image(img_fp, annotation, cameraMatrix, face_model, eye_image_width=60,
                   eye_image_height=36, CLAHE=True, clahe=None):
    img = read_img(img_fp, CLAHE, clahe)
    headpose_hr, headpose_ht, hR = get_headpose(annotation)
    Fc = calculate_Fc(face_model, headpose_ht, hR)
    eye_center_right, eye_center_left = calculate_eye_position(Fc)
    gaze_target = get_gaze_target(annotation)

    right_eye_img, right_headpose, right_gaze = normalizeImg(img, eye_center_right, hR, gaze_target,
                                                             (eye_image_width, eye_image_height), cameraMatrix)
    left_eye_img, left_headpose, left_gaze = normalizeImg(img, eye_center_left, hR, gaze_target,
                                                          (eye_image_width, eye_image_height), cameraMatrix)

    # convert to the polar coordinate system
    right_gaze, right_headpose = get_theta_phi(right_gaze, right_headpose)
    left_gaze, left_headpose = get_theta_phi(left_gaze, left_headpose)

    return {
        'right_eye': {'img': right_eye_img, 'headpose': right_headpose, 'gaze': right_gaze},
        'left_eye': {'img': left_eye_img, 'headpose': left_headpose, 'gaze': left_gaze},
    }

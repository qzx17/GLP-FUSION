import os
import math
import random

import cv2
import dlib
import torch
import face_recognition

import numpy as np
import pandas as pd
import scipy as sp

from imutils import face_utils
from concurrent.futures import ThreadPoolExecutor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
PATH = 'DAiSEE'
DATASET = 'DAiSEE'

# PATH = 'EngageNet'
# DATASET = 'EngageNet'

FPS = 30
BATCH_SIZE = 1
VIDEO_LENGTH = 10
FRAME_INTERVAL = 2
GLOBAL_BATCH_SIZE = 92


def get_subdirectories(path, sort=False):
    """Get a list of sub-directories in parent directory 'path'."""
    sub_directories = [os.path.join(path, subdir) for subdir in os.listdir(path) if
                       os.path.isdir(os.path.join(path, subdir))]

    if sort:
        sub_directories.sort()
    else:
        random.shuffle(sub_directories)

    return sub_directories


def get_videos(path, sort=False):
    """Get a list of video paths from a directory 'path'."""
    videos = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(('.mp4', '.avi', '.mov'))]

    if sort:
        videos.sort()
    else:
        random.shuffle(videos)

    return videos


# def get_video_paths(path, sort=False):
#     """Get all video paths in a dataset."""
#     videos = []
#
#     subject_list = get_subdirectories(path)
#
#     for subject in subject_list:
#         subject_subdir_list = get_subdirectories(subject, sort)
#
#         for subdir in subject_subdir_list:
#             subdir_video_paths = get_videos(subdir, sort)
#             videos.extend(subdir_video_paths)
#
#     return videos


def get_video_paths(path, sort=False):
    """Get a list of video paths from a directory 'path'."""
    # 确保路径存在且是目录
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"The specified path is not a directory: {path}")

    # 获取所有视频文件的路径
    videos = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(('.mp4', '.avi', '.mov'))]

    # 根据需要对视频列表进行排序或随机打乱
    if sort:
        videos.sort()
    else:
        random.shuffle(videos)

    return videos


def get_frames(subject_videos, frame_interval=FRAME_INTERVAL, resize_to=None):
    """Get frames from a list of 'subject_videos' paths."""
    frames_subject = []

    for subject_video in subject_videos:
        video_capture = cv2.VideoCapture(subject_video)

        if not video_capture.isOpened():
            print(f"Error opening video file {subject_video}")
            continue

        count = 0
        frames = []

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            count += 1
            if count % frame_interval == 0:
                if resize_to:
                    frame = cv2.resize(frame, resize_to)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        video_capture.release()

        if frames:
            frames_required = (FPS * VIDEO_LENGTH) // frame_interval
            frames = frames[:frames_required]
            while len(frames) < frames_required:
                frames.append(frames[-1])
            frames_subject.append(frames)
        else:
            print(f"No frames extracted from video file {subject_video}")

    return frames_subject if frames_subject else [[]]


def get_labels(paths):
    """Get labels for boredom, engagement, confusion, frustration."""
    data = pd.read_csv("D:\\datasets\\DAiSEE_frame_ALL\\AllLabels.csv")
    tails = [os.path.split(path)[1] for path in paths]
    filtered_data = data[data['ClipID'].isin(tails)]
    engagement_data = filtered_data[['Engagement']]

    id_data = filtered_data[['ClipID']]
    return engagement_data.values,id_data.values


iterator = 0


def load_data(path, dataset, batch_size):
    """Load random videos from 'path'."""
    global iterator
    path = os.path.join(path, dataset)
    paths = get_video_paths(path, sort=True)
    X = get_frames(paths[iterator:batch_size + iterator], FRAME_INTERVAL)
    Y, ID = get_labels(paths[iterator:batch_size+iterator])

    iterator += batch_size
    if iterator >= len(paths):
        iterator = 0

    return X, Y, ID


shape_predictor_path = "/Openface-extractor/models/dlib/shape_predictor_68_face_landmarks.dat"
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    print("Successfully loaded")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()


def eye_aspect_ratio_torch(eye):
    A = torch.norm(eye[1] - eye[5])
    B = torch.norm(eye[2] - eye[4])
    C = torch.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_head_pose_torch(shape, size, device='cpu'):
    model_points = torch.tensor([
        [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
    ], dtype=torch.float64).to(device)

    image_points = shape[[30, 8, 36, 45, 48, 54], :]

    if image_points.shape[0] < 4:
        raise ValueError("Not enough points to perform solvePnP")

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = torch.tensor([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=torch.float64).to(device)

    dist_coeffs = torch.zeros((4, 1), dtype=torch.float64).to(device)

    image_points_np = image_points.cpu().numpy()
    model_points_np = model_points.cpu().numpy()
    camera_matrix_np = camera_matrix.cpu().numpy()
    dist_coeffs_np = dist_coeffs.cpu().numpy()

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points_np, image_points_np, camera_matrix_np, dist_coeffs_np
    )

    if not success:
        raise ValueError("solvePnP failed to find a solution")

    rotation_vector = torch.tensor(rotation_vector, dtype=torch.float64).to(device)
    translation_vector = torch.tensor(translation_vector, dtype=torch.float64).to(device)

    return rotation_vector, translation_vector


def rotation_vector_to_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)


def pad_features(features, target_length):
    padded_features = []
    for feature_vector in features:
        if isinstance(feature_vector, tuple):
            feature_vector = list(feature_vector)
        if len(feature_vector) < target_length:
            padded_vector = feature_vector + [0] * (target_length - len(feature_vector))
        else:
            padded_vector = feature_vector[:target_length]
        padded_features.append(padded_vector)
    return padded_features


def classify_eye_openness(ear, threshold_open=0.3, threshold_closed=0.2):
    """
    Classifies eye openness based on EAR (Eye Aspect Ratio).

    Parameters:
    - ear (float): The EAR value for the current frame.
    - threshold_open (float): The EAR threshold above which eyes are considered fully open.
    - threshold_closed (float): The EAR threshold below which eyes are considered closed.

    Returns:
    - str: A classification of the eye openness ("Fully Open", "Partially Open", "Closed").
    """
    if ear > threshold_open:
        return 0
    elif threshold_closed <= ear <= threshold_open:
        return 1
    else:
        return 2


def compute_gaze_direction(shape_tensor, rotation_vector, image_size):
    """
    Compute gaze direction based on eye landmarks and head pose.

    Parameters:
        shape_tensor (torch.Tensor): Landmarks of the face.
        rotation_vector (torch.Tensor): Rotation vector from head pose estimation.
        image_size (tuple): Size of the image (height, width).

    Returns:
        gaze_direction (tuple): Estimated gaze direction (pitch, yaw).
    """
    left_eye_landmarks = shape_tensor[36:42]
    right_eye_landmarks = shape_tensor[42:48]

    left_eye_center = left_eye_landmarks.mean(dim=0)
    right_eye_center = right_eye_landmarks.mean(dim=0)

    eye_center = (left_eye_center + right_eye_center) / 2

    image_center = torch.tensor([image_size[1] / 2, image_size[0] / 2], dtype=torch.float32)

    gaze_vector = image_center - eye_center[:2]

    pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector.cpu().numpy())

    return pitch, yaw, gaze_vector.tolist()


def classify_head_pose(pitch, yaw, roll, gaze_vector):
    """
    Classifies head pose based on pitch, yaw, roll, and gaze direction.

    Parameters:
        pitch (float): The pitch angle.
        yaw (float): The yaw angle.
        roll (float): The roll angle.
        gaze_vector (list): The gaze direction vector [x, y].

    Returns:
        str: A classification string representing the head pose.
    """
    gaze_direction = 0
    if gaze_vector[0] > 0.5:
        gaze_direction = 1
    elif gaze_vector[0] < -0.5:
        gaze_direction = 2

    if pitch > 10:
        position = 3
    elif pitch < -10:
        position = 4
    elif yaw > 10:
        position = 5
    elif yaw < -10:
        position = 6
    elif roll > 10:
        position = 7
    elif roll < -10:
        position = 8
    else:
        position = 0

    return position, gaze_direction


def process_frame(frame):
    frame_np = frame
    gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 0)
    features = []

    shapes = [face_utils.shape_to_np(predictor(gray, rect)) for rect in rects]
    best_face_index = None
    if len(rects) == 0:
        return features
    max_width = 0
    best_features = None
    rects = sorted(rects, key=lambda rect: rect.width() * rect.height(), reverse=True)
    MIN_FACE_SIZE = 5000
    for i, rect in enumerate(rects):
        if rect.width() * rect.height() < MIN_FACE_SIZE:
            continue
        shape = shapes[i]
        shape_tensor = torch.tensor(shape, dtype=torch.float32)

        leftEye = shape_tensor[42:48]
        rightEye = shape_tensor[36:42]
        ear = (eye_aspect_ratio_torch(leftEye) + eye_aspect_ratio_torch(rightEye)) / 2.0

        size = (frame_np.shape[1], frame_np.shape[0])
        rotation_vector, translation_vector = get_head_pose_torch(shape_tensor, size)
        pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector.numpy())

        if pitch > 90:
            if pitch < 0:
                pitch = abs(pitch) - 180
                pitch = 0 - pitch
            else:
                pitch = abs(pitch) - 190
        gaze_vector = compute_gaze_direction(shape_tensor, rotation_vector, size)
        position, gaze_direction = classify_head_pose(pitch, yaw, roll, gaze_vector)
        eye_category = classify_eye_openness(ear.item())

        current_features = [eye_category, position, gaze_direction]

        face_width = rect.width() * rect.height()
        if face_width > max_width:
            max_width = face_width
            best_face_index = i
            best_features = current_features

    if best_face_index is not None:
        features = best_features

    return features


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]

    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sp.sparse.spdiags(diags_data, diags_index,
                          (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def _process_video(frames, roi):
    """Calculates the average value within the specified ROI for each frame."""
    RGB = []
    for frame in frames:
        x, y, w, h = roi
        frame_roi = frame[y:y + h, x:x + w]
        summation = np.sum(np.sum(frame_roi, axis=0), axis=0)
        RGB.append(summation / (frame_roi.shape[0] * frame_roi.shape[1]))
    return np.asarray(RGB)


def pos_wang(frames, fs=FPS):
    frames = np.array(frames)
    WinSec = 1.6
    N = frames.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)
    RGB = None
    cumulative_systolic_sum = 0
    cumulative_diastolic_sum = 0

    for n in range(N):
        frame_rgb = frames[n]

        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)

        if RGB is None:
            face_landmarks_list = face_recognition.face_landmarks(frame_rgb)
            if face_landmarks_list:
                landmarks = face_landmarks_list[0]
                roi = extract_forehead_roi(landmarks)
            else:
                roi = (0, 0, frame_rgb.shape[1], frame_rgb.shape[0])

            RGB = _process_video(frames, roi)

        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.asmatrix(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.asmatrix(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = sp.signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = sp.signal.filtfilt(b, a, BVP.astype(np.double))

    BVP = BVP.copy()

    peaks, _ = sp.signal.find_peaks(BVP, distance=fs * 0.6)

    RR_intervals = np.diff(peaks) / fs

    avg_peak_to_peak_interval = np.mean(RR_intervals)

    heart_rate = 60 / RR_intervals

    systolic_peaks = BVP[peaks]
    diastolic_peaks = BVP[peaks - 1]

    cumulative_systolic_sum = np.sum(systolic_peaks)
    cumulative_diastolic_sum = np.sum(diastolic_peaks)

    average_heart_rate = np.mean(heart_rate)

    return [average_heart_rate, avg_peak_to_peak_interval, cumulative_systolic_sum, cumulative_diastolic_sum]


def extract_forehead_roi(landmarks):
    """Extracts the ROI for the forehead based on facial landmarks."""
    forehead_center_x = (landmarks['left_eyebrow'][0][0] + landmarks['right_eyebrow'][-1][0]) // 2
    forehead_center_y = landmarks['left_eyebrow'][0][1] - 40
    forehead_roi = (forehead_center_x - 50, forehead_center_y - 30, 100, 30)
    return forehead_roi


def process_subject_batch(batch, label,id, n=(FPS * VIDEO_LENGTH) // FRAME_INTERVAL):
    ear_choices = np.array([1, 2])
    eye_gaze_choices = np.array([1, 2])
    head_pose_choices = np.array([3, 4, 5, 6, 7, 8])
    subject_frames = []
    try:
        for frame in batch:
            frame_features = process_frame(frame)
            if frame_features:
                subject_frames.append(frame_features)
            else:
                subject_frames.append([np.random.choice(ear_choices), np.random.choice(eye_gaze_choices),
                                       np.random.choice(head_pose_choices)])
        max_length = max(len(f) for f in subject_frames) if subject_frames else 0
        subject_frames = pad_features(subject_frames, max_length)
        subject_frames = np.array(subject_frames)

        if isinstance(subject_frames, np.ndarray):
            l = subject_frames.shape[0] // n
            # subject_frames = subject_frames[:l * n].reshape(l, n, -1)
            subject_frames = sp.stats.mode(subject_frames,keepdims=True)[0][0]

        subject_frames = np.array(subject_frames)
        
        X_batch_N = np.array(pos_wang(batch))
        return subject_frames, X_batch_N, label,id
    except Exception as e:
        print(f"Error processing batch: {e}")
        return subject_frames


X_batch_A = []
X_batch_N = []
all_labels = []
all_ids = []
iters = GLOBAL_BATCH_SIZE // BATCH_SIZE

with ThreadPoolExecutor() as executor:
    futures = []

    for i in range(iters):
        X_batch, Y_batch, ID_batch= load_data(PATH, DATASET, BATCH_SIZE)

        print(f'{i + 1} iterations done')

        for batch, label, id in zip(X_batch, Y_batch, ID_batch):
            futures.append(executor.submit(process_subject_batch, batch, label, id))

    for future in futures:
        subject_frames, phys_features, label, id= future.result()
        X_batch_A.append(subject_frames)
        X_batch_N.append(phys_features)
        all_labels.append(label)
        all_ids.append(id)
all_labels = np.array(all_labels)
all_ids = np.array(all_ids)
# combined_features = np.hstack((X_batch_A, X_batch_N))
combined_features = np.hstack((all_ids, X_batch_A))
combined_features = np.hstack((combined_features, X_batch_N))
combined_features_with_labels = np.hstack((combined_features, all_labels))
df = pd.DataFrame(combined_features_with_labels,
                  columns=['all_ids','eye_category', 'eye_position', 'gaze_direction', 'heart_rates', 'p2p_intervals',
                           'sys_peaks', 'dys_peaks', 'engagement_labels'])

filename = f'/SL_DAiSEE.csv'
df.to_csv(filename, index=False)
print(f'Successfully saved to {filename}')
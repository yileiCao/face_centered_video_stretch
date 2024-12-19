import mediapipe as mp
import cv2
import numpy as np
from collections import deque

def process_video_faces(video_frames, confidence_threshold=0.5):
    """
    Detects faces frame-by-frame in a video using MediaPipe Face Detection.

    :param video_frames: List of video frames.
    :param confidence_threshold: Minimum confidence for detecting a face.
    :return: List of tuples with detected face coordinates per frame.
    
    Each tuple is (frame_index, x, y, w, h).
    """
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=confidence_threshold)

    
    detected_faces = []
    frame_index = 0

    for frame in video_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bbox_c = detection.location_data.relative_bounding_box
                x = int(bbox_c.xmin * frame.shape[1])
                y = int(bbox_c.ymin * frame.shape[0])
                w = int(bbox_c.width * frame.shape[1])
                h = int(bbox_c.height * frame.shape[0])
                detected_faces.append((frame_index, x, y, w, h))

        frame_index += 1

    face_detection.close()
    return detected_faces


def filter_too_big_faces(faces, pixel_num):
    """
    Filter out faces that are too big (norminally misdetected)

    :param faces: List of tuples (frame_index, x, y, w, h) for detected faces.
    :param pixel_num: face bigger than the number is considered misdetected.
    :return: List of filtered face detections.
    """
    filtered_faces = []
    for frame_idx, x, y, w, h in faces:
        if h <= pixel_num:
            filtered_faces.append((frame_idx, x, y, w, h))
    return filtered_faces


def process_faces(detected_faces, total_frames, min_consecutive_frames=10, interpolation=True):
    # First track the faces
    face_tracks = track_faces(detected_faces, min_consecutive_frames)
    
    # Then interpolate (you can comment this out to disable interpolation)
    if interpolation:
        face_tracks = interpolate_face_tracks(face_tracks, total_frames)
    
    return face_tracks


def track_faces(detected_faces, min_consecutive_frames=10):
    """
    Groups faces into tracks and filters out transient faces.
    
    :param detected_faces: List of tuples (frame_index, x, y, w, h) for detected faces.
    :param min_consecutive_frames: Minimum number of consecutive frames a face must appear to be retained.
    :return: Dictionary of valid face tracks without interpolation
    """
    face_tracks = {}

    def is_same_face(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right > x_left and y_bottom > y_top:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            smaller_area = min(area1, area2)
            return intersection_area / smaller_area > 0.8
        return False

    # Group faces into tracks
    for frame_index, x, y, w, h in detected_faces:
        box = (x, y, w, h)
        matched = False
        for face_id, track in face_tracks.items():
            if any(is_same_face(box, prev_box) for prev_frame, prev_box in track if prev_frame > frame_index-45):
                if not any(prev_frame == frame_index for prev_frame, _ in track):
                    track.append((frame_index, box))
                    matched = True
                    break
        if not matched:
            face_id = len(face_tracks)
            face_tracks[face_id] = [(frame_index, box)]

    # Filter tracks based on consecutive frames
    valid_tracks = {}
    for face_id, track in face_tracks.items():
        track.sort(key=lambda x: x[0])
        consecutive_count = 1
        valid = False
        for i in range(1, len(track)):
            if track[i][0] == track[i - 1][0] + 1:
                consecutive_count += 1
            else:
                consecutive_count = 1
            if consecutive_count >= min_consecutive_frames:
                valid = True
                break
        if valid:
            valid_tracks[face_id] = track

    return valid_tracks


def interpolate_face_tracks(face_tracks, total_frames):
    """
    Interpolates missing frames for each face track.
    
    :param face_tracks: Dictionary of face tracks from track_faces()
    :param total_frames: Total number of frames in the video
    :return: Dictionary with interpolated face tracks
    """
    processed_tracks = {}
    for face_id, track in face_tracks.items():
        known_frames = [frame for frame, box in track]
        known_coords = np.array([box for frame, box in track])
        all_frames = np.arange(total_frames)
        interpolated_coords = np.zeros((total_frames, 4))

        for i in range(4):
            interpolated_coords[:, i] = np.interp(all_frames, known_frames, known_coords[:, i])

        processed_tracks[face_id] = [
            (frame, tuple(map(int, interpolated_coords[frame])))
            for frame in all_frames
        ]

    return processed_tracks


def smooth_bounding_boxes_by_frame(processed_tracks, window_size=20):
    """
    Smooth bounding box coordinates using a moving average and arrange the results by frame number.

    :param processed_tracks: Dictionary where keys are face IDs, and values are lists of tuples
                             (frame_index, x, y, w, h) for each detected face.
    :param window_size: Number of frames for moving average smoothing.
    :return: Dictionary where keys are frame numbers, and values are lists of smoothed bounding boxes.
    """
    smoothed_frames = {}

    # Iterate over each face track
    for face_id, track in processed_tracks.items():
        smoothing_queue = deque(maxlen=window_size)
        track.sort(key=lambda x: x[0])  # Ensure the track is sorted by frame_index

        for frame_index, (x, y, w, h) in track:
            current_box = np.array([x, y, w, h])
            smoothing_queue.append(current_box)

            # Compute the smoothed box as the mean of the queue
            smoothed_box = np.mean(smoothing_queue, axis=0)
            smoothed_box = tuple(map(int, smoothed_box))

            # Add the smoothed box to the corresponding frame in the output
            smoothed_frames.setdefault(frame_index, []).append(smoothed_box)

    return smoothed_frames



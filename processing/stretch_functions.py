# processing/video_filters.py
import cv2
import numpy as np

def unified_stretch(frame):
    """
    Resize the frame from 4:3 to 16:9 aspect ratio.
    Args:
        frame: Input image frame to be resized.

    Returns:
        resized_frame: Resized frame with 16:9 aspect ratio.
    """
    # Get the current dimensions of the frame
    height, width = frame.shape[:2]
    
    # Calculate the new dimensions for 16:9 aspect ratio
    new_height = height
    new_width = int(new_height * 16 / 9)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def low_center_stretch_based_on_faces(x, width, faces):
    if not faces:
        faces = [width / 2,]
    k = (4 / 3 - 1) / (width / 2) ** 2  # 调整积分条件的系数
    closest_face_idx = []
    for i in range(width):
        # Find the closest peak to the current index
        closest_face = min(faces, key=lambda face: abs(face - i))
        closest_face_idx.append(closest_face)
    return 1 + k * (x - closest_face_idx) ** 2


def gaussian_stretch_based_on_faces(x, width, faces):
    if not faces:
        faces = [width / 2]
    
    # Parameters
    target_ratio = 16/9
    sigma = width / 4  # Controls how wide the protected region is
    max_stretch = target_ratio  # Maximum stretch factor at edges
    min_stretch = 1.0  # Minimum stretch factor at face centers
    
    # Initialize with maximum stretch
    stretch_factors = np.full_like(x, max_stretch)
    
    # Apply Gaussian influence from each face
    for face_center in faces:
        # Calculate Gaussian weight for each point
        gaussian = np.exp(-((x - face_center) ** 2) / (2 * sigma ** 2))
        
        # Interpolate between max_stretch and min_stretch based on Gaussian
        current_stretch = max_stretch - (max_stretch - min_stretch) * gaussian
        
        # Take the minimum stretch factor at each point
        stretch_factors = np.minimum(stretch_factors, current_stretch)
    
    # Normalize to ensure we achieve the target ratio
    current_width_factor = np.mean(stretch_factors)
    stretch_factors *= target_ratio / current_width_factor
    
    return stretch_factors


def stretch_image(image, stretch_func, faces):
    """
    Stretch an image using a stretch function based on detected faces and update face coordinates.

    :param image: Input image as a NumPy array.
    :param stretch_func: Function to calculate the stretch rate.
    :param faces: List of detected face bounding boxes (x, y, w, h).
    :return: Stretched image, updated face coordinates, and the stretch rate used for transformation.
    """
    face_centers = [x + w // 2 for x, y, w, h in faces]
    height, width, channels = image.shape
    target_width = int(height * 16 / 9)
    stretched_image = np.zeros((height, target_width, channels), dtype=image.dtype)

    # Compute the stretch rate for each x-coordinate
    x_coords = np.linspace(0, width - 1, width)
    stretch_rate = stretch_func(x_coords, width, face_centers)
    cumulative_map = np.cumsum(stretch_rate)
    cumulative_map = cumulative_map / cumulative_map[-1] * (target_width - 1)

    # Map input pixels to stretched coordinates
    for i in range(target_width):
        src_x = np.searchsorted(cumulative_map, i)
        src_x = min(src_x, width - 1)
        stretched_image[:, i, :] = image[:, src_x, :]

    # Update face coordinates based on the stretch mapping
    updated_faces = []
    for x, y, w, h in faces:
        # Calculate the new x position for the stretched image
        new_x1 = int(cumulative_map[max(x, 0)])
        new_x2 = int(cumulative_map[min(x + w - 1, width - 1)])
        updated_faces.append((new_x1, y, new_x2 - new_x1, h))

    return stretched_image, updated_faces, stretch_rate


def stretch_image_old(image, stretch_func, faces):
    """
    Stretch an image using a stretch function based on detected faces and update face coordinates.

    :param image: Input image as a NumPy array.
    :param stretch_func: Function to calculate the stretch rate.
    :param faces: List of detected face bounding boxes (x, y, w, h).
    :return: Stretched image, updated face coordinates, and the stretch rate used for transformation.
    """
    face_centers = faces
    height, width, channels = image.shape
    target_width = int(height * 16 / 9)
    stretched_image = np.zeros((height, target_width, channels), dtype=image.dtype)

    # Compute the stretch rate for each x-coordinate
    x_coords = np.linspace(0, width - 1, width)
    stretch_rate = stretch_func(x_coords, width, face_centers)
    cumulative_map = np.cumsum(stretch_rate)
    cumulative_map = cumulative_map / cumulative_map[-1] * (target_width - 1)

    # Map input pixels to stretched coordinates
    for i in range(target_width):
        src_x = np.searchsorted(cumulative_map, i)
        src_x = min(src_x, width - 1)
        stretched_image[:, i, :] = image[:, src_x, :]

    return stretched_image #, updated_faces, stretch_rate
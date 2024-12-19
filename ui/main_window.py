# ui/main_window.py
import sys
import os
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QGridLayout, QSizePolicy
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal
import cv2
import numpy as np
from processing.scene_detection import detect_scenes
from processing.stretch_functions import gaussian_stretch_based_on_faces, low_center_stretch_based_on_faces, stretch_image, unified_stretch
from processing.face_processing import process_video_faces, filter_too_big_faces, process_faces, smooth_bounding_boxes_by_frame
import mediapipe as mp
import queue


class SceneProcessor(QThread):
    finished = pyqtSignal(int)  # Signal to emit when processing is complete

    def __init__(self, app, scene_index):
        super().__init__()
        self.app = app
        self.scene_index = scene_index
        
    def run(self):
        self.app._process_scene(self.scene_index)
        self.finished.emit(self.scene_index)


class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing Application")
        self.setGeometry(100, 100, 1200, 800)

        # Main Layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Processed Results Section (Top)
        self.processed_layout = QHBoxLayout()  # Change to QHBoxLayout for horizontal arrangement
        self.layout.addLayout(self.processed_layout)

        self.result_labels = []
        self.processed_players = []
        self.processed_video_widgets = []

        for i in range(4):  # Assuming 4 processing protocols
            # Create video widget and player for each preview
            video_widget = QVideoWidget()
            player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
            player.setVideoOutput(video_widget)
            
            # Create container widget and its layout
            container = QWidget()
            container.setFixedHeight(250) 
            container_layout = QVBoxLayout(container)
            container_layout.addWidget(video_widget)
            
            # Set size policy for video widget to allow shrinking
            video_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            video_widget.setMinimumWidth(200)  # Set minimum width to prevent too much shrinking
            
            # Add label below video widget
            label = QLabel(f"Result Preview {i + 1}")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedHeight(20)  # Set a fixed small height
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Allow horizontal stretch but fix vertical size
            label.setStyleSheet("QLabel { padding: 2px; } QLabel:hover { background-color: #e0e0e0; }")
            label.mousePressEvent = lambda e, idx=i: self.load_video(self.media_player, self.current_scene_index, 
                                                                     self.processed_players[idx].position(), idx)
            container_layout.addWidget(label)
            
            # Store references
            self.processed_video_widgets.append(video_widget)
            self.processed_players.append(player)
            self.result_labels.append(container)
            
            # Add to horizontal layout
            self.processed_layout.addWidget(container)

        # Video Player Section (Bottom)
        self.video_widget = QVideoWidget()
        self.layout.addWidget(self.video_widget)

        # Media Player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # Scene Info
        self.scenes = []  # List of (start_time, end_time)
        self.current_scene_index = 0

        # Controls
        self.controls_layout = QHBoxLayout()
        self.layout.addLayout(self.controls_layout)

        self.open_button = QPushButton("Open Video")
        self.open_button.clicked.connect(self.open_video)
        self.controls_layout.addWidget(self.open_button)

        self.previous_button = QPushButton("Previous Scene")
        self.previous_button.clicked.connect(self.previous_scene)
        self.controls_layout.addWidget(self.previous_button)

        self.play_button = QPushButton("Play Scene")
        self.play_button.clicked.connect(self.play_scene)
        self.controls_layout.addWidget(self.play_button)

        self.next_button = QPushButton("Next Scene")
        self.next_button.clicked.connect(self.next_scene)
        self.controls_layout.addWidget(self.next_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.controls_layout.addWidget(self.pause_button)

        # Timer for scene playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_scene_end)

        # Add these new attributes
        self.current_video_path = None
        self.processing_thread = None
        self.processed_scenes = set()  # Keep track of which scenes are processed
        self.processing_queue = queue.Queue()  # Queue for scenes to process

        # Connect error signals for debugging
        self.media_player.error.connect(self.handle_media_error)
        for player in self.processed_players:
            player.error.connect(self.handle_media_error)

        # Ensure video widgets are visible
        for video_widget in self.processed_video_widgets:
            video_widget.show()

        # Initialize MediaPipe face detection as class attribute
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def open_video(self):
        """Open a video file and perform scene detection."""
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.current_video_path = file_path
            self.scenes = detect_scenes(file_path)  # Detect scenes
            print(self.scenes)
            self.current_scene_index = 0
            self.process_current_scene()
            
            self.set_videos()
        # Update labels
        method_names = ["Unified Stretch", "Low Center Stretch", 
                       "Gaussian Stretch", "Original Scene"]
        for label, name in zip(self.result_labels, method_names):
            label.findChild(QLabel).setText(name)


    def set_videos(self):
        for i in range(4):
            self.load_video(self.processed_players[i], self.current_scene_index, 0, i)
        self.load_video(self.media_player, self.current_scene_index, 0, 3)
        self.play_scene()

    def process_current_scene(self):
        """Process the current scene and queue processing for the next scene."""
        if not self.scenes or self.current_scene_index >= len(self.scenes):
            return

        # Process current scene if not already processed
        if self.current_scene_index not in self.processed_scenes:
            if (self.current_scene_index not in self.processing_queue.queue and
                (not self.processing_thread or not self.processing_thread.isRunning() or
                 self.processing_thread.scene_index != self.current_scene_index)):
                self._process_scene(self.current_scene_index)
            self.processed_scenes.add(self.current_scene_index)

        # Queue next two scenes for processing if they exist
        for offset in range(1, 3):  # Check the next two scenes
            next_scene_index = self.current_scene_index + offset
            if next_scene_index < len(self.scenes) and next_scene_index not in self.processed_scenes:
                if (next_scene_index not in self.processing_queue.queue and
                    (not self.processing_thread or not self.processing_thread.isRunning() or
                     self.processing_thread.scene_index != next_scene_index)):  # Check if not being processed
                    self.queue_scene_processing(next_scene_index)

    def queue_scene_processing(self, scene_index):
        """Start processing a scene in the background."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_queue.put(scene_index)
        else:
            self.start_scene_processing(scene_index)

    def start_scene_processing(self, scene_index):
        """Start the background processing thread for a scene."""
        self.processing_thread = SceneProcessor(self, scene_index)
        self.processing_thread.finished.connect(self.on_scene_processing_finished)
        self.processing_thread.start()

    def on_scene_processing_finished(self, scene_index):
        """Handle completion of scene processing."""
        self.processed_scenes.add(scene_index)
        
        # Process next scene in queue if any
        try:
            next_scene_index = self.processing_queue.get_nowait()
            self.start_scene_processing(next_scene_index)
        except queue.Empty:
            pass

    def _process_scene(self, scene_index):
        """Internal method to process a specific scene."""
        if scene_index >= len(self.scenes):
            return

        start_time, end_time = self.scenes[scene_index]
        
        # Create temporary processed videos for the current scene with different methods
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time.get_seconds() * fps)
        end_frame = int(end_time.get_seconds() * fps)
        
        # Set output paths for processed videos
        cwd = os.getcwd()
        processed_paths = [
            os.path.join(cwd, f"temp_processed_scene_{scene_index}_method{i}.mp4")
            for i in range(4)
        ]
        
        # Get frame dimensions and prepare video writers
        ret, frame = cap.read()
        if not ret:
            return

        height = frame.shape[0]
        width = frame.shape[1]

        # Initialize video writers for each method
        writers = []
        for i in range(4):
            if sys.platform == 'darwin':
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            if i < 3:  # For methods 1-3 (processed versions)
                target_width = int(height * 16 / 9)
                writer = cv2.VideoWriter(processed_paths[i], fourcc, fps, (target_width, height))
            else:  # For method 4 (original version)
                writer = cv2.VideoWriter(processed_paths[i], fourcc, fps, (width, height))
            writers.append(writer)
        
        # Extract frames for the current scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        scene_frames = []
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            scene_frames.append(frame)

        # Detect and process faces for the entire scene
        faces = process_video_faces(scene_frames, confidence_threshold=0.5)
        faces = filter_too_big_faces(faces, pixel_num=int(height * 0.7))
        faces = process_faces(faces, len(scene_frames), interpolation=True)
        faces = smooth_bounding_boxes_by_frame(faces, window_size=20)

        # Process each frame in the scene
        for frame_index, frame in enumerate(scene_frames):
            # Get faces for the current frame
            faces_in_frame = list(faces.get(frame_index, []))

            # Process frame with different methods
            # Method 1: Unified stretch (original)
            processed_frame = unified_stretch(frame)
            writers[0].write(processed_frame)
            
            # Method 2: Low center stretch
            processed_frame, _, _ = stretch_image(frame, low_center_stretch_based_on_faces, faces_in_frame)
            writers[1].write(processed_frame)
            
            # Method 3: Gaussian stretch
            processed_frame, _, _ = stretch_image(frame, gaussian_stretch_based_on_faces, faces_in_frame)
            writers[2].write(processed_frame)
            
            # Method 4: Original scene (no processing)
            writers[3].write(frame)  # Write the original frame without processing
        
        # Clean up
        cap.release()
        for writer in writers:
            writer.release()
        
        time.sleep(0.5)


    def synchronize_videos(self):
        """Synchronize all video players to maintain consistent playback."""
        if not self.scenes:
            return

        main_position = self.media_player.position()
        for player in self.processed_players:
            player.setPosition(main_position)

        # Ensure all videos have the same play/pause state as the main video
        main_is_playing = self.media_player.state() == QMediaPlayer.PlayingState
        for player in self.processed_players:
            if main_is_playing:
                player.play()
            else:
                player.pause()

    def load_video(self, video_player, scene_index, scene_position, method_index):
        """Switch the video player to play the specified scene starting from the given position."""
        if not self.scenes or scene_index >= len(self.scenes):
            return

        was_playing = video_player.state() == QMediaPlayer.PlayingState

        selected_video_path = os.path.join(
            os.getcwd(),
            f"temp_processed_scene_{scene_index}_method{method_index}.mp4"
        )

        video_player.setMedia(QMediaContent(QUrl.fromLocalFile(selected_video_path)))
        video_player.setPosition(scene_position)

        if was_playing:
            video_player.play()


    def play_scene(self):
        """Play the current scene with synchronization."""
        self.media_player.play()
        self.synchronize_videos()
        self.timer.start(50)

    def check_scene_end(self):
        """Loop playback at the end of the current scene."""
        if self.media_player.duration() > 0 and self.media_player.position() >= self.media_player.duration():
            # Reset all players to beginning
            self.media_player.setPosition(0)
            for player in self.processed_players:
                player.setPosition(0)
            
            # Restart playback
            self.media_player.play()
            for player in self.processed_players:
                player.play()

    def pause_video(self):
        """Pause video playback."""
        self.media_player.pause()
        self.synchronize_videos()
        self.timer.stop()

    def previous_scene(self):
        """Go to the previous scene."""
        if self.current_scene_index > 0:
            self.current_scene_index -= 1
            self.process_current_scene()
            self.set_videos()

    def next_scene(self):
        """Go to the next scene."""
        if self.current_scene_index < len(self.scenes) - 1:
            self.current_scene_index += 1
            self.process_current_scene()
            self.set_videos()

    def choose_result(self, idx):
        """Handle user choice for processing protocol."""
        print(f"User selected protocol {idx + 1}")


    def cleanup(self):
        """Clean up resources before closing."""
        try:
            # Stop the timer first
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
            
            # Clean up media players if they still exist
            if hasattr(self, 'media_player'):
                self.media_player.stop()
                self.media_player.setMedia(QMediaContent())
            
            if hasattr(self, 'processed_players'):
                for player in self.processed_players:
                    player.stop()
                    player.setMedia(QMediaContent())
            
            # Clean up temporary files
            if hasattr(self, 'scenes'):
                for i in range(len(self.scenes)):
                    for j in range(4):
                        temp_file = f"temp_processed_scene_{i}_method{j}.mp4"
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception as e:
                                print(f"Error removing temporary file {temp_file}: {e}")
            
            # Add face detection cleanup
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            
            # Stop background processing
            if self.processing_thread:
                if self.processing_thread.isRunning():
                    self.processing_thread.terminate()
                    self.processing_thread.wait()  # Ensure the thread has finished
                self.processing_thread = None  # Clear the reference to the thread
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def on_close(self, event):
        """Handle the window close event."""
        self.cleanup()
        event.accept()

    def handle_media_error(self, error):
        """Handle media player errors."""
        print(f"Media player error: {error}, {self.sender().errorString()}")

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec_())


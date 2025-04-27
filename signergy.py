import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

        # Initialize drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Define landmarks for fingertips
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

        # Initialize results attribute
        self.results = None

    def find_hands(self, img, draw=True):
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image
        self.results = self.hands.process(img_rgb)

        # Draw landmarks if hands detected and draw flag is set
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return img

    def find_positions(self, img):
        h, w, c = img.shape
        self.landmark_list = []

        # Check if hands detected
        if self.results.multi_hand_landmarks:
            for hand_id, hand_lms in enumerate(self.results.multi_hand_landmarks):
                hand_landmarks = []

                # Get landmarks for each hand
                for id, lm in enumerate(hand_lms.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_landmarks.append([id, cx, cy])

                self.landmark_list.append(hand_landmarks)

        return self.landmark_list

    def extract_landmark_features(self):
        """
        Extract features from hand landmarks for sign language recognition.
        Returns flattened coordinates and computed features.
        """
        features = []

        if not self.results.multi_hand_landmarks:
            return None

        # Process the primary hand (first detected hand)
        hand_landmarks = self.results.multi_hand_landmarks[0]

        # Extract x, y, z coordinates
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        # Add computed features - distances between key points
        # Distance from wrist to each fingertip
        wrist = hand_landmarks.landmark[0]  # Wrist landmark

        # Calculate distances from wrist to each fingertip
        for tip_id in self.tip_ids:
            tip = hand_landmarks.landmark[tip_id]
            # Euclidean distance in 3D space
            distance = np.sqrt((tip.x - wrist.x) ** 2 +
                               (tip.y - wrist.y) ** 2 +
                               (tip.z - wrist.z) ** 2)
            features.append(distance)

        # Add angles between fingers
        # This could be expanded with more sophisticated geometric features

        return features


class SignLanguageRecognizer:
    def __init__(self, model_path=None):
        self.model = None
        self.label_encoder = None
        self.feature_columns = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train_model(self, data_path):
        """
        Train the sign language recognition model using the Kaggle dataset.

        Args:
            data_path: Path to the CSV file containing the dataset
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print("Available columns:", df.columns.tolist())
        # Extract features and labels
        # The Kaggle dataset likely has columns for each landmark x,y,z and a label column
        
        label_column = 'W - w'  # Change this to the actual label column name
    
        # Extract features and labels
        feature_cols = [col for col in df.columns if col != label_column]
        self.feature_columns = feature_cols

        X = df[feature_cols].values
        y = df[label_column].values  # Use the correct label column name

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42)

        print(f"Training model on {len(X_train)} samples...")
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        score = self.model.score(X_test, y_test)
        print(f"Model accuracy: {score:.4f}")


    def save_model(self, model_path="sign_language_model.pkl"):
        """Save the trained model to disk"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path="sign_language_model.pkl"):
        """Load a trained model from disk"""
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            return False

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {model_path}")
        return True

    def predict(self, features):
        """
        Predict the sign from hand landmark features

        Args:
            features: Extracted features from the HandTracker

        Returns:
            The predicted sign label
        """
        if self.model is None:
            return "Model not loaded"

        if features is None:
            return None

        # Ensure features match expected format
        if len(features) != len(self.feature_columns):
            print(f"Feature count mismatch. Expected {len(self.feature_columns)}, got {len(features)}")
            return None

        # Make prediction
        prediction = self.model.predict([features])[0]

        # Convert back to label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        return predicted_label


class SignLanguageUI:
    def __init__(self, model_path=None):
        # Initialize the tracker
        self.tracker = HandTracker()

        # Initialize the recognizer
        self.recognizer = SignLanguageRecognizer(model_path)

        # Initialize text buffer for sign language
        self.text_buffer = ""
        self.last_detected_sign = ""
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # 1 second cooldown between detections

        # UI settings
        self.panel_color = (40, 40, 40)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 120, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2

        # Setup camera capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            exit()

        # Get screen resolution for full screen display
        self.screen_width = 1280  # Default width
        self.screen_height = 720  # Default height

        # Set camera to capture at desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calculate UI layout
        self.video_width = int(self.screen_width * 0.7)  # 70% for video
        self.text_panel_width = self.screen_width - self.video_width

        # Create window
        cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Sign Language Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Initialize prediction confidence threshold
        self.confidence_threshold = 0.7

    def run(self):
        prev_time = 0

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # Process the frame to detect hands
            frame = self.tracker.find_hands(frame)
            landmark_list = self.tracker.find_positions(frame)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Process landmarks for sign recognition
            if self.tracker.results and self.tracker.results.multi_hand_landmarks:
                features = self.tracker.extract_landmark_features()
                if features and self.recognizer.model is not None:
                    # Predict the sign
                    sign = self.recognizer.predict(features)

                    # Only process if cooldown has elapsed
                    if sign and (current_time - self.last_detection_time >= self.detection_cooldown):
                        self.last_detected_sign = sign

                        # Add to text buffer
                        if self.text_buffer:
                            # Add space if the sign is different from the last one
                            if not self.text_buffer.endswith(sign):
                                self.text_buffer += " " + sign
                        else:
                            self.text_buffer = sign

                        self.last_detection_time = current_time

            # Create the UI layout
            ui_frame = self.create_ui_layout(frame, fps, landmark_list)

            # Display the image
            cv2.imshow("Sign Language Detection", ui_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear text buffer
                self.text_buffer = ""
            elif key == ord('s'):
                # Save text to file
                self.save_text_to_file()

        self.cap.release()
        cv2.destroyAllWindows()

    def create_ui_layout(self, frame, fps, landmark_list):
        # Resize the camera frame to fit our layout
        resized_frame = cv2.resize(frame, (self.video_width, self.screen_height))

        # Create a full canvas for our UI
        full_ui = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        full_ui[:, :] = self.panel_color  # Set background color

        # Place the resized camera frame on the left side
        full_ui[0:self.screen_height, 0:self.video_width] = resized_frame

        # Draw the separator line
        cv2.line(full_ui, (self.video_width, 0), (self.video_width, self.screen_height),
                 self.accent_color, 2)

        # Display FPS
        cv2.putText(full_ui, f"FPS: {int(fps)}", (10, 30),
                    self.font, self.font_scale, self.accent_color, self.thickness)

        # Display model status
        model_status = "Model: Loaded" if self.recognizer.model is not None else "Model: Not Loaded"
        cv2.putText(full_ui, model_status, (10, 70),
                    self.font, self.font_scale,
                    (0, 255, 0) if self.recognizer.model is not None else (0, 0, 255),
                    self.thickness)

        # Create the text panel on the right side
        self.draw_text_panel(full_ui)

        return full_ui

    def draw_text_panel(self, frame):
        # Draw header for text panel
        cv2.putText(frame, "Sign Language Text",
                    (self.video_width + 20, 40),
                    self.font, 1, self.text_color, self.thickness)

        # Draw separator under header
        cv2.line(frame,
                 (self.video_width + 10, 60),
                 (self.screen_width - 10, 60),
                 self.accent_color, 1)

        # Draw the last detected sign (if any)
        if self.last_detected_sign:
            cv2.putText(frame, f"Detected: {self.last_detected_sign}",
                        (self.video_width + 20, 100),
                        self.font, self.font_scale, self.accent_color, self.thickness)

        # Draw the text buffer with wrapping
        text_y = 160
        text_x = self.video_width + 20
        max_width = self.text_panel_width - 40

        # Split the text buffer into words
        words = self.text_buffer.split()
        if words:
            line = ""
            for word in words:
                test_line = line + " " + word if line else word
                # Check if adding this word would exceed the panel width
                text_size = cv2.getTextSize(test_line, self.font, self.font_scale, self.thickness)[0]

                if text_size[0] > max_width:
                    # Draw the current line and start a new one
                    cv2.putText(frame, line, (text_x, text_y),
                                self.font, self.font_scale, self.text_color, self.thickness)
                    text_y += 30
                    line = word
                else:
                    line = test_line

            # Draw the last line
            cv2.putText(frame, line, (text_x, text_y),
                        self.font, self.font_scale, self.text_color, self.thickness)

        # Draw controls at the bottom
        controls_y = self.screen_height - 90
        cv2.putText(frame, "Controls:",
                    (self.video_width + 20, controls_y),
                    self.font, self.font_scale, self.accent_color, self.thickness)

        cv2.putText(frame, "Q - Quit",
                    (self.video_width + 20, controls_y + 30),
                    self.font, self.font_scale, self.text_color, self.thickness)

        cv2.putText(frame, "C - Clear text",
                    (self.video_width + 20, controls_y + 60),
                    self.font, self.font_scale, self.text_color, self.thickness)

        cv2.putText(frame, "S - Save text",
                    (self.video_width + 20, controls_y + 90),
                    self.font, self.font_scale, self.text_color, self.thickness)

    def save_text_to_file(self, filename="sign_language_text.txt"):
        """Save the current text buffer to a file"""
        if not self.text_buffer:
            print("No text to save")
            return

        with open(filename, 'w') as f:
            f.write(self.text_buffer)
        print(f"Text saved to {filename}")

        # Display save confirmation on screen
        self.last_detected_sign = f"Saved to {filename}"
        self.last_detection_time = time.time()


def prepare_kaggle_dataset(file_path, output_path):
    """
    Process the Kaggle BSL dataset to prepare it for training.

    Args:
        file_path: Path to the downloaded dataset
        output_path: Path to save the processed dataset
    """
    print(f"Processing dataset from {file_path}...")

    # Load the dataset
    # The actual implementation depends on the specific format of the Kaggle dataset
    # This is a placeholder - you'll need to adjust based on the actual data format

    # Example assuming the dataset is in CSV format with columns for each landmark
    df = pd.read_csv(file_path)

    # Preprocessing steps might include:
    # 1. Normalizing coordinates
    # 2. Handling missing values
    # 3. Feature engineering

    # Save the processed dataset
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Sign Language Recognition System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--dataset', type=str, help='Path to the dataset for training')
    parser.add_argument('--model', type=str, default='sign_language_model.pkl', help='Path to save/load the model')
    parser.add_argument('--process-dataset', type=str, help='Path to the raw Kaggle dataset to process')
    args = parser.parse_args()

    # Process dataset if requested
    if args.process_dataset:
        output_path = 'processed_dataset.csv'
        prepare_kaggle_dataset(args.process_dataset, output_path)
        print(f"Use '--dataset {output_path}' to train the model with the processed data.")
        return

    # Train the model if requested
    if args.train:
        if not args.dataset:
            print("Please provide a dataset path with --dataset")
            return

        recognizer = SignLanguageRecognizer()
        recognizer.train_model(args.dataset)
        recognizer.save_model(args.model)
        return

    # Run the application
    app = SignLanguageUI(model_path=args.model)
    app.run()


if __name__ == "__main__":
    main()

from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.p', 'rb'))

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Adjust as per your labels

# Attempt to open the first camera
cap = cv2.VideoCapture(1)  # Try camera index 0 first
if not cap.isOpened():
    print("Error: Camera not found. Trying camera 1...")
    cap = cv2.VideoCapture(1)  # If the first doesn't work, try the second one

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        exit()  # Exit if no camera is found

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # # Create a more distinct rectangle and better text box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                # # cv2.putText(frame, predicted_character, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                # cv2.putText(frame, predicted_character, (x1 + 12, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0), 5, cv2.LINE_AA)  # Shadow
                # cv2.putText(frame, predicted_character, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)  # Text
                # First, draw a background rectangle behind the text
                # cv2.rectangle(frame, (x1 - 10, y1 - 30), (x1 + 130, y1 + 30), (0, 0, 0), -1)  # Black background box

# Apply Shadow effect (darker text behind the main text)
                cv2.putText(frame, predicted_character, (x1 + 12, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 0, 0), 10, cv2.LINE_AA)  # Shadow

# Apply Main Text (White)
                cv2.putText(frame, predicted_character, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 3.5, (255, 255, 255), 7, cv2.LINE_AA)  # White Text

# Apply Outline Effect with Dark Blue for visibility on lighter backgrounds
                cv2.putText(frame, predicted_character, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 0, 255), 3, cv2.LINE_AA)  # Outline (Dark Blue)



                

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, Response
import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
from imutils import face_utils
import pygame
import datetime
import time
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Inisialisasi pygame untuk memainkan suara alarm
pygame.mixer.init()

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    
    if ratio > 0.20:
        return 2
    elif ratio > 0.18:
        return 1
    else:
        return 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
video_url = "http://192.168.7.169:81/stream"
cap = None

# Set codec untuk menyimpan video ke format MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 60
COUNTER = 0
ALARM_ON = False
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Variabel untuk menyimpan nilai EAR
ear_values = []

def play_alarm(alarm_file):
    pygame.mixer.music.load(alarm_file)
    pygame.mixer.music.play()

def initialize_video_capture():
    global cap
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Gagal membuka stream. Memeriksa kembali...")
        cap.release()
        time.sleep(2)  # Beri jeda sebelum mencoba kembali
        cap = cv2.VideoCapture(video_url)
    return cap

def generate_frames():
    global sleep, drowsy, active, status, color, ALARM_ON, out, ear_values, cap

    cap = initialize_video_capture()

    while True:
        if cap is None or not cap.isOpened():
            print("Stream tidak terbuka, mencoba reconnect...")
            cap = initialize_video_capture()
            continue

        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame dari stream.")
            time.sleep(2)  # Jeda sejenak sebelum mencoba kembali
            cap.release()
            cap = initialize_video_capture()
            continue

        if out is None:
            # Inisialisasi penulisan video saat frame pertama diterima
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out = cv2.VideoWriter(f"{timestamp}.mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_eye_pts = landmarks[36:42]
            right_eye_pts = landmarks[42:48]

            for i in range(6):
                cv2.line(frame, tuple(left_eye_pts[i]), tuple(left_eye_pts[(i+1) % 6]), (0, 0, 255), 1)
            for i in range(6):
                cv2.line(frame, tuple(right_eye_pts[i]), tuple(right_eye_pts[(i+1) % 6]), (0, 0, 255), 1)

            leftEAR = eye_aspect_ratio(left_eye_pts)
            rightEAR = eye_aspect_ratio(right_eye_pts)
            
            # Simpan nilai EAR rata-rata ke dalam list
            ear_values.append((leftEAR + rightEAR) / 2)

            cv2.putText(frame, f"Left EAR: {leftEAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Right EAR: {rightEAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if leftEAR < EYE_AR_THRESH or rightEAR < EYE_AR_THRESH:
                sleep += 1
                drowsy += 1
                active = 0
                if sleep > EYE_AR_CONSEC_FRAMES:
                    status = "TIDURRR !!!"
                    color = (0, 0, 255)
                    if not ALARM_ON or pygame.mixer.music.get_busy() == 0:
                        ALARM_ON = True
                        play_alarm("alarm2.wav")
                    print("TIDURRR !!!")
            else:
                sleep = 0
                active += 1
                if drowsy > EYE_AR_CONSEC_FRAMES:
                    status = "mengantuk!"
                    color = (0, 0, 255)
                    if not ALARM_ON or pygame.mixer.music.get_busy() == 0:
                        ALARM_ON = True
                        play_alarm("alarm.wav")
                    print("mengantuk!")
                elif active > 1:
                    status = "Active :)"
                    color = (0, 255, 0)
                    if ALARM_ON:
                        ALARM_ON = False
                        pygame.mixer.music.stop()
                    print("Active :)")
                drowsy = 0

            # Posisi untuk menampilkan status "Active" di pinggir kiri bawah
            height, width = frame.shape[:2]
            cv2.putText(frame, status, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        out.write(frame)  # Simpan frame ke file video MP4

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Gagal mengenkode frame ke format JPEG.")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_ear_plot():
    while True:
        if len(ear_values) == 0:
            time.sleep(0.1)
            continue

        plt.figure(figsize=(5, 3))
        plt.plot(ear_values, label="EAR")
        plt.axhline(y=EYE_AR_THRESH, color='r', linestyle='--', label="Threshold")
        plt.legend(loc="upper right")
        plt.xlabel("Frame")
        plt.ylabel("EAR")
        plt.title("Eye Aspect Ratio over Time")
        plt.ylim(0, 0.4)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buf.getvalue() + b'\r\n')

@app.route('/')
def index():
    return '''
        <html>
        <body>
            <h1>Video Stream with Drowsiness Detection</h1>
            <div style="display: flex; justify-content: space-between;">
                <img src="/video_feed" width="640" height="480">
                <img src="/ear_plot_feed" width="640" height="480">
            </div>
        </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ear_plot_feed')
def ear_plot_feed():
    return Response(generate_ear_plot(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)
    finally:
        if out is not None:
            out.release()  # Tutup file video saat server dihentikan
        if cap is not None:
            cap.release()  # Tutup koneksi kamera

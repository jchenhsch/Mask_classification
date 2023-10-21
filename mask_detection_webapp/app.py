from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import dlib
import sys
sys.path.append('../')
from machine_learning_project.live_detection_dlib import live_detect_dlib
from machine_learning_project.live_detection_face_cascade import live_detect_face_cascade
from threading import Thread
import time

app = Flask(__name__)
cap = None
mask_prob = 0.0
mask_detection_thread = None 


# def generate_frames():
#     global mask_prob
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_detection():
    # global mask_detection_thread
    # if mask_detection_thread is None or not mask_detection_thread.is_alive():
    #     mask_detection_thread = Thread(target=live_detect_dlib, args=("/Users/james/mask_classification/machine_learning_project/my_model"))
    #     mask_detection_thread.start()

    return 'Mask detection started!'

@app.route('/stop')
def stop():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    return 'Mask detection stopped!'

@app.route('/video_feed')
def video_feed():

    return Response(live_detect_face_cascade("/Users/james/mask_classification/machine_learning_project/my_model"), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
if __name__ == '__main__':
    app.run(debug=True)

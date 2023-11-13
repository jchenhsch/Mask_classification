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
from flask_socketio import SocketIO
import time

app = Flask(__name__)
cap = None
SocketIO = SocketIO(app)
app.config['SECRET_KEY'] = 'secret!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_user():
    pass

@app.route('/sign_up')
def signup_user():
    pass


@app.route('/start')
def start_detection():
    with open('run_flag.txt', 'w') as file:
        file.write('True')
    return 'Mask detection started!'

@app.route('/stop')
def stop():
    with open('run_flag.txt', 'w') as file:
        file.write('False')
    return "stop mask detection"

@app.route('/video_feed')
def video_feed():
    with open('run_flag.txt', 'w') as file:
        file.write('True')
    print("here")
    return Response(live_detect_face_cascade("/Users/james/mask_classification/machine_learning_project/my_model"), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
if __name__ == '__main__':
    SocketIO.run(app,debug=True)

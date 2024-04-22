from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import torch
import cv2
import numpy as np
import requests
import math
from super_gradients.training import models

app = Flask(__name__)
socketio = SocketIO(app)

Model_arch = 'yolo_nas_m'
best_model = models.get(
    Model_arch,
    num_classes=4,  
    checkpoint_path='best.pth'  
)

names = ["balloon", "bird", "drone", "plane"]

TELEGRAM_BOT_TOKEN = '6322950453:AAH4TkChrwHMzG8XELiI6Mi4FX0UR1dtSRQ'
TELEGRAM_CHAT_ID = '5156316571'  

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    params = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    response = requests.get(url, params=params)
    return response.json()

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

count = 0

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

def gen():
    global count
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            prediction = best_model.predict(frame, conf=0.35)
            
            bbox_xyxys = prediction.prediction.bboxes_xyxy.tolist()
            confidences = prediction.prediction.confidence
            labels = prediction.prediction.labels.tolist()
            
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = names[classname]
                conf = math.ceil((confidence*100))/100
                
                if conf > 0.4:
                    if class_name == 'drone':
                        if conf > 0.5:
                            socketio.emit('alert', 'Drone detected!')
                            send_telegram_message("Drone detected!")
                        
                    label = f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
            
                    cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (
                        x2, y2), (0, 255, 255), 3)
            
            out.write(frame)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)

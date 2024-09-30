import torch
import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response

app = Flask(__name__)

# YOLOv8 모델 불러오기
model = YOLO('best.pt')

# 웹캠에서 비디오 스트림을 캡처
def gen_frames():
    cap = cv2.VideoCapture(2)  # 웹캠 사용 (2번이 웹캠)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # YOLOv8 모델 적용 (프레임에서 객체 감지)
            results = model(frame)
            frame = results[0].plot()  # 탐지된 객체를 프레임 위에 표시

            # 프레임을 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # 프레임을 클라이언트로 스트리밍
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # 웹캠에서 받아온 프레임을 스트리밍
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # index.html 렌더링
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

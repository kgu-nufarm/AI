from flask import Flask, jsonify, render_template, Response, request
import cv2
import threading
import os
import time
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model1 = YOLO('abnormal_1.pt')
model2 = YOLO('best.pt')

# 웹캠 인덱스
camera = cv2.VideoCapture(2)

# 이미지 저장 경로
img_path1 = './static/captured_image_model1.jpg'
img_path2 = './static/captured_image_model2.jpg'

# 파일 접근을 보호하기 위한 Lock 생성
lock = threading.Lock()

# 클래스 이름과 색상 정의 (각 모델에 대해)
class_names_model1 = {
    0: 'hole',
    1: 'wither'
}
class_names_model2 = {
    0: 'level_1',
    1: 'level_2',
    2: 'level_3',
}

class_colors_model1 = {
    0: (255, 0, 0),  # 빨간색 (구멍)
    1: (127, 255, 212),  # 민트색 (시든거)
}
class_colors_model2 = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 255),
}

# 감지 상태를 저장하는 변수
detected_objects_model1 = []
detected_objects_model2 = []
send_updates = True

def capture_image_periodically(model, img_path, class_names, class_colors, detected_objects):
    global send_updates
    while True:
        if not send_updates:
            time.sleep(1)
            continue

        camera.grab()
        ret, frame = camera.retrieve()

        if ret:
            results = model(frame)
            current_detected = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_id = int(box.cls)
                    color = class_colors.get(label_id, (255, 255, 255))
                    label = class_names.get(label_id, 'Unknown')

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    current_detected.append(label)

            with lock:
                cv2.imwrite(img_path, frame)
                if set(current_detected) != set(detected_objects):
                    detected_objects.clear()
                    detected_objects.extend(current_detected)
                    print(f'Detected objects updated: {detected_objects}')

        time.sleep(0.1)

# 스레드 생성 (각 모델마다)
thread1 = threading.Thread(target=capture_image_periodically, args=(model1, img_path1, class_names_model1, class_colors_model1, detected_objects_model1))
thread1.daemon = True
thread1.start()

thread2 = threading.Thread(target=capture_image_periodically, args=(model2, img_path2, class_names_model2, class_colors_model2, detected_objects_model2))
thread2.daemon = True
thread2.start()

@app.route('/model1')
def index_model1():
    return render_template('index_test.html')

@app.route('/model2')
def index_model2():
    return render_template('index_test.html')

@app.route('/image_model1', methods=['GET'])
def get_image_model1():
    with lock:
        if os.path.exists(img_path1):
            with open(img_path1, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available for model1.'})

@app.route('/image_model2', methods=['GET'])
def get_image_model2():
    with lock:
        if os.path.exists(img_path2):
            with open(img_path2, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available for model2.'})

@app.route('/status_model1', methods=['GET'])
def get_detection_status_model1():
    return jsonify({'detected_objects': detected_objects_model1})

@app.route('/status_model2', methods=['GET'])
def get_detection_status_model2():
    return jsonify({'detected_objects': detected_objects_model2})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

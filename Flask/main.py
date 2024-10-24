from flask import Flask, jsonify, render_template, Response, request
import cv2
import threading
import os
import time
from ultralytics import YOLO
from flask_cors import CORS
import requests
from flasgger import Swagger

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

model1 = YOLO('abnormal.pt')
model2 = YOLO('growth.pt')

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


status = False # 감지 상태 저장
freeze_status = False # 상태 정보 고정
status_lock = threading.Lock() # 동시 접근 제어를 위한 lock

# 백엔드로 이상감지 정보 전달
def send_notification(class_counts):
    url = "http://3.34.153.235:8080/api/notification/save?userId=1"
    params = {
        'hole': class_counts.get('hole', 0),
        'wither': class_counts.get('wither', 0),
        'timestamp': datetime.now().isoformat()  # 현재 시간을 ISO 형식으로 추가
    }

    headers = {
        'Authorization': 'Bearer YOUR_API_TOKEN'  # 필요하다면 인증 토큰 추가
    }
    
    print(f"Sending request to {url} with params: {params}")

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification: {response.status_code}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def capture_image_periodically(model, img_path, class_names, class_colors):
    global status, freeze_status
    while True:
        camera.grab()
        ret, frame = camera.retrieve()

        if ret:
            results = model(frame)
            detected_counts = {name: 0 for name in class_names.values()}

            # 감지된 객체 처리
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_id = int(box.cls)
                    color = class_colors.get(label_id, (255, 255, 255))
                    label = class_names.get(label_id, 'Unknown')

                    # 감지된 클래스별 개수 증가
                    if label in detected_counts:
                        detected_counts[label] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # freeze_status True일 때 일정 시간 동안 상태를 변경하지 않음
            if freeze_status:
                # print("freeze_status is active, not changing status.")
                time.sleep(100)  # 100초간 상태 고정
                freeze_status = False

            # 감지된 객체가 있는 경우 상태를 True로 변경
            elif any(count > 0 for count in detected_counts.values()):
                with status_lock:
                    status = True
                    # print("Status changed to True due to new detection.")

            # 감지된 객체가 없는 경우 상태를 False로 유지
            else:
                with status_lock:
                    status = False

            # 로그로 상태 출력
            print(f'Status: {status}, Counts: {detected_counts}')

            with lock:
                cv2.imwrite(img_path, frame)

        time.sleep(2) # 나중에 0.1로 바꾸기

# 스레드 생성 (model1에 대한 감지)
thread1 = threading.Thread(target=capture_image_periodically, args=(model1, img_path1, class_names_model1, class_colors_model1))
thread1.daemon = True
thread1.start()

# 스레드 생성 (model2에 대한 감지)
thread2 = threading.Thread(target=capture_image_periodically, args=(model2, img_path2, class_names_model2, class_colors_model2))
thread2.daemon = True
thread2.start()

@app.route('/model1')
def index_model1():
    return render_template('index.html')

@app.route('/model2')
def index_model2():
    return render_template('index.html')

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

# 1) 유림 >> 건우 : model1에서 감지된 박스가 있을 때 status를 true로 보내는 API
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'status': status})



# 2) 건우 >> 유림 : 버튼 클릭 시 status를 false로 바꾸는 API
@app.route('/reset_status', methods=['POST'])
def reset_status():
    global status, freeze_status
    with status_lock:
        status = False
        freeze_status = True  # 100초간 false로 고정
    print(f'Status = {status}')
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, jsonify, render_template, Response
import cv2
import threading
import os
import time
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드 (best.pt 파일 사용)
model = YOLO('best.pt')

# 웹캠 인덱스
camera = cv2.VideoCapture(2)

# 이미지 저장 경로
img_path = './static/captured_image.jpg'

# 파일 접근을 보호하기 위한 Lock 생성
lock = threading.Lock()

# 클래스 이름을 정의 (YOLO 모델에서 사용하는 클래스 이름에 맞게 수정)
class_names = {
    0: 'level_1',  # 클래스 0의 이름
    1: 'level_2',  # 클래스 1의 이름
    2: 'level_3',  # 클래스 2의 이름
}

# 클래스별 박스 개수를 저장할 전역 변수 선언
class_counts = {0: 0, 1: 0, 2: 0}

# 주기적으로 웹캠에서 이미지를 캡처하고 YOLO로 처리하는 함수
def capture_image_periodically():
    global class_counts  # 전역 변수 사용
    while True:
        # 버퍼를 제거하기 위해 grab()을 먼저 호출
        camera.grab()

        # 최신 프레임을 가져오기 위해 retrieve() 호출
        ret, frame = camera.retrieve()
        if ret:
            # YOLO 모델로 객체 감지
            results = model(frame)

            # 클래스별 박스 개수 초기화
            class_counts = {0: 0, 1: 0, 2: 0}

            # 객체 감지된 결과를 이미지에 표시
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표
                    label_id = int(box.cls)  # 클래스 레이블 (int로 변환)
                    
                    # 클래스별 박스 개수 카운트 증가
                    if label_id in class_counts:
                        class_counts[label_id] += 1

                    # 클래스 이름을 가져옴 (클래스 레이블에 해당하는 이름)
                    label = class_names.get(label_id, 'Unknown')

                    # 바운딩 박스와 클래스 이름 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            with lock:
                # 감지된 이미지를 저장
                cv2.imwrite(img_path, frame)

        time.sleep(0.1)  # n초마다 이미지 캡처

# 별도의 스레드로 주기적으로 이미지 캡처 실행
thread = threading.Thread(target=capture_image_periodically)
thread.daemon = True
thread.start()

# 클래스별 박스 개수를 반환하는 API
@app.route('/get_class_counts', methods=['GET'])
def get_class_counts():
    global class_counts  # 전역 변수 사용
    # 클래스 개수를 JSON 형식으로 반환
    return jsonify({
        'level_1': class_counts[0],
        'level_2': class_counts[1],
        'level_3': class_counts[2]
    })

@app.route('/')
def index():
    return render_template('index_test.html')

@app.route('/image', methods=['GET'])
def get_image():
    with lock:
        if os.path.exists(img_path):
            # 이미지 파일을 바이너리로 읽어서 전송
            with open(img_path, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import cv2
from ultralytics import YOLO

def detect_people(video_path):
    """
    Выполняет детекцию людей на видео и сохраняет результат в файл output.mp4.
    Над каждым боксом отображаются имя класса и уровень уверенности.

    Args:
        video_path (str): Путь к видеофайлу, на котором выполняется детекция.
    """
    model = YOLO('yolo11x.pt')
    
    cap = cv2.VideoCapture(video_path)
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame) 

        for result in results:
            boxes = result.boxes.xyxy  # Координаты ббоксов
            confidences = result.boxes.conf  # Уровень уверенности
            classes = result.boxes.cls  # Классы объектов

            # Отрисовка боксов только для людей
            for box, confidence, cls in zip(boxes, confidences, classes):
                if int(cls) == 0:  # Класс "человек"
                    cv2.rectangle(
                        frame, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 
                        2
                    )
                    label = f'Person: {confidence:.2f}'
                    cv2.putText(
                        frame, 
                        label, 
                        (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (255, 0, 0), 
                        2
                    )

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))

        out.write(frame)  # Запись кадра с ббоксами в выходное видео

    cap.release()
    
    if out is not None:
        out.release()

if __name__ == '__main__':
    detect_people('crowd.mp4')

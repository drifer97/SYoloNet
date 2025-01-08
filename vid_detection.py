import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Carregar o modelo YOLO
model = YOLO('yolov8n.pt')

# Caminho do vídeo de entrada e saída
input_video_path = 'E:\\Yolo\\42696-432087106.mp4'
output_video_path = 'E:\Yolo\output video\\ output.avi'

# Abrir o vídeo de entrada
cap = cv2.VideoCapture(input_video_path)

# Obter propriedades do vídeo
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Criar o objeto de escrita de vídeo
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Processar o vídeo frame a frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção no frame atual
    results = model.predict(frame, device='cpu')

    # Anotar o frame com as detecções
    annotated_frame = results[0].plot()

    # Salvar o frame anotado no vídeo de saída
    out.write(annotated_frame)

    # Exibir o frame anotado usando Matplotlib
    plt.imshow(annotated_frame[:, :, ::-1])  # Converte BGR para RGB
    plt.axis('off')  # Remove os eixos
    plt.pause(0.01)  # Pequeno atraso para exibição em tempo real

# Liberar recursos
cap.release()
out.release()
plt.close()

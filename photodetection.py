import matplotlib.pyplot as plt
from ultralytics import YOLO

# Carregar o modelo pré-treinado
model = YOLO('yolov8n.pt')

# Caminho da imagem
image_path = 'E:\\Yolo\\3975553-VWF-GettyImages-1217804090-ab9938daa07f4c5db6c10b982f4ce078.jpg'

# Realizar a detecção
results = model.predict(image_path, device='cpu')

# Exibir o resultado usando Matplotlib
for result in results:
    annotated_image = result.plot()

    # Exibir a imagem
    plt.imshow(annotated_image[:, :, ::-1])  # Converte de BGR para RGB
    plt.axis('off')  # Remove os eixos
    plt.show()

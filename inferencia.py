
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logs_DIR = os.path.join(BASE_DIR, 'logs')

# ==================== PREDICCIÓN ====================
def predecir_imagen(model, img_path, class_names, img_size=(300, 300)):
    """
    Realiza predicción sobre una imagen individual
    """
    # Cargar y preprocesar imagen
    img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predicción
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Predicción: {predicted_class}\nConfianza: {confidence:.2f}%')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(class_names, predictions[0])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Probabilidad')
    plt.title('Distribución de Probabilidades')
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence, predictions[0]

if __name__ == "__main__":

    print("\n==================== PREDICCIÓN DE IMAGEN ====================")
    img_path = input("Ingrese la ruta de la imagen a predecir: ")
    class_names = ""
    model_PATH = os.path.join(logs_DIR, 'modelo_final_maiz.h5')
    model = tf.keras.models.load_model(model_PATH)
    predecir_imagen(model, img_path, class_names)

# Ejemplo de predicción en imagen individual
    # predecir_imagen(model, 'ruta/a/tu/imagen.jpg', class_names)

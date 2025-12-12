import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def crear_generadores_datos(params):
    """
    Crea generadores de datos con aumento de imágenes para entrenamiento
    y normalización para validación/prueba
    """
    # Generador de entrenamiento con data augmentation
    # Generador de prueba
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        params['test_dir'],
        target_size=(params['img_height'], params['img_width']),
        batch_size=params['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    params['num_classes'] = len(test_generator.class_indices)

    return test_generator,test_generator.class_indices
    
def evaluar_modelo(model, test_gen, class_names):
    """Evalúa el modelo en el conjunto de prueba"""
    # Obtener predicciones
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Reporte de clasificación
    print("\n=== REPORTE DE CLASIFICACIÓN ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    return y_true, y_pred, cm, predictions

def plotear_confusion_matrix(cm, class_names):
    """Visualiza la matriz de confusión"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    params ={
    'img_height': 300,
        'img_width': 300,
        'batch_size': 10,
        'epochs': 50,
        'learning_rate': 0.001,
        'num_classes': None,  # Se define automáticamente
        'test_dir': "Ingresar directorio de la"
}
    test_gen, class_indices=crear_generadores_datos(params)
    class_names = list(class_indices.keys())
    # 6. Evaluación
    print("\n[6/8] Evaluando modelo...")
    y_true, y_pred, cm, predictions = evaluar_modelo(model, test_gen, class_names)
    
    # 7. Visualizaciones
    print("\n[7/8] Generando visualizaciones...")
    plotear_confusion_matrix(cm, class_names)
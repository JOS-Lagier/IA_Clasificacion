
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ==================== CONFIGURACIÓN ====================
def configurar_parametros(train_dir,val_dir):
    """Define los parámetros del modelo"""
    params = {
        'img_height': 300,
        'img_width': 300,
        'batch_size': 10,
        'epochs': 50,
        'learning_rate': 0.001,
        'num_classes': None,  # Se define automáticamente
        'train_dir': train_dir,
        'val_dir': val_dir
    }
    return params

# ==================== PREPROCESAMIENTO DE DATOS ====================
def crear_generadores_datos(params):
    """
    Crea generadores de datos con aumento de imágenes para entrenamiento
    y normalización para validación/prueba
    """
    # Generador de entrenamiento con data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Generador de validación (solo normalización)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    
    # Cargar imágenes desde directorios
    train_generator = train_datagen.flow_from_directory(
        params['train_dir'],
        target_size=(params['img_height'], params['img_width']),
        batch_size=params['batch_size'],
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        params['val_dir'],
        target_size=(params['img_height'], params['img_width']),
        batch_size=params['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    # Actualizar número de clases
    params['num_classes'] = len(train_generator.class_indices)
    
    return train_generator, val_generator, train_generator.class_indices

# ==================== CONSTRUCCIÓN DEL MODELO ====================
def construir_modelo(params):
    """
    Construye el modelo usando EfficientNetB3 como base
    con capas personalizadas para clasificación
    """
    # Cargar EfficientNetB3 pre-entrenado en ImageNet
    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(params['img_height'], params['img_width'], 3)
    )
    
    # Congelar las capas base inicialmente
    base_model.trainable = False
    
    # Construir el modelo completo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(params['num_classes'], activation='softmax')
    ])
    
    return model, base_model

# ==================== COMPILACIÓN Y ENTRENAMIENTO ====================
def compilar_modelo(model, learning_rate):
    """Compila el modelo con optimizador y función de pérdida"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model

def crear_callbacks(patience=7):
    """Crea callbacks para mejorar el entrenamiento"""
    callbacks = [
        # Guardar el mejor modelo
        keras.callbacks.ModelCheckpoint(
            'mejor_modelo_maiz.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reducir learning rate cuando no mejore
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

def entrenar_modelo(model, train_gen, val_gen, epochs, callbacks):
    """Entrena el modelo"""
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

def fine_tuning(model, base_model, train_gen, val_gen, epochs_fine, learning_rate_fine):
    """
    Realiza fine-tuning descongelando las últimas capas del modelo base
    """
    # Descongelar las últimas capas de EfficientNetB3
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30  # Últimas 30 capas
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_fine),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Continuar entrenamiento
    callbacks = crear_callbacks(patience=5)
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_fine,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_fine

# ==================== EVALUACIÓN ====================

def plotear_historico_entrenamiento(history):
    """Visualiza las métricas de entrenamiento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('historico_entrenamiento.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== PIPELINE COMPLETO ====================
def ejecutar_pipeline_completo():
    """
    Ejecuta el pipeline completo de entrenamiento
    """
    print("="*50)
    print("SISTEMA DE CLASIFICACIÓN DE PLAGAS EN MAÍZ")
    print("="*50)
    
    # 1. Configuración
    print("\n[1/8] Configurando parámetros...")
    params = configurar_parametros()
    
    # 2. Cargar datos
    print("\n[2/8] Cargando datos...")
    train_gen, val_gen, class_indices = crear_generadores_datos(params)
    class_names = list(class_indices.keys())
    print(f"Clases detectadas: {class_names}")
    print(f"Número de clases: {params['num_classes']}")
    
    # 3. Construir modelo
    print("\n[3/8] Construyendo modelo EfficientNetB3...")
    model, base_model = construir_modelo(params)
    model = compilar_modelo(model, params['learning_rate'])
    print(model.sumgmary())
    
    # 4. Entrenamiento inicial
    print("\n[4/8] Iniciando entrenamiento...")
    callbacks = crear_callbacks()
    history = entrenar_modelo(model, train_gen, val_gen, params['epochs'], callbacks)
    
    # 5. Fine-tuning
    print("\n[5/8] Realizando fine-tuning...")
    history_fine = fine_tuning(model, base_model, train_gen, val_gen, 
                               epochs_fine=20, learning_rate_fine=1e-5)
    
    # 6. Evaluación
    print("\n[6/8] Evaluando modelo...")
    # 7. Visualizaciones
    plotear_historico_entrenamiento(history)
    
    # 8. Guardar modelo final
    print("\n[8/8] Guardando modelo final...")
    model.save('modelo_final_maiz.h5')
    model.save('modelo_final_maiz_savedmodel', save_format='tf')
    
    print("\n" + "="*50)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*50)
    print(f"Modelo guardado en: modelo_final_maiz.h5")
    print(f"Mejor accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, class_names

# ==================== USO DEL SISTEMA ====================
if __name__ == "__main__":
    """
    Estructura de carpetas esperada:
    
    data/
    ├── train/
    │   ├── sana/
    │   ├── plaga_1/
    │   ├── plaga_2/
    │   └── ...
    ├── validation/
    │   ├── sana/
    │   ├── plaga_1/
    │   └── ...
    └── test/
        ├── sana/
        ├── plaga_1/
        └── ...
    """
    
    # Ejecutar pipeline completo
    model, class_names = ejecutar_pipeline_completo()
    
    # Ejemplo de predicción en imagen individual
    # predecir_imagen(model, 'ruta/a/tu/imagen.jpg', class_names)

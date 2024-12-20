from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Configuración inicial
st.title("Clasificador de Dígitos - MNIST con Interfaz Interactiva")
st.sidebar.header("Configuración de Entrenamiento")

# Parámetros configurables
epochs = st.sidebar.slider("Número de épocas", min_value=1, max_value=20, value=5, step=1)
BATCHSIZE = 32

# Carga de datos
@st.cache_data
def load_data():
    dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    num_train_examples = int(metadata.splits['train'].num_examples)
    num_test_examples = int(metadata.splits['test'].num_examples)
    return list(train_dataset), list(test_dataset), num_train_examples, num_test_examples

train_dataset, test_dataset, num_train_examples, num_test_examples = load_data()

# Normalización
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).map(normalize).shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).map(normalize).batch(BATCHSIZE)

# Definición del modelo
@st.cache_resource
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Entrenamiento
if st.sidebar.button("Entrenar Modelo"):
    st.write(f"Entrenando el modelo con {epochs} épocas...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE),
        validation_data=test_dataset,
        validation_steps=math.ceil(num_test_examples / BATCHSIZE)
    )
    # Graficar resultados
    st.write("### Gráficas de Entrenamiento")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfica de pérdida
    ax[0].plot(history.history['loss'], label='Pérdida de entrenamiento')
    ax[0].plot(history.history['val_loss'], label='Pérdida de validación')
    ax[0].set_title('Pérdida')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Pérdida')
    ax[0].legend()
    
    # Gráfica de precisión
    ax[1].plot(history.history['accuracy'], label='Precisión de entrenamiento')
    ax[1].plot(history.history['val_accuracy'], label='Precisión de validación')
    ax[1].set_title('Precisión')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Precisión')
    ax[1].legend()

    st.pyplot(fig)

# Dibujo de dígito
st.write("### Dibujar un Dígito")
st.write("Dibuja un número (0-9) en el lienzo y haz clic en **Predecir**.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Procesar la imagen dibujada
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'), mode='L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Realizar la predicción
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        st.write(f"### El modelo predijo: {predicted_digit}")
        st.bar_chart(prediction[0])
    else:
        st.write("Por favor, dibuja un número antes de predecir.")

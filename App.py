import os
import streamlit as st
import base64
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import matplotlib.pyplot as plt

# Función de predicción de dígitos
def predict_digit(img: Image.Image):
    # Cargar modelo solo una vez
    model = tf.keras.models.load_model('model/handwritten.h5')
    # Convertir a escala de grises
    gray = img.convert('L')
    # Invertir para que fondo sea blanco y trazo negro
    inverted = ImageOps.invert(gray)
    # Redimensionar a 28x28
    resized = inverted.resize((28, 28))
    # Normalizar
    arr = np.array(resized, dtype='float32') / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    # Predecir\ n    pred = model.predict(arr)
    return int(np.argmax(pred[0]))

# Configuración de la página\ nst.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')

# Inicializar estado
def init_state():
    if 'digits' not in st.session_state:
        st.session_state.digits = []
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
init_state()

# Título
st.title('✍️ Reconocimiento de Dígitos Escritos a Mano')
st.subheader('Dibuja un dígito, presiona Predecir y continúa')

# Controles de canvas
st.sidebar.title('Opciones')
st.sidebar.markdown('Ancho de línea')
stroke_width = st.sidebar.slider('', 1, 30, 15)

# Crear canvas con key dinámico para limpiar tras predecir
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=200,
    width=200,
    drawing_mode='freedraw',
    key=f'canvas_{st.session_state.canvas_key}'
)

# Botón de predecir
if st.button('🔍 Predecir'):
    img_data = canvas_result.image_data
    if img_data is None:
        st.warning('Por favor dibuja un dígito antes de predecir')
    else:
        # Convertir array de canvas a imagen PIL
        arr = (img_data[:, :, 0:3] * 255).astype('uint8')
        img = Image.fromarray(arr)
        # Predecir con corrección de inversión
        digit = predict_digit(img)
        # Guardar dígito
        st.session_state.digits.append(digit)
        # Incrementar canvas_key para refrescar canvas
        st.session_state.canvas_key += 1

# Mostrar resultados acumulados
if st.session_state.digits:
    st.subheader('Historial de dígitos')
    st.write(st.session_state.digits)
    total = sum(st.session_state.digits)
    st.subheader(f'🔢 Suma acumulada: {total}')

# Sidebar info
st.sidebar.title('Acerca de')
st.sidebar.write('App basada en TensorFlow y Streamlit Canvas')
st.sidebar.write('Modelo de ejemplo: handwritten.h5')

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
    model = tf.keras.models.load_model('model/handwritten.h5')
    gray = ImageOps.grayscale(img)
    resized = gray.resize((28, 28))
    arr = np.array(resized, dtype='float32') / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    pred = model.predict(arr)
    return int(np.argmax(pred[0]))

# Configuración de la página
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')

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
def on_predict():
    img_data = canvas_result.image_data
    if img_data is None:
        st.warning('Por favor dibuja un dígito antes de predecir')
        return
    # Convertir y predecir
    arr = (img_data * 255).astype('uint8')
    img = Image.fromarray(arr).convert('RGB')
    digit = predict_digit(img)
    # Guardar dígito y actualizar key para limpiar canvas
    st.session_state.digits.append(digit)
    st.session_state.canvas_key += 1
    st.experimental_rerun()

if st.button('🔍 Predecir'):
    on_predict()

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

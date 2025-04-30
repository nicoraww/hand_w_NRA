import os
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import matplotlib.pyplot as plt

# Función de predicción de dígitos
def predict_digit(img: Image.Image):
    model = tf.keras.models.load_model('model/handwritten.h5')
    gray = img.convert('L')
    inverted = ImageOps.invert(gray)
    resized = inverted.resize((28, 28))
    arr = np.array(resized, dtype='float32') / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    pred = model.predict(arr)
    return int(np.argmax(pred[0]))

# Funciones para mensajes de cada dígito
DIGIT_FACTS = {
    0: "Cero es el único número que NO tiene valor posicional.",
    1: "Uno es el número multiplicativo neutro y símbolo de unidad.",
    2: "Dos es el primer número primo y el único par primo.",
    3: "Tres es un número triangular y comúnmente asociado a la tríada.",
    4: "Son 4 estaciones en el año, las vacas tienen 4 patas, 4 es número de la suerte en Japón.",
    5: "Cinco sentidos tenemos los humanos y 5 dedos en cada mano.",
    6: "Seis caras tiene un cubo y 6 cuerdas una guitarra estándar.",
    7: "Siete días tiene la semana y siete maravillas del mundo clásico.",
    8: "Ocho es el número atemporal infinito en posición horizontal.",
    9: "Nueve planetas hubo en el sistema solar antes de la redefinición en 2006."
}

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

# Crear canvas
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
        arr = (img_data[:, :, 0:3] * 255).astype('uint8')
        img = Image.fromarray(arr)
        digit = predict_digit(img)
        st.session_state.digits.append(digit)
        # Mostrar mensaje relevante para el dígito
        fact = DIGIT_FACTS.get(digit, '')
        st.info(f'**Dígito {digit}:** {fact}')
        # Incrementar canvas_key para limpio
        st.session_state.canvas_key += 1

# Mostrar historial y suma
if st.session_state.digits:
    st.subheader('Historial de dígitos ingresados')
    st.write(st.session_state.digits)
    total = sum(st.session_state.digits)
    st.subheader(f'🔢 Suma acumulada: {total}')

# Sidebar info
st.sidebar.title('Acerca de')
st.sidebar.write('App basada en TensorFlow y Streamlit Canvas')
st.sidebar.write('Modelo de ejemplo: handwritten.h5')

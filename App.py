import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    # Carga del modelo
    model = tf.keras.models.load_model("model/handwritten.h5")
    # Conversión a escala de grises y redimensionamiento
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = img_array.reshape((1,28,28,1))
    # Predicción
    pred = model.predict(img_array)
    result = int(np.argmax(pred[0]))
    return result

# Datos curiosos por dígito
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

# Streamlit UI
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Parámetros de dibujo
drawing_mode = 'freedraw'
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Canvas\ ncanvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Botón de predicción
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        # Preparar imagen para predicción
        array_data = np.array(canvas_result.image_data[:, :, 0:3])
        input_image = Image.fromarray(array_data.astype('uint8'), 'RGB')
        # Ejecutar predicción
        digit = predictDigit(input_image)
        # Mostrar resultado y dato curioso
        st.header(f'El dígito es: {digit}')
        fact = DIGIT_FACTS.get(digit, 'No hay datos curiosos para este dígito.')
        st.info(fact)
    else:
        st.warning('Por favor dibuja en el canvas antes de predecir.')

# Sidebar\ nst.sidebar.title("Acerca de:")
st.sidebar.text("Esta aplicación evalúa la capacidad de un RNA")
st.sidebar.text("para reconocer dígitos escritos a mano.")
st.sidebar.text("Basado en el desarrollo de Vinay Uniyal")

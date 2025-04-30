import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Función de predicción de dígitos
def predictDigit(image):
    # Carga el modelo entrenado
    model = tf.keras.models.load_model("model/handwritten.h5")
    # Convierte a escala de grises y redimensiona
    gray = ImageOps.grayscale(image)
    img = gray.resize((28, 28))
    # Normaliza pixel a [0,1]
    arr = np.array(img, dtype='float32') / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    # Predice y devuelve dígito
    pred = model.predict(arr)
    return int(np.argmax(pred[0]))

# Datos curiosos de cada dígito
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

# Configuración de Streamlit
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')
st.title('✍️ Reconocimiento de Dígitos Escritos a Mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Parámetros del canvas
drawing_mode = 'freedraw'
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Creación del canvas
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode=drawing_mode,
    key='canvas'
)

# Botón de predicción
i f st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        # Convierte datos del canvas a imagen PIL
        array_data = (canvas_result.image_data[:, :, :3] * 255).astype('uint8')
        input_image = Image.fromarray(array_data)
        # Predicción
        digit = predictDigit(input_image)
        # Muestra resultado y dato curioso
        st.header(f'El dígito es: {digit}')
        fact = DIGIT_FACTS.get(digit, 'No hay dato curioso para este dígito.')
        st.info(fact)
    else:
        st.warning('Por favor dibuja un dígito antes de predecir.')

# Barra lateral
st.sidebar.title('Acerca de')
st.sidebar.write('Esta aplicación evalúa la capacidad de un RNA para reconocer dígitos escritos a mano.')
st.sidebar.write('Basado en el desarrollo de Vinay Uniyal.')

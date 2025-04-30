import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Función de predicción de dígitos
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    # Convertir a escala de grises
gray = ImageOps.grayscale(image)
    # Redimensionar a 28x28
img = gray.resize((28, 28))
    # Normalizar\arr = np.array(img, dtype='float32') / 255.0
arr = arr.reshape((1, 28, 28, 1))
    # Predecir
pred = model.predict(arr)
return int(np.argmax(pred[0]))

# Datos curiosos para cada dígito
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
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Slider para el ancho de línea
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)

# Canvas para dibujo
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Color de fondo del trazo
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',  # Color del trazo
    background_color='#000000',  # Color de fondo del canvas
    height=200,
    width=200,
    drawing_mode='freedraw',
    key='canvas'
)

# Botón de predicción
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        # Convertir datos a imagen PIL
        array_data = (canvas_result.image_data[:, :, :3] * 255).astype('uint8')
        input_image = Image.fromarray(array_data)
        # Predecir dígito
        digit = predictDigit(input_image)
        # Mostrar resultado y dato curioso
        st.header(f'El dígito es: {digit}')
        fact = DIGIT_FACTS.get(digit, 'No hay dato curioso para este dígito.')
        st.info(fact)
    else:
        st.warning('Por favor dibuja un dígito antes de predecir.')

# Sidebar informativo
st.sidebar.title('Acerca de')
st.sidebar.write('Esta aplicación evalúa la capacidad de un modelo')
st.sidebar.write('para reconocer dígitos escritos a mano.')
st.sidebar.write('Basado en el desarrollo de Vinay Uniyal.')

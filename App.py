import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Función de predicción de dígitos
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return int(result)

# Mensajes para cada dígito
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

# Streamlit
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')
st.title('✍️ Reconocimiento de Dígitos Escritos a Mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Slider para el ancho de línea
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)

# Canvas de dibujo
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=200,
    width=200,
    drawing_mode='freedraw',
    key='canvas'
)

# Botón de predicción
if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        # Convertir los datos del canvas a imagen PIL
        array_data = (canvas_result.image_data[:, :, :3] * 255).astype('uint8')
        input_image = Image.fromarray(array_data)
        # Predecir
        digit = predictDigit(input_image)
        # Mostrar dato curioso
        fact = DIGIT_FACTS.get(digit, "No se encontró información para este dígito.")
        st.info(f'**Dígito {digit}:** {fact}')
        # Limpiar canvas recargando la app
        st.experimental_rerun()
    else:
        st.warning('Por favor dibuja un dígito antes de predecir.')

# Sidebar informativo
st.sidebar.title("Acerca de")
st.sidebar.write("Esta aplicación evalúa la capacidad de un modelo de TensorFlow")
st.sidebar.write("para reconocer dígitos escritos a mano usando Streamlit Canvas.")
st.sidebar.write("Basado en el desarrollo de Vinay Uniyal.")

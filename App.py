import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Configuración de la página (debe ser el primer comando de Streamlit)
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

# Carga del modelo con st.cache_resource
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

# Función para predecir un dígito
def predictDigit(image):
    # Convierte la imagen a escala de grises y redimensiona
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    return int(np.argmax(pred[0]))

# Título y subtítulo de la app
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja dos dígitos en los paneles y presiona 'Predecir' if you dare")

# Parámetros de dibujo
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # Color del trazo (blanco)
bg_color = '#000000'      # Color de fondo (negro)

# Disposición de canvases y símbolo +
col1, col2, col3 = st.columns([1, 0.2, 1])
with col1:
    canvas1 = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas1"
    )
with col2:
    st.markdown("<h1 style='text-align: center; margin-top: 60px;'>+</h1>", unsafe_allow_html=True)
with col3:
    canvas2 = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas2"
    )

# Botón de predicción
if st.button('Predecir'):
    if canvas1.image_data is not None and canvas2.image_data is not None:
        # Convierte a imagen PIL
        img1 = Image.fromarray(canvas1.image_data.astype('uint8'), 'RGBA')
        img2 = Image.fromarray(canvas2.image_data.astype('uint8'), 'RGBA')
        # Predicciones
        res1 = predictDigit(img1)
        res2 = predictDigit(img2)
        suma = res1 + res2
        # Mostrar resultados
        st.header(f'Primer dígito: {res1}')
        st.header(f'Segundo dígito: {res2}')
        st.header(f'Suma: {res1} + {res2} = {suma}')
    else:
        st.header('Por favor dibuja en ambos canvas los dígitos.')

# Barra lateral
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalúa")
st.sidebar.text("la capacidad de un RNA de reconocer")
st.sidebar.text("dos dígitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")

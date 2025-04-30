import os
import streamlit as st
from openai import OpenAI
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')

# Sidebar: API Key para OpenAI
api_key = st.sidebar.text_input('🔑 Ingresa tu OpenAI API Key', type='password')
client = OpenAI(api_key=api_key) if api_key else None

# Estilos para animar emojis
st.markdown("""
<style>
  @keyframes floatEmoji {
    0% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0); }
  }
  .emoji-anim {
    display: inline-block;
    animation: floatEmoji 2s ease-in-out infinite;
    font-size: 2rem;
    margin: 0.1rem;
  }
</style>
""", unsafe_allow_html=True)

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

# Función de predicción de dígitos
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

def predict_digit(image: Image.Image) -> int:
    model = load_model()
    gray = ImageOps.grayscale(image)
    img = gray.resize((28, 28))
    arr = np.array(img, dtype='float32') / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    pred = model.predict(arr)
    return int(np.argmax(pred[0]))

# Interfaz de usuario
st.title('✍️ Reconocimiento de Dígitos Escritos a Mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Parámetros del canvas
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'
\ n# Componente de dibujo
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key='canvas'
)

# Predicción y presentación de datos
if st.button('🔍 Predecir'):
    if canvas_result and canvas_result.image_data is not None:
        img_array = (canvas_result.image_data[:, :, :3] * 255).astype('uint8')
        input_image = Image.fromarray(img_array)
        digit = predict_digit(input_image)

        # Mostrar resultado y dato curioso
        st.header(f'El dígito es: {digit}')
        st.info(DIGIT_FACTS.get(digit, 'No hay dato curioso para este dígito.'))

        # Mostrar emojis animados
        emojis_html = ''.join(["<span class='emoji-anim'>🎉</span>" for _ in range(digit)])
        st.markdown(f"<div>{emojis_html}</div>", unsafe_allow_html=True)

        # Generar breve historia con ChatGPT
        if client:
            prompt = f"Crea una breve historia en español basada en estos emojis: {'🎉'*digit}."
            resp = client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role':'user','content':prompt}]
            )
            story = resp.choices[0].message.content
            st.subheader('📖 Historia:')
            st.write(story)
    else:
        st.warning('Por favor dibuja un dígito antes de predecir.')

# Barra late

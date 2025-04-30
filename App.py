import os
import streamlit as st
from openai import OpenAI
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Configuración de Streamlit
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')

# Entrada de API Key para ChatGPT
ostr_api_key = st.sidebar.text_input('🔑 Ingresa tu OpenAI API Key', type='password')
if ostr_api_key:
            # Inicializar cliente de OpenAI
            client = OpenAI(api_key=ostr_api_key)
            prompt = f"Crea una breve historia en español basada en estos emojis: {'🎉'*digit}."
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role':'user','content':prompt}]
            )(
                model='gpt-4o',
                messages=[{'role':'user','content':prompt}]
            )
            story = response.choices[0].message.content
            st.subheader('📖 Historia:')
            st.write(story)
    else:
        st.warning('Por favor dibuja un dígito antes de predecir.')

# Barra lateral
st.sidebar.title('Acerca de')
st.sidebar.write('Modelo de ejemplo: handwritten.h5')
st.sidebar.write('Desarrollado con TensorFlow, Streamlit Canvas y ChatGPT.')

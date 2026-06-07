import streamlit as st
import subprocess
import sys
import os

# Tenta instalar o tensorflow via comando de sistema antes de importar
def instalar_tf():
    try:
        import tensorflow as tf
    except ImportError:
        st.warning("Instalando motor de IA (aguarde)...")
        # Tenta a versão estável, se falhar, tenta a nightly
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
        except:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-nightly"])
        import tensorflow as tf
    return tf

st.set_page_config(page_title="VeroRad", page_icon="🦴", layout="centered")
st.title("🦴 VeroRad")

# Força a instalação antes de carregar o resto
tf = instalar_tf()

import numpy as np
import requests
from PIL import Image
from streamlit_paste_button import paste_image_button
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# ... resto do seu código ...

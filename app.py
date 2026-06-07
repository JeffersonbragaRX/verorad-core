import streamlit as st
import subprocess
import sys

# Instalação dinâmica se o ambiente falhar
def verificar_instalacao():
    try:
        import tensorflow
        import streamlit_paste_button
    except ImportError:
        st.warning("Configurando ambiente... aguarde.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.16.1", "streamlit-paste-button"])
        st.rerun()

verificar_instalacao()

import numpy as np
import requests
import os
from PIL import Image
from streamlit_paste_button import paste_image_button
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Conv2D, LocallyConnected2D, Dense, Dropout, GlobalAveragePooling2D, multiply, Lambda, BatchNormalization
from tensorflow.keras.models import Model

# --- CONFIGURAÇÃO ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_h3QRlUhrYIaVMFaC6WrT0304ciqaGoL"
MODEL_PATH = "bone_age_weights.best.hdf5"

def baixar_modelo():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("A baixar pesos da IA..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

@st.cache_resource
def carregar_ia():
    baixar_modelo()
    in_lay = Input(shape=(384, 384, 3))
    base = VGG16(input_shape=(384, 384, 3), include_top=False, weights=None)
    pt = base(in_lay)
    bn = BatchNormalization()(pt)
    attn = Conv2D(64, 1, activation='relu')(bn)
    attn = Conv2D(16, 1, activation='relu')(attn)
    attn = LocallyConnected2D(1, 1, activation='sigmoid')(attn)
    
    # Placeholder para alinhar tensores
    up_c2 = Conv2D(512, 1, activation='linear', use_bias=False)
    attn = up_c2(attn)
    
    mask = multiply([attn, bn])
    gap = GlobalAveragePooling2D()(mask)
    gap_m = GlobalAveragePooling2D()(attn)
    gap = Lambda(lambda x: x[0]/x[1])([gap, gap_m])
    dr = Dropout(0.5)(gap)
    out = Dense(1, activation='linear')(Dropout(0.25)(Dense(1024, activation='elu')(dr)))
    
    m = Model(inputs=[in_lay], outputs=[out])
    m.load_weights(MODEL_PATH)
    return m

st.title("🦴 VeroRad")
modelo_ia = carregar_ia()
paste_result = paste_image_button(label="📋 Colar Raio-X")

if paste_result.image_data:
    img = Image.open(paste_image_button.image_data).convert('RGB')
    st.image(img)
    if st.button("Analisar"):
        img_arr = np.expand_dims(np.array(img.resize((384, 384))), axis=0)
        idade = float(modelo_ia.predict(preprocess_input(img_arr))[0][0])
        st.success(f"Idade: {int(idade//12)} anos e {int(idade%12)} meses.")

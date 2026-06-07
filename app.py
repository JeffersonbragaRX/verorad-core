import streamlit as st
import numpy as np
import requests
import os
from PIL import Image
import tensorflow as tf
from streamlit_paste_button import paste_image_button
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Conv2D, LocallyConnected2D, Dense, Dropout, GlobalAveragePooling2D, multiply, Lambda, BatchNormalization
from tensorflow.keras.models import Model

# --- CONFIGURAÇÃO ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_h3QRlUhrYIaVMFaC6WrT0304ciqaGoL"
MODEL_PATH = "bone_age_weights.best.hdf5"

@st.cache_resource
def carregar_ia():
    # 1. Baixar modelo se não existir
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    
    # 2. Montar arquitetura da rede
    in_lay = Input(shape=(384, 384, 3))
    base = VGG16(input_shape=(384, 384, 3), include_top=False, weights=None)
    pt = base(in_lay)
    bn = BatchNormalization()(pt)
    attn = Conv2D(64, 1, activation='relu')(bn)
    attn = Conv2D(16, 1, activation='relu')(attn)
    attn = LocallyConnected2D(1, 1, activation='sigmoid')(attn)
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

# --- INTERFACE ---
st.set_page_config(page_title="VeroRad", page_icon="🦴", layout="centered")
st.title("🦴 VeroRad")
st.subheader("Inteligência Artificial para Radiologia")

try:
    modelo_ia = carregar_ia()
    
    # Botão de colar ou upload
    paste_result = paste_image_button(label="📋 Colar Raio-X", background_color="#0066cc")
    upload = st.file_uploader("Ou envie o ficheiro:", type=["png", "jpg", "jpeg"])
    
    # Prioridade para o botão de colar
    img_data = paste_result.image_data if (paste_result and paste_result.image_data) else upload

    if img_data:
        img = Image.open(img_data).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("Analisar Imagem"):
            with st.spinner("Analisando..."):
                img_arr = np.expand_dims(np.array(img.resize((384, 384))), axis=0)
                idade = float(modelo_ia.predict(preprocess_input(img_arr), verbose=0)[0][0])
                st.success(f"Idade óssea: {int(idade//12)} anos e {int(idade%12)} meses.")
            
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

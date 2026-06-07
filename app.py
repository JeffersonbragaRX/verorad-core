import streamlit as st
import numpy as np
import requests
import os
import tensorflow as tf
from PIL import Image
from streamlit_paste_button import paste_image_button
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Conv2D, LocallyConnected2D, Dense, Dropout, GlobalAveragePooling2D, multiply, Lambda, BatchNormalization
from tensorflow.keras.models import Model

# --- CONFIGURAÇÃO ---
# Link direto do seu modelo no Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_h3QRlUhrYIaVMFaC6WrT0304ciqaGoL"
MODEL_PATH = "bone_age_weights.best.hdf5"

def baixar_modelo():
    """Baixa o modelo do Drive se ele não existir no servidor."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("VeroRad: Inicializando motor de diagnóstico..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

@st.cache_resource
def carregar_ia():
    """Monta a arquitetura da IA e carrega os pesos."""
    baixar_modelo()
    
    # Arquitetura do Modelo
    in_lay = Input(shape=(384, 384, 3))
    base_pretrained_model = VGG16(input_shape=(384, 384, 3), include_top=False, weights=None)
    pt_features = base_pretrained_model(in_lay)
    
    # Atenção e processamento
    bn_features = BatchNormalization()(pt_features)
    attn_layer = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)
    attn_layer = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn_layer)
    attn_layer = LocallyConnected2D(1, kernel_size=(1,1), padding='valid', activation='sigmoid')(attn_layer)
    
    up_c2_w = np.ones((1, 1, 1, 512))
    up_c2 = Conv2D(512, kernel_size=(1,1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    
    gap = Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.5)(gap)
    dr_steps = Dropout(0.25)(Dense(1024, activation='elu')(gap_dr))
    out_layer = Dense(1, activation='linear')(dr_steps)
    
    modelo = Model(inputs=[in_lay], outputs=[out_layer])
    modelo.load_weights(MODEL_PATH)
    return modelo

# --- INTERFACE ---
st.set_page_config(page_title="VeroRad", page_icon="🦴", layout="centered")
st.title("🦴 VeroRad")
st.subheader("Inteligência Artificial para Radiologia")

# Carregamento do modelo
with st.spinner("Preparando o ambiente VeroRad..."):
    modelo_ia = carregar_ia()

# Entrada de imagem
paste_result = paste_image_button(label="📋 Colar Raio-X", background_color="#0066cc")
uploaded_file = st.file_uploader("Ou escolha uma imagem:", type=["png", "jpg", "jpeg"])

imagem_analisada = paste_result.image_data if paste_result.image_data else uploaded_file

if imagem_analisada:
    img = Image.open(imagem_analisada).convert('RGB')
    st.image(img, use_container_width=True)
    
    if st.button("Analisar Imagem"):
        img_arr = np.expand_dims(np.array(img.resize((384, 384))), axis=0)
        idade_meses = float(modelo_ia.predict(preprocess_input(img_arr), verbose=0)[0][0])
        
        anos = int(idade_meses // 12)
        meses = int(idade_meses % 12)
        
        st.success(f"Resultado estimado: {anos} anos e {meses} meses.")
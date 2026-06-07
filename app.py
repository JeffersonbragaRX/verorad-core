import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import streamlit as st
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Conv2D, LocallyConnected2D, Dense, Dropout, GlobalAveragePooling2D, multiply, Lambda, BatchNormalization
from tensorflow.keras.models import Model

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="VeroRad | Inteligência em Diagnóstico", page_icon="🦴", layout="centered")

# O caminho agora é apenas o nome do ficheiro, pois ele estará na mesma pasta na nuvem
CAMINHO_DO_MODELO = "bone_age_weights.best.hdf5"

@st.cache_resource
def carregar_ia():
    in_lay = Input(shape=(384, 384, 3))
    base_pretrained_model = VGG16(input_shape=(384, 384, 3), include_top=False, weights=None)
    pt_features = base_pretrained_model(in_lay)
    
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
    modelo.load_weights(CAMINHO_DO_MODELO)
    return modelo

# --- INTERFACE VERORAD ---
st.title("VeroRad")
st.subheader("A verdade diagnóstica, com a precisão da inteligência.")

with st.spinner("A inicializar o núcleo VeroRad..."):
    try:
        modelo_ia = carregar_ia()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo HDF5: {e}")

# Botão mágico de colar imagem
colar_resultado = paste_image_button(
    label="📋 Colar Raio-X da área de transferência",
    background_color="#0066cc",
    hover_background_color="#005bb5"
)

# Verifica se a imagem veio do botão de colar ou do upload tradicional
imagem_analisada = colar_resultado.image_data if colar_resultado.image_data else st.file_uploader("Ou envie o ficheiro (JPG/PNG):", type=["png", "jpg", "jpeg"])

if imagem_analisada:
    img = imagem_analisada.convert('RGB')
    st.image(img, use_container_width=True, caption="Imagem capturada com sucesso")
    
    if st.button("Analisar com VeroRad", type="primary"):
        with st.spinner("A calcular idade óssea com IA..."):
            img_arr = np.expand_dims(np.array(img.resize((384, 384))), axis=0)
            resultado = modelo_ia.predict(preprocess_input(img_arr), verbose=0)
            idade = float(resultado[0][0])
            
            anos = int(idade // 12)
            meses = int(idade % 12)
            
            st.success(f"### 🎯 Resultado VeroRad: **{anos} anos e {meses} meses**")
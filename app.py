import streamlit as st
import numpy as np
import requests
import os
import onnxruntime as ort
from PIL import Image
from streamlit_paste_button import paste_image_button

# --- CONFIGURAÇÃO ---
# ATENÇÃO: Você deve converter seu modelo .hdf5 para .onnx e subir esse arquivo para o drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=SEU_ID_DO_MODELO_ONNX"
MODEL_PATH = "bone_age_model.onnx"

@st.cache_resource
def carregar_ia():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("A baixar modelo ONNX..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    
    # Inicia a sessão do ONNX Runtime
    return ort.InferenceSession(MODEL_PATH)

st.set_page_config(page_title="VeroRad", page_icon="🦴", layout="centered")
st.title("🦴 VeroRad (ONNX)")

try:
    session = carregar_ia()
    paste_result = paste_image_button(label="📋 Colar Raio-X")
    upload = st.file_uploader("Ou envie o ficheiro:", type=["png", "jpg", "jpeg"])
    
    img_data = paste_result.image_data if (paste_result and paste_result.image_data) else upload

    if img_data:
        img = Image.open(img_data).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("Analisar Imagem"):
            with st.spinner("Analisando..."):
                # Preprocessamento (deve ser idêntico ao usado no treinamento)
                img_arr = np.array(img.resize((384, 384))).astype(np.float32)
                img_arr = np.expand_dims(img_arr, axis=0) / 255.0
                
                # Executa a inferência
                input_name = session.get_inputs()[0].name
                resultado = session.run(None, {input_name: img_arr})
                
                idade = float(resultado[0][0][0])
                st.success(f"Idade óssea: {int(idade//12)} anos e {int(idade%12)} meses.")
except Exception as e:
    st.error(f"Erro ao carregar o modelo ONNX: {e}")

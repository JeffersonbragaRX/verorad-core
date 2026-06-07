import streamlit as st
import numpy as np
import requests
import os
import onnxruntime as ort
from PIL import Image
from streamlit_paste_button import paste_image_button

# --- CONFIGURAÇÃO ---
# URL com formato de download direto via Google Drive
MODEL_URL = "https://docs.google.com/uc?export=download&id=1NYeHyK6Rg9v9paePeyraKCC3dKML6qWr"
MODEL_PATH = "bone_age_model.onnx"

@st.cache_resource
def carregar_ia():
    # Verifica se o modelo já existe localmente
    if not os.path.exists(MODEL_PATH):
        # A mensagem de status é apresentada fora do fluxo de renderização principal
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code != 200:
            raise Exception(f"Erro ao baixar modelo: Código {response.status_code}")
        
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return ort.InferenceSession(MODEL_PATH)

# Configuração da página para evitar conflitos de renderização no DOM
st.set_page_config(page_title="VeroRad", page_icon="🦴", layout="centered")

st.title("🦴 VeroRad (Análise de Idade Óssea)")

try:
    session = carregar_ia()
    
    # Utilizamos o botão de paste de forma simplificada
    paste_result = paste_image_button(label="📋 Colar Raio-X")
    upload = st.file_uploader("Ou envie o ficheiro:", type=["png", "jpg", "jpeg"])
    
    img_data = paste_result.image_data if (paste_result and paste_result.image_data) else upload

    if img_data:
        img = Image.open(img_data).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("Analisar Imagem"):
            # Processamento direto sem spinner complexo para evitar erros de nó do DOM
            img_arr = np.array(img.resize((384, 384))).astype(np.float32)
            img_arr = np.expand_dims(img_arr, axis=0) / 255.0
            
            input_name = session.get_inputs()[0].name
            resultado = session.run(None, {input_name: img_arr})
            
            idade = float(resultado[0][0][0])
            st.success(f"Idade óssea estimada: {int(idade//12)} anos e {int(idade%12)} meses.")
            
except Exception as e:
    st.error(f"Erro no sistema: {e}")

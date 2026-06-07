import streamlit as st
import numpy as np
import requests
import os
import onnxruntime as ort
from PIL import Image
from streamlit_paste_button import paste_image_button

# --- CONFIGURAÇÃO ---
# URL com formato de download direto
MODEL_URL = "https://docs.google.com/uc?export=download&id=1NYeHyK6Rg9v9paePeyraKCC3dKML6qWr"
MODEL_PATH = "bone_age_model.onnx"

@st.cache_resource
def carregar_ia():
    # Verifica se o modelo já existe localmente
    if not os.path.exists(MODEL_PATH):
        with st.spinner("A baixar modelo (primeira vez)..."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code != 200:
                raise Exception(f"Erro ao baixar modelo: Código {response.status_code}")
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    # Validação mínima de segurança para garantir que não é um HTML de erro
    if os.path.getsize(MODEL_PATH) < 100000:
        raise Exception("O ficheiro baixado é muito pequeno. Verifique se o link do Drive está público.")

    return ort.InferenceSession(MODEL_PATH)

st.set_page_config(page_title="VeroRad", page_icon="🦴", layout="centered")
st.title("🦴 VeroRad (Análise de Idade Óssea)")

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
                # Preprocessamento (deve casar com a entrada que definiu no converter.py)
                img_arr = np.array(img.resize((384, 384))).astype(np.float32)
                img_arr = np.expand_dims(img_arr, axis=0)
                
                # Inferência
                input_name = session.get_inputs()[0].name
                resultado = session.run(None, {input_name: img_arr})
                
                idade = float(resultado[0][0][0])
                st.success(f"Idade óssea estimada: {int(idade//12)} anos e {int(idade%12)} meses.")
except Exception as e:
    st.error(f"Erro no sistema: {e}")
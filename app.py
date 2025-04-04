# Requiere: streamlit, pillow, torch, torchvision, numpy, opencv-python-headless
import streamlit as st
import tempfile
import os
from PIL import Image
import zipfile
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as T
import numpy as np
import cv2

# Función para cargar modelo dummy directamente sin archivo externo
@st.cache_resource
def load_model(weights_path="canopy_height_model.pth"):
    from canopy_model import CanopyHeightNet
    import urllib.request, os

    if not os.path.exists(weights_path):
        url = "https://nativas-climatech.nyc3.digitaloceanspaces.com/canopy_height_model.pth"
        urllib.request.urlretrieve(url, weights_path)

    model = CanopyHeightNet()
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Procesar imagen y predecir altura

def predict_canopy_height(model, pil_image):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    input_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)[0].squeeze().numpy()  # <- Asegura que sea 2D
    return output  # Mapa 2D de altura en metros

st.set_page_config(page_title="Canopy Height Map Tool", layout="wide")
st.title("🌲 Canopy Height Map Generator (modelo dummy funcional)")

uploaded_file = st.file_uploader("Subí una imagen satelital o archivo KMZ", type=["jpg", "jpeg", "png", "kmz"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_dir = tempfile.mkdtemp()

    model = load_model()

    def process_and_predict(pil_img):
        st.image(pil_img, caption="Imagen subida", use_container_width=True)
        st.markdown("**Generando mapa estimado de altura del dosel...**")
        height_map = predict_canopy_height(model, pil_img)

        if height_map is not None and height_map.size > 0:
            norm_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
            norm_map = np.nan_to_num(norm_map).astype(np.uint8)
            if norm_map.ndim == 2:
                color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                st.image(color_map, caption="Mapa generado (altura del dosel simulada)", use_container_width=True)
            else:
                st.error("Error: el mapa de altura no tiene el formato esperado (2D).")
        else:
            st.error("Error: no se pudo generar el mapa de altura.")

    if suffix in [".jpg", ".jpeg", ".png"]:
        img = Image.open(uploaded_file).convert("RGB")
        process_and_predict(img)

    elif suffix == ".kmz":
        kmz_path = os.path.join(temp_dir, uploaded_file.name)
        with open(kmz_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            kmz.extractall(temp_dir)
            st.markdown("### Archivos extraídos:")
            for name in kmz.namelist():
                st.write("-", name)

        kml_file = [f for f in os.listdir(temp_dir) if f.endswith('.kml')]
        image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if kml_file:
            kml_path = os.path.join(temp_dir, kml_file[0])
            tree = ET.parse(kml_path)
            root = tree.getroot()
            st.success("Archivo KMZ procesado exitosamente. Extraído: " + kml_file[0])

            names = [elem.text for elem in root.iter() if 'name' in elem.tag and elem.text]
            if names:
                st.markdown("### Elementos encontrados en el KML:")
                for n in names:
                    st.write("- " + n)

            if image_files:
                overlay_path = os.path.join(temp_dir, image_files[0])
                overlay_img = Image.open(overlay_path).convert("RGB")
                process_and_predict(overlay_img)

        else:
            st.error("No se encontró archivo KML en el KMZ subido.")

    st.markdown("---")
    st.markdown("Esta herramienta utiliza un modelo simple para simular la altura del dosel. Pronto se integrará el modelo real de Meta (HighResCanopyHeight).")

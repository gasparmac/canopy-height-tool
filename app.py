# Requiere: streamlit, pillow, simplekml, xml
import streamlit as st
import tempfile
import os
from PIL import Image
import zipfile
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Canopy Height Map Tool", layout="wide")
st.title("🌲 Canopy Height Map Generator")

uploaded_file = st.file_uploader("Subí una imagen satelital o archivo KMZ", type=["jpg", "jpeg", "png", "kmz"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_dir = tempfile.mkdtemp()

    if suffix in [".jpg", ".jpeg", ".png"]:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen subida", use_column_width=True)
        st.markdown("**Simulando predicción de altura del dosel...**")

        # Simulación rápida del resultado (en reemplazo del modelo real de Meta)
        canopy_image = img.convert("L")  # Simulación con escala de grises
        st.image(canopy_image, caption="Mapa simulado de altura del dosel", use_column_width=True)

    elif suffix == ".kmz":
        kmz_path = os.path.join(temp_dir, uploaded_file.name)
        with open(kmz_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with zipfile.ZipFile(kmz_path, 'r') as kmz:
            kmz.extractall(temp_dir)

        kml_file = [f for f in os.listdir(temp_dir) if f.endswith('.kml')]
        image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if kml_file:
            kml_path = os.path.join(temp_dir, kml_file[0])
            tree = ET.parse(kml_path)
            root = tree.getroot()
            st.success("Archivo KMZ procesado exitosamente. Extraído: " + kml_file[0])

            # Mostrar nombres de elementos (si existen)
            names = [elem.text for elem in root.iter() if 'name' in elem.tag and elem.text]
            if names:
                st.markdown("### Elementos encontrados en el KML:")
                for n in names:
                    st.write("- " + n)

            # Mostrar imagen del overlay
            if image_files:
                overlay_path = os.path.join(temp_dir, image_files[0])
                overlay_img = Image.open(overlay_path)
                st.image(overlay_img, caption="Imagen del Overlay (extraída del KMZ)", use_column_width=True)

                st.markdown("**Simulando predicción de altura del dosel...**")
                canopy_overlay = overlay_img.convert("L")
                st.image(canopy_overlay, caption="Mapa simulado de altura del dosel (overlay)", use_column_width=True)

        else:
            st.error("No se encontró archivo KML en el KMZ subido.")

    st.markdown("---")
    st.markdown("Esta es una herramienta demo. Para implementar el modelo real de altura del dosel de Meta, integrá su red neuronal disponible en GitHub: [facebookresearch/HighResCanopyHeight](https://github.com/facebookresearch/HighResCanopyHeight)")
    

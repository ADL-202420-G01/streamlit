import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("cnn_model.h5")
label_map = {0:"cat",1:"Dog"}

def preprocess_iamge(image):
    image = image.resize((128,128))
    image = np.array(image) / 255
    image = np.expand_dims(image,axis=0)
    return image

st.set_page_config(
    page_title="Detección de predios",  # Título de la pestaña
    page_icon="🏡",  # Puedes usar un emoji o una ruta a una imagen como favicon
    layout="centered",  # Opciones: "centered" o "wide"
    initial_sidebar_state="expanded"  # Opciones: "expanded", "collapsed", "auto"
)

st.write("Este es un modelo de detección de imágenes que localiza **predios** y **no predios**.")

# **Texto permanente**
st.header("Localización de construcciones con imágenes satelitales")
st.image("label.jpg")

st.markdown(
    """
    ## Problemática
    La insuficiencia en los registros de bienes inmuebles en zonas rurales es un problema recurrente en muchos países, especialmente en aquellos donde el acceso a tecnología y servicios gubernamentales puede ser limitado en áreas remotas. Este problema tiene varias causas y consecuencias, así como posibles estrategias para su resolución.

    ## Soluciones previas
    - Metodologías de gran costo y demandante en tiempo (solución por encuestas presenciales).
    
    ## Metodologías aplicadas
    - Uso de modelo generativos para crear imágenes a partir de un set limitado de información.
    - Uso de redes preentrenadas **U-NET** para la identificación de construcciones en zonas rurales.

    ## ¿Cómo usar esta aplicación?
    1. Se puede seleccionar una imagen que tenga 512x512 pixeles de tamaño.
    2. El modelo analizará la imagen y predecirá si existe o no un predio.

    ## Ejecución de modelo de *deep learning*
    - Elija imagen...
    """
)

upload_image = st.file_uploader("",type=["jpg","png"])

if upload_image is not None:
    image = Image.open(upload_image)
    st.image(image,caption="Imagen cargada",use_column_width=True)

    image_preprocessed = preprocess_iamge(image=image)

    prediction = model.predict(image_preprocessed)
    predicted_class = np.argmax(prediction)
    predicted_label = label_map[predicted_class]

    st.write(f"Predicción: {predicted_label}")
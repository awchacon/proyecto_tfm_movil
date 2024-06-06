from sys import stderr
import pickle
import numpy as np
import streamlit as st
import pandas as pd 
import time 
from streamlit_option_menu import option_menu
import plotly
import plotly.express as px

#import webbrowser
import streamlit as st
#from UI import *
#import os

from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

#import geopandas as gpd
#import json
import streamlit.components.v1 as components
import base64

#import replicate

import streamlit as st
from PIL import Image
import tensorflow as tf

#configuramos el title de nuestra pagina -- streamlit run .\app.py

st.set_page_config(page_title="DeepCarVision", page_icon="static/images/logo.png", layout="wide")

st.write("")
st.markdown("<h1 style='text-align: center; font-size: 110px; color: #210a93; font-style: italic;'>DeepCarVision</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
selected = option_menu(
    menu_title=None,
    options=["Inicio", "Imagen", "DB Coches", "Inspeccion", "Predecir Precio"],  # ,"Mapa","Chatbot"
    icons=['house', 'book', 'book', 'book', 'book', 'map'],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#000000"},
        "icon": {"color": "Blue", "font-size": "25px"},
        "nav-link": {"font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#1248BA"},
    }
)

theme_plotly = None

# Declaracion de funciones

# Funcion Home Page
def HomePage():
    file_ = open("static/images/cars_gif.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<div class="centered-content"><img src="data:image/gif;base64,{data_url}" alt="main gif"></div>',
        unsafe_allow_html=True,
    )

    # Cargar el archivo CSS
    with open("static/styles/styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def Coches():
    try: 
        st.sidebar.header("Por favor filtre aquí:")
        but_alq, but_com = st.sidebar.columns(2)

        Accion = ['Comprar', 'Vender']
        filtro_accion = st.sidebar.selectbox('Elija una opción:', Accion, index=0)

        if filtro_accion == 'Comprar':
            # Importo los datos de alquiler
            df = pd.read_csv('data/bd_final.csv', sep=',', header=0)
        elif filtro_accion == 'Vender':
            # Importo los datos de compras
            df = pd.read_csv('data/bd_final.csv', sep=',', header=0)

        # Convertir posibles valores nulos en la columna 'Carrocería' a 'Sin especificar'
        df['Carrocería'] = df['Carrocería'].fillna('Sin especificar')

        df_marca_unique = df['Marca'].unique()
        df_cambio_unique = df['Cambio'].unique()
        df_distintivo_unique = df['Distintivo'].unique()
        df_carroceria_unique = df['Carrocería'].unique()

        Marca = st.sidebar.multiselect('Marca de coche:', df_marca_unique, default=df_marca_unique)
        Cambio = st.sidebar.multiselect('Cambio:', df_cambio_unique, default=df_cambio_unique)
        Distintivo = st.sidebar.multiselect('Distintivo:', df_distintivo_unique, default=df_distintivo_unique)
        Carroceria = st.sidebar.multiselect('Carrocería:', df_carroceria_unique, default=df_carroceria_unique)

        # Verificar si alguna lista está vacía y establecer un valor predeterminado para evitar errores en query
        if not Marca:
            Marca = df_marca_unique.tolist()
        if not Cambio:
            Cambio = df_cambio_unique.tolist()
        if not Distintivo:
            Distintivo = df_distintivo_unique.tolist()
        if not Carroceria:
            Carroceria = df_carroceria_unique.tolist()

        # Convertir listas a cadenas compatibles con la consulta
        Marca_str = '", "'.join(Marca)
        Cambio_str = '", "'.join(Cambio)
        Distintivo_str = '", "'.join(Distintivo)
        Carroceria_str = '", "'.join(Carroceria)

        # Aplicar filtros usando query
        query_str = f'Marca in ["{Marca_str}"] & Cambio in ["{Cambio_str}"] & Distintivo in ["{Distintivo_str}"] & Carrocería in ["{Carroceria_str}"]'
        df_selection = df.query(query_str)

        # Mostrar solo las columnas seleccionadas del DataFrame
        with st.expander("💵 Mi database de Coches en Madrid 🚗"):
            default_columns = ["Marca", "Cambio", "Codigo", "Serie", "Distintivo", "Carrocería", "Kilómetros"]
            shwdata = st.multiselect('Filtro :', df.columns.tolist(), default=default_columns)
            st.dataframe(df_selection[shwdata], use_container_width=True)

        # Crear la gráfica de barras
        Marca_Cambio = (
            df_selection.groupby('Cambio')['Marca']
            .value_counts()
            .groupby(level=0, group_keys=False)
            .nlargest(5)
            .reset_index(name='count')
        )

        fig_Marca_Cambio = px.bar(
            Marca_Cambio,
            x='Cambio',
            y='count',
            color='Marca',
            barmode='group', 
            title='Top Tipos de Marcas por caja de Cambios'
        )

        fig_Marca_Cambio.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            template="plotly_white",
        )

        # Crear la gráfica de línea
        Carroceria_kilometros = df_selection.groupby('Carrocería')['Kilómetros'].mean().nlargest(5).reset_index()
        fig_state = px.line(
            Carroceria_kilometros,
            x='Carrocería',
            y='Kilómetros',
            title="Top Mayor kilometraje por carrocerías",
            template="plotly_white"
        )

        fig_state.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickmode="linear"),
            yaxis=dict(showgrid=False)
        )

        # Crear la gráfica de pastel
        fig_pie = px.pie(
            df_selection,
            values='Precio',
            names='Cambio',
            title='Cambio por Precio'
        )
        fig_pie.update_layout(
            legend_title="Cambio",
            legend_y=0.9
        )
        fig_pie.update_traces(
            textinfo='percent+label',
            textposition='inside'
        )

        # Mostrar las gráficas en columnas
        left_column, right_column, center_column = st.columns(3)
        left_column.plotly_chart(fig_state, use_container_width=True)
        right_column.plotly_chart(fig_Marca_Cambio, use_container_width=True)
        center_column.plotly_chart(fig_pie, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")


# Función para mostrar Inspeccion
def Inspeccion():
    but_alq, but_com = st.sidebar.columns(2)

    Accion = ['Comprar', 'Vender']
    filtro_accion = st.sidebar.selectbox('Elegir Acción:', Accion, index=0)

    if filtro_accion == 'Comprar':
        df = pd.read_csv('data/bd_final.csv', sep=',', header=0)
    elif filtro_accion == 'Vender':
        df = pd.read_csv('data/bd_final.csv', sep=',', header=0)

    # Defino la función que muestra las imágenes.
    def main():
        try:
            image_urls = list(df['Imagen'][(df['Precio'] >= filtro_Precio[0]) & (df['Precio'] <= filtro_Precio[1]) &
                                           (df['Marca'] == filtro_Marca) & (df['Cambio'] == filtro_Cambio) &
                                           (df['Carrocería'] == filtro_Carroceria)])
            
            datos = df[['Serie', 'Precio', 'Imagen', 'Extra', 'Carrocería','Kilómetros','Distintivo','Potencia','Año']][(df['Precio'] >= filtro_Precio[0]) & (df['Precio'] <= filtro_Precio[1]) &
                                                                              (df['Marca'] == filtro_Marca) & (df['Cambio'] == filtro_Cambio) &
                                                                              (df['Carrocería'] == filtro_Carroceria)]
            datos.reset_index(inplace=True, drop=True)

            if 'current_image_index' not in st.session_state:
                st.session_state.current_image_index = 0
            
            current_image_index = st.session_state.current_image_index

            st.image(image_urls[current_image_index], use_column_width=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            if col1.button("◄ Anterior"):
                st.session_state.current_image_index = (current_image_index - 1) % len(image_urls)
            if col3.button("Siguiente ►"):
                st.session_state.current_image_index = (current_image_index + 1) % len(image_urls)

            col2.write(f'{current_image_index} de {len(image_urls)-1}')

            st.write(f"Marca:  {filtro_Marca}")
            st.write(f"Serie:  {datos['Serie'][current_image_index]}")
            st.write(f"Carrocería:  {datos['Carrocería'][current_image_index]}")
            st.write(f"Potencia:  {datos['Potencia'][current_image_index]}")
            st.write(f"Descripción de Extras:  {datos['Extra'][current_image_index]}")
            st.write(f"Precio:  {datos['Precio'][current_image_index]} €")
            st.write(f"Cambio:  {filtro_Cambio}")
            st.write(f"Kilómetros:  {datos['Kilómetros'][current_image_index]}")
            st.write(f"Distintivo:  {datos['Distintivo'][current_image_index]}")
            st.write(f"Año:  {datos['Año'][current_image_index]}")
            st.write(f"Url :  {datos['Imagen'][current_image_index]}")


        except Exception as e:
            st.error('NO HAY VEHÍCULOS QUE COINCIDAN CON LOS FILTROS APLICADOS')
            st.error(f"Error: {e}")

    if filtro_accion: 
        if filtro_accion == 'Vender':
            st.title('Coches en venta')
            min_precio = 1950
            max_precio = 209900
        elif filtro_accion == 'Comprar':
            st.title('Coches en Compra')
            min_precio = 1950
            max_precio = 209900

        st.write('Seleccione los filtros deseados para que le muestre aquellos coches que cumplen dichos filtros.')

        st.sidebar.header('Filtros')

        filtro_Marca = st.sidebar.selectbox('Filtrar por marca de coche:', sorted(df['Marca'].unique()))
        filtro_Cambio = st.sidebar.selectbox('Filtrar por Cambio:', sorted(df['Cambio'].unique()))
        filtro_Carroceria = st.sidebar.selectbox('Filtrar por Carrocería:', sorted(df['Carrocería'].unique()))
        filtro_Precio = st.sidebar.slider('Filtrar por precio:', min_value=min_precio, max_value=max_precio, value=(min_precio, max_precio))

        if st.sidebar.button('Aplicar'):
            data_filtrada = df[['Marca', 'Cambio', 'Precio', 'Carrocería']][
                (df['Precio'] >= filtro_Precio[0]) & (df['Precio'] <= filtro_Precio[1]) &
                (df['Marca'] == filtro_Marca) & (df['Cambio'] == filtro_Cambio) & (df['Carrocería'] == filtro_Carroceria)
            ]
            data_filtrada.reset_index(inplace=True, drop=True)
            st.session_state.current_image_index = 0

        try:
            st.table(data_filtrada.head())
        except:
            st.table(df[['Marca', 'Cambio', 'Precio', 'Carrocería','Distintivo']].head())

    return main()


########## Imagen ##########

# Función para cargar el modelo de TensorFlow (esto puede tardar unos segundos)
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model('model/modelo_tfm_best.keras')
#     return model

# # Función para preprocesar la imagen
# def preprocess_image(image):
#     image = tf.image.resize(image, [224, 224])
#     image = tf.cast(image, tf.float32) / 255.0
#     return image

# # Función para mostrar la imagen y las predicciones
# def show_foto_predictions(predictions, class_labels):
#     st.write("Classification Probabilities:")
#     for label, probability in predictions.items():
#         st.write(f"{class_labels[label]}: {probability:.4f}")

# # Definir las etiquetas de clase
# class_labels = ['Abarth', 'Audi', 'BMW', 'Citroen', 'Hyundai', 'Peugeot', 'Seat', 'Toyota']

# # Cargar el modelo
# model = load_model()

# # Función para cargar el logo de la marca
# def cargar_logo(marca):
#     logos = {
#         'CITROEN': 'static/logo/citroen.png',
#         'PEUGEOT': 'static/logo/peugeot.png',
#         'AUDI': 'static/logo/audi.png',
#         'TOYOTA': 'static/logo/toyota.png',
#         'BMW': 'static/logo/bmw.png',
#         'SEAT': 'static/logo/seat.png',
#         'HYUNDAI': 'static/logo/hyundai.png'
#     }
#     return logos.get(marca.upper(), 'path/to/default_logo.png')

# Función para cargar y clasificar la imagen
def Imagen():
    # st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    
    # # Subir imagen
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption='Uploaded Image.', use_column_width=True)
    #     st.write("")
    #     st.write("Classifying...")
    #     with st.spinner('Wait for it...'):
    #         time.sleep(3)
    #         image_array = np.array(image)
    #         image_preprocessed = preprocess_image(image_array)
    #         image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

    #         predictions = model.predict(image_preprocessed)
    #         top_7 = predictions[0].argsort()[-7:][::-1]  # Assuming you have 7 classes

    #         top_7_labels = {i: predictions[0][i] for i in top_7}

    #         # Mostrar el logo y las predicciones en columnas
    #         col1, col2 = st.columns([1, 2])
            
    #         with col1:
    #             logo_path = cargar_logo(class_labels[top_7[0]])
    #             logo_image = Image.open(logo_path)
    #             st.image(logo_image, width=300)
            
    #         with col2:
    #             show_foto_predictions(top_7_labels, class_labels)
    
    st.markdown('</div>', unsafe_allow_html=True)


###### Cargar el footer  ######

#Pestañas de pagina
if selected =="Inicio":
    HomePage()


if selected == "DB Coches":
    df_selection = Coches()
    try:
        st.markdown("""---""") #Diferencia entre dos
        total_tipo_coche = float(df_selection['Marca'].count())
        coche_mode = df_selection['Marca'].mode().iloc[0]

        total1,total2 = st.columns(2,gap='large')
        with total1:
            st.info('Total de habitantes en Madrid 👫', icon="🔍")
            st.metric(label = 'nº habitantes', value= f"{total_tipo_coche:,.0f}")

        with total2:
            st.info('Tipo de coche que mas se encuentra 🏘', icon="🔍")
            st.metric(label='Tipo coche', value=f"{coche_mode}")
    except:
            st.warning("Escoja entre Comprar o Vender ")


if selected == "Inspeccion":
    Inspeccion()

if selected == "Imagen":    
    Imagen()

elif selected == "Predecir Precio":
    # Cargar el modelo RandomForestRegressor y el encoder
    with open("data/coche_sale_buy/mejor_modelo_rf.pkl", "rb") as file:
        mejor_rf = pickle.load(file)

    with open("data/coche_sale_buy/ordinal_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    # Columnas categóricas y numéricas
    categorical_cols = ['Marca', 'Combustible', 'Distintivo', 'Carrocería', 'Cambio']
    numerical_cols = ['Año', 'Kilómetros', 'Potencia']
    column_features = categorical_cols + numerical_cols

    # Función para predecir el precio del coche
    def predecir_precio(coche):
        # Convertir el diccionario a un DataFrame
        coche_df = pd.DataFrame([coche])
        
        # Codificar columnas categóricas
        coche_df[categorical_cols] = encoder.transform(coche_df[categorical_cols])
        
        # Asegurarse de que las columnas estén en el mismo orden que las columnas de entrenamiento del modelo
        coche_df = coche_df[column_features]
        
        # Realizar la predicción
        precio_predicho = mejor_rf.predict(coche_df)
        
        return precio_predicho[0]

    # Función para cargar la imagen del logo
    def cargar_logo(marca):
        logos = {
            'CITROEN': 'static/logo/citroen.png',
            'PEUGEOT': 'static/logo/peugeot.png',
            'AUDI': 'static/logo/audi.png',
            'TOYOTA': 'static/logo/toyota.png',
            'BMW': 'static/logo/bmw.png',
            'SEAT': 'static/logo/seat.png',
            'HYUNDAI': 'static/logo/hyundai.png'
        }
        return logos.get(marca, 'path/to/default_logo.png')

    # Función para la interfaz de usuario
    def main():
        if "precio_predicho" not in st.session_state:
            st.session_state.precio_predicho = None

        if st.session_state.precio_predicho is None:
            st.markdown("<h2 style='text-align: center; font-size:24px;'>Predicción de Precio de Coches</h2>", unsafe_allow_html=True)

        # Interfaz para ingresar datos del coche
        st.sidebar.header("Ingrese los datos del coche:")
        marca = st.sidebar.selectbox("Marca:", ['CITROEN', 'PEUGEOT', 'AUDI', 'TOYOTA', 'BMW', 'SEAT', 'HYUNDAI'])
        año = st.sidebar.slider("Año:", 2017, 2023)
        combustible = st.sidebar.selectbox("Combustible:", ['Gasolina', 'Diesel', 'Híbrido'])
        distintivo = st.sidebar.selectbox("Distintivo:", ['C', 'ECO', 'B', '0 EMISIONES'])
        carroceria = st.sidebar.selectbox("Carrocería:", ['Berlina', 'Todo Terreno', 'Stationwagon', 'Monovolumen', 'Coupe', 'Convertible'])
        kilometros = st.sidebar.number_input("Kilómetros:", min_value=0, value=0)
        cambio = st.sidebar.selectbox("Cambio:", ['Manual', 'Automático'])
        potencia = st.sidebar.number_input("Potencia:", min_value=0, value=100)

        coche = {
            'Marca': marca,
            'Año': año,
            'Combustible': combustible,
            'Distintivo': distintivo,
            'Carrocería': carroceria,
            'Kilómetros': kilometros,
            'Cambio': cambio,
            'Potencia': potencia
        }

        # Realizar la predicción cuando se hace clic en el botón
        if st.sidebar.button("Predecir Precio"):
            precio_predicho = predecir_precio(coche)
            precio_formateado = f"{precio_predicho:,.2f}".replace(',', ' ')
            st.markdown(f"<h2 style='text-align: center; color: green;'>El precio predicho para el coche es: {precio_formateado} € euros</h2>", unsafe_allow_html=True)
            
            
             # Mostrar el logo del coche y las características en columnas
            col1, col2 = st.columns([1, 2])
            
            with col1:
                logo_path = cargar_logo(marca)
                logo = Image.open(logo_path)
                st.image(logo, width=400)
            
            with col2:
                st.write(f"**Marca:** {marca}")
                st.write(f"**Año:** {año}")
                st.write(f"**Combustible:** {combustible}")
                st.write(f"**Distintivo:** {distintivo}")
                st.write(f"**Carrocería:** {carroceria}")
                st.write(f"**Kilómetros:** {kilometros}")
                st.write(f"**Cambio:** {cambio}")
                st.write(f"**Potencia:** {potencia}")

    main()

footer = """
<style>
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    height: 7%;
    bottom: 0;
    width: 100%;
    background-color: #243946;
    color: black;
    text-align: center;
}

.footer img {
    width: 50px; /* ajusta el tamaño según sea necesario */
    height: auto;
    margin: 10px; /* ajusta el margen según sea necesario */
}

.footer p {
    margin: 10px; /* ajusta el margen según sea necesario */
}

.social-icons {
    display: inline-block;
}
</style>
<div class="footer">
    <div class="social-icons">
        <p>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Instagram_logo_2022.svg/1200px-Instagram_logo_2022.svg.png" alt="Instagram" title="Instagram">
            Síguenos en <a href="url_de_tu_perfil_de_instagram" target="_blank">Instagram</a>
        </p>
    </div>
    <div class="social-icons">
        <p>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/2021_Facebook_icon.svg/220px-2021_Facebook_icon.svg.png" alt="Facebook" title="Facebook">
            Síguenos en <a href="url_de_tu_perfil_de_facebook" target="_blank">Facebook</a>
        </p>
    </div>
    <div class="social-icons">
        <p>
            <img src="https://store-images.s-microsoft.com/image/apps.31120.9007199266245564.44dc7699-748d-4c34-ba5e-d04eb48f7960.bc4172bd-63f0-455a-9acd-5457f44e4473" alt="LinkedIn" title="LinkedIn">
            Síguenos en <a href="url_de_tu_perfil_de_linkedin" target="_blank">LinkedIn</a>
        </p>
    </div>
    <p>Creador 🤖 por <a href="www.linkedin.com/in/aaron-chacon" target="_blank">Los de atras</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
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

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="DeepCarVision", page_icon="static/images/logo.png", layout="centered")

# Cargar el archivo CSS
def load_css():
    with open("static/styles/styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Título principal
st.markdown("<h2 class='center-text'>DeepCarVision</h2>", unsafe_allow_html=True)
st.write("")

# Menú de opciones
selected = st.selectbox(
    "Menu",
    ["Inicio", "Imagen", "DB Coches", "Inspeccion", "Predecir Precio"]
)

# Función HomePage
def HomePage():
    file_ = open("static/images/cars_gif.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<div class="centered-content"><img src="data:image/gif;base64,{data_url}" alt="main gif"></div>',
        unsafe_allow_html=True,
    )

def Coches():
    try:
        st.header("Por favor filtre aquí:")
        Accion = ['Comprar', 'Vender']
        filtro_accion = st.selectbox('Elija una opción:', Accion, index=0)

        if filtro_accion == 'Comprar':
            df = pd.read_csv('data/bd_final.csv', sep=',', header=0)
        elif filtro_accion == 'Vender':
            df = pd.read_csv('data/bd_final.csv', sep=',', header=0)

        df['Carrocería'] = df['Carrocería'].fillna('Sin especificar')
        df_marca_unique = df['Marca'].unique()
        df_cambio_unique = df['Cambio'].unique()
        df_distintivo_unique = df['Distintivo'].unique()
        df_carroceria_unique = df['Carrocería'].unique()

        Marca = st.multiselect('Marca de coche:', df_marca_unique, default=df_marca_unique)
        Cambio = st.multiselect('Cambio:', df_cambio_unique, default=df_cambio_unique)
        Distintivo = st.multiselect('Distintivo:', df_distintivo_unique, default=df_distintivo_unique)
        Carroceria = st.multiselect('Carrocería:', df_carroceria_unique, default=df_carroceria_unique)

        if not Marca:
            Marca = df_marca_unique.tolist()
        if not Cambio:
            Cambio = df_cambio_unique.tolist()
        if not Distintivo:
            Distintivo = df_distintivo_unique.tolist()
        if not Carroceria:
            Carroceria = df_carroceria_unique.tolist()

        Marca_str = '", "'.join(Marca)
        Cambio_str = '", "'.join(Cambio)
        Distintivo_str = '", "'.join(Distintivo)
        Carroceria_str = '", "'.join(Carroceria)

        query_str = f'Marca in ["{Marca_str}"] & Cambio in ["{Cambio_str}"] & Distintivo in ["{Distintivo_str}"] & Carrocería in ["{Carroceria_str}"]'
        df_selection = df.query(query_str)

        with st.expander("💵 Mi database de Coches en Madrid 🚗"):
            default_columns = ["Marca", "Cambio", "Codigo", "Serie", "Distintivo", "Carrocería", "Kilómetros"]
            shwdata = st.multiselect('Filtro :', df.columns.tolist(), default=default_columns)
            st.dataframe(df_selection[shwdata], use_container_width=True)

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

        st.plotly_chart(fig_state, use_container_width=True)
        st.plotly_chart(fig_Marca_Cambio, use_container_width=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

def Inspeccion():
    Accion = ['Comprar', 'Vender']
    filtro_accion = st.selectbox('Elegir Acción:', Accion, index=0)

    if filtro_accion == 'Comprar':
        df = pd.read_csv('data/bd_final.csv', sep=',', header=0)
    elif filtro_accion == 'Vender':
        df = pd.read_csv('data/bd_final.csv', sep=',', header=0)

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

        st.header('Filtros')

        filtro_Marca = st.selectbox('Filtrar por marca de coche:', sorted(df['Marca'].unique()))
        filtro_Cambio = st.selectbox('Filtrar por Cambio:', sorted(df['Cambio'].unique()))
        filtro_Carroceria = st.selectbox('Filtrar por Carrocería:', sorted(df['Carrocería'].unique()))
        filtro_Precio = st.slider('Filtrar por precio:', min_value=min_precio, max_value=max_precio, value=(min_precio, max_precio))

        if st.button('Aplicar'):
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

def Imagen():
    st.markdown('</div>', unsafe_allow_html=True)

if selected =="Inicio":
    HomePage()

if selected == "DB Coches":
    df_selection = Coches()
    try:
        st.markdown("""---""")
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
    with open("data/coche_sale_buy/mejor_modelo_rf.pkl", "rb") as file:
        mejor_rf = pickle.load(file)

    with open("data/coche_sale_buy/ordinal_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    categorical_cols = ['Marca', 'Combustible', 'Distintivo', 'Carrocería', 'Cambio']
    numerical_cols = ['Año', 'Kilómetros', 'Potencia']
    column_features = categorical_cols + numerical_cols

    def predecir_precio(coche):
        coche_df = pd.DataFrame([coche])
        coche_df[categorical_cols] = encoder.transform(coche_df[categorical_cols])
        coche_df = coche_df[column_features]
        precio_predicho = mejor_rf.predict(coche_df)
        return precio_predicho[0]

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

    def main():
        if "precio_predicho" not in st.session_state:
            st.session_state.precio_predicho = None

        if st.session_state.precio_predicho is None:
            st.markdown("<h2 style='text-align: center; font-size:24px;'>Predicción de Precio de Coches</h2>", unsafe_allow_html=True)

        st.header("Ingrese los datos del coche:")
        marca = st.selectbox("Marca:", ['CITROEN', 'PEUGEOT', 'AUDI', 'TOYOTA', 'BMW', 'SEAT', 'HYUNDAI'])
        año = st.slider("Año:", 2017, 2023)
        combustible = st.selectbox("Combustible:", ['Gasolina', 'Diesel', 'Híbrido'])
        distintivo = st.selectbox("Distintivo:", ['C', 'ECO', 'B', '0 EMISIONES'])
        carroceria = st.selectbox("Carrocería:", ['Berlina', 'Todo Terreno', 'Stationwagon', 'Monovolumen', 'Coupe', 'Convertible'])
        kilometros = st.number_input("Kilómetros:", min_value=0, value=0)
        cambio = st.selectbox("Cambio:", ['Manual', 'Automático'])
        potencia = st.number_input("Potencia:", min_value=0, value=100)

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

        if st.button("Predecir Precio"):
            precio_predicho = predecir_precio(coche)
            precio_formateado = f"{precio_predicho:,.2f}".replace(',', ' ')
            st.markdown(f"<h2 style='text-align: center; color: green;'>El precio predicho para el coche es: <br>{precio_formateado} € euros</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                logo_path = cargar_logo(marca)
                logo = Image.open(logo_path)
                st.image(logo, width=200)
            
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
    position: relative;
    left: 0;
    height: auto;
    bottom: 0;
    width: 100%;
    background-color: #243946;
    color: black;
    text-align: center;
    padding: 10px 0;
}

.footer img {
    width: 20px;
    height: auto;
    margin: 3px;
}

.footer p {
    margin: 3px;
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
    <p>Creador 🤖 por <a href="www.linkedin.com/in/aaron-chacon" target="_blank">DeepCarVision</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)


import streamlit as st
from streamlit_option_menu import option_menu
import warnings
import openai
from transformers import pipeline
import torch
import requests
import PyPDF2
import pdfplumber
import PyPDF2
from vosk import Model, KaldiRecognizer
import wave
import json
import tempfile
import base64


warnings.filterwarnings('ignore')

HUGGINGFACE_API_KEY = "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"

# URL del modelo GPT-Neo en Hugging Face
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"  # Modelo GPT-Neo

# Función para resumir texto usando la API de Hugging Face
def summarize_text_with_huggingface(text):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
    }
    # Dividir el texto en fragmentos manejables
    max_chunk_size = 500
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    summary = ""
    for chunk in chunks:
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/pegasus-xsum",  # Cambia al modelo PEGASUS para resumen
            headers=headers,
            json={"inputs": chunk}
        )

        # Manejo detallado de la respuesta
        if response.status_code == 200:
            result = response.json()
            # Verifica si la respuesta es una lista con texto generado
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'summary_text' in result[0]:
                summary += result[0]['summary_text'] + " "
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
                summary += result[0] + " "
            else:
                st.error(f"Formato inesperado en la respuesta: {result}")
                break
        elif response.status_code == 503:
            st.error("El modelo está sobrecargado o no disponible temporalmente. Intenta más tarde.")
            break
        else:
            st.error(f"Error al conectarse a la API de Hugging Face: {response.status_code} - {response.text}")
            break

    return summary

# Configuración de la página
st.set_page_config(
    page_title="LuMI",
    page_icon=":school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS para diseño personalizado del sidebar
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #1E1E1E; /* Fondo gris oscuro */
        color: white;
    }
    .sidebar .element-container { 
        background-color: #1E1E1E; /* Asegura que el fondo sea uniforme */
    }
    .sidebar-title {
        font-size: 48px; /* Tamaño más grande para LuMI */
        color: black; /* Color blanco para el título */
        margin-bottom: 20px;
        font-weight: bold;
        text-align: center;
    }
    .css-1d391kg { /* Cambia el color del botón seleccionado */
        background-color: #333333 !important; /* Fondo gris más oscuro */
        color: white !important;
    }
    .css-1d391kg:hover {
        background-color: #444444 !important; /* Gris oscuro al pasar el ratón */
    }
    </style>
""", unsafe_allow_html=True)

# Inicialización del estado para la navegación interna
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(page_name):
    st.session_state.page = page_name

# Barra lateral con menú de opciones
with st.sidebar:
    st.markdown("<h1 class='sidebar-title'>LuMI</h1>", unsafe_allow_html=True)
    st.markdown("---") 
    st.write("Acerca de")
    st.write("User Name")
    st.markdown("---")
    
    # Menú de opciones con los nombres e iconos correctos
    selected = option_menu(
        menu_title=None,
        options=["Inicio", "Cursos", "Mensajes", "Calificaciones", "Herramientas"],
        icons=["globe", "journal-bookmark-fill", "people-fill", "envelope-fill", "file-earmark-text-fill", "tools"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#FFF"},  # Fondo gris oscuro uniforme
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#444444"},
            "nav-link-selected": {"background-color": "#333333"},  # Gris más oscuro para botón seleccionado
        }
    )

    # Espacio y botón de cerrar sesión
    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
    st.button("Cerrar sesión")
# Contenido de cada página según selección
if selected == "Inicio":
    st.title("LuMI")
    st.write("Aquí puedes ver todas las actividades recientes y programadas.")

    # Sección de tarjetas y contenido
    col1, col2 = st.columns([3, 3])

    # Contenido y descripción en la primera columna
    with col1:
        st.markdown("""
        <style>
        .info-section {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .info-section h2 {
            font-size: 24px;
            color: #FF6F61;
            margin-bottom: 10px;
        }
        .info-section ul {
            margin: 0;
            padding-left: 20px;
        }
        .info-section ul li {
            margin-bottom: 10px;
            line-height: 1.5;
        }
        </style>
        
        <div class="info-section">
            <h2>IA en la Educación</h2>
            <ul>
                <li>Aprendizaje personalizado y adaptado.</li>
                <li>Atiende las necesidades individuales de los estudiantes, ofreciendo una educación verdaderamente personalizada e individualizada.</li>
                <li>Proporciona un esquema de aprendizaje personalizado para cada estudiante.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.image("https://www.earthlingsecurity.com/wp-content/uploads/2018/09/educational-institution.png", caption="IA en la Educación")
    # Tarjetas de características en la segunda columna
    with col2:
        st.markdown("""
        <style>
        .card {
            background-color: #2C3E50;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
            display: flex;
            align-items: center;
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
        }
        .icon {
            background-color: #1ABC9C;
            padding: 10px;
            border-radius: 50%;
            margin-right: 10px;
            display: inline-block;
        }
        </style>

        <div class="card">
            <span class="icon">📝</span> Resúmenes Inteligentes de Clases
        </div>

        <div class="card">
            <span class="icon">📚</span> Recomendaciones de Contenido Inteligentes
        </div>

        <div class="card">
            <span class="icon">🧠</span> Aprendizaje Personalizado y Adaptativo
        </div>

        <div class="card">
            <span class="icon">🔍</span> Evaluaciones y Retroalimentación Instantánea
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style='color: #FF6F61; text-align: center;'>LUMINA</h3>
        <p style='text-align: center;'>Herramientas innovadoras para transformar la educación.</p>
        """, unsafe_allow_html=True)






elif selected == "Cursos":
    if st.session_state.page == "home":
        st.title("Cursos")
        st.write("Sección para explorar y gestionar tus cursos.")
        
        # Barra de búsqueda y filtros
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.text_input("🔍 Busque sus cursos", placeholder="Ingrese el nombre del curso...")

        with col2:
            st.selectbox("Períodos", ["Todos los períodos", "2024-I", "2023-II", "2023-I"])

        with col3:
            st.selectbox("Filtros", ["Cursos abiertos", "Cursos cerrados"])

        # Tarjetas de cursos
        col1, col2, col3 = st.columns(3)

        # Primera tarjeta de curso
        with col1:
            if st.button("Business Intelligence", key="btn1"):
                go_to_page("business_intelligence")
            st.image("https://www.ceupe.pe/images/easyblog_articles/233/big.jpg", use_column_width=True)
            st.markdown("**Business Intelligence**")  # Nombre en negrita
            st.write("Javier Wam")

        # Segunda tarjeta de curso
        with col2:
            if st.button("Machine Learning", key="btn2"):
                go_to_page("machine_learning")
            st.image("https://www.repsol.com/content/dam/repsol-corporate/es/energia-e-innovacion/robot-machine-learning.jpg.transform/rp-rendition-sm/image.jpg", use_column_width=True)
            st.markdown("**Machine Learning**")  # Nombre en negrita
            st.write("Victor Hugo Ayma")

        # Tercera tarjeta de curso
        with col3:
            if st.button("Data Mining", key="btn3"):
                go_to_page("data_mining")
            st.image("https://lh3.googleusercontent.com/7VpB3YyzvA3ztEjLVoOZzRbB9dBh6WCLeZMbtxVj9uAKemJPOoJMkgsfR35JpCzjCEDpyy_5GJ4pVFyWQTC4LHCqEgVSFIdvNwB-S8YV8Slz3fSBuA1dMpuJ-NP-wui6yX55kVzwLrmDw_Y9thYaNwg", use_column_width=True)
            st.markdown("**Data Mining**")  # Nombre en negrita
            st.write("Soledad Espezúa")

# Definición de los cursos fuera del bloque para su uso en múltiples secciones
    courses = [
        {"nombre": "Introducción y Repaso SQL","calificacion": "4.7 ★ (71,109 valoraciones)", "detalles": "Este curso introduce los conceptos básicos de la ciencia de datos..."},
        {"nombre": "Funciones, Procedimientos, Vistas, Triggers", "duracion": "18 horas", "calificacion": "4.5 ★ (28,855 valoraciones)", "detalles": "Aprende sobre las herramientas esenciales utilizadas en la ciencia de datos..."},
        {"nombre": "Conceptos BI", "duracion": "6 horas", "calificacion": "4.6 ★ (20,241 valoraciones)", "detalles": "Explora las metodologías clave para la aplicación de la ciencia de datos..."},
        {"nombre": "Cubos OLAP", "duracion": "25 horas", "calificacion": "4.6 ★ (37,177 valoraciones)", "detalles": "Profundiza en Python y su aplicación en IA y desarrollo..."},
    ]

    # Gestión del estado de la navegación
    if "page" not in st.session_state:
        st.session_state.page = "business_intelligence"

    # Página principal de Business Intelligence
    if st.session_state.page == "business_intelligence":
        st.title("Business Intelligence")
        st.markdown("""
        <style>
        .course-card {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .course-title {
            font-size: 20px;
            color: #1E90FF;
            font-weight: bold;
            margin-bottom: 5px;
            cursor: pointer;
        }
        .course-details {
            font-size: 14px;
            color: #555;
            margin-bottom: 10px;
        }
        .back-button {
            background-color: #FF6F61;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            text-align: center;
        }
        .back-button:hover {
            background-color: #e55a4f;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
Especialízate en Data Analytics y conviértete en un experto en la interpretación de datos y la generación de soluciones innovadoras. El diploma te brindará las habilidades necesarias para comunicar de manera efectiva tus hallazgos, gestionar grandes volúmenes de datos y adaptarte a las últimas tendencias del mercado. Dominarás herramientas como Power BI, SQL y Python para el análisis, modelado y visualización de datos, mientras exploras métodos y técnicas avanzadas de Business Analytics para una toma de decisiones fundamentada en los datos.
        """)

        # División de la página en dos columnas
        col1, col2 = st.columns([2, 1])

        with col1:
            # Lista de cursos incluidos
            st.subheader("Cursos Incluidos:")
            for idx, course in enumerate(courses):
                #st.markdown(f"<div class='course-card'>", unsafe_allow_html=True)
                if st.button(course['nombre'], key=f"course_{idx}"):
                    st.session_state.page = f"course_{idx}"  # Navega a la página del curs
                st.markdown(f"<div class='course-details'>Calificación: {course['calificacion']}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Información sobre los instructores
            st.subheader("Instructores:")
            instructors = [
                {"nombre": "Javier Wam", "participantes": "15"},
            ]
            
            for instructor in instructors:
                st.markdown(f"**{instructor['nombre']}**")
                st.write(f"- Participantes: {instructor['participantes']}")

        # Botón para volver a la página principal con diseño
        st.button("Volver", on_click=lambda: st.session_state.update(page="home"))

    for idx, course in enumerate(courses):
        if st.session_state.page == f"course_{idx}":
            st.title(course['nombre'])
            st.write(f"Calificación: {course['calificacion']}")
            st.write(course['detalles'])

            # Opción para subir videos
            st.subheader("Subir un Video")
            video_file = st.file_uploader("Elige un archivo de video", type=["mp4", "mov", "avi"], key=f"video_uploader_{idx}")
            if video_file is not None:
                st.video(video_file)
                

            pdf_path = "Business Intelligence - Clase1 - Presentacion.pdf"

            # Función para mostrar el PDF en la aplicación
            def mostrar_pdf(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            # Leer el contenido del PDF
            def extraer_texto(pdf_path):
                texto = ""
                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        texto += page.extract_text()
                return texto

            # Layout en dos columnas
            col1, col2 = st.columns(2)

            # Mostrar PDF en la primera columna
            with col1:
                st.subheader("Visualización del PDF")
                mostrar_pdf(pdf_path)

            # Mostrar el resumen en la segunda columna
            with col2:
                st.subheader("Resumen del PDF")
                texto = extraer_texto(pdf_path)

                # Botón para resumir el contenido del PDF
                if st.button("Resumir PDF"):
                    if texto:
                        try:
                            # Llamar a la API de GPT para resumir el texto
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "Eres un experto en resúmenes de documentos."},
                                    {"role": "user", "content": f"Por favor, resume el siguiente texto: {texto}"}
                                ]
                            )

                            # Obtener el resumen generado por GPT
                            resumen = response.choices[0].message['content']
                            st.write(resumen)

                        except Exception as e:
                            st.error(f"Error al generar el resumen: {e}")
                    else:
                        st.warning("No se pudo extraer texto del PDF.")
# Botón para generar un esquema jerárquico (mapa mental)
                if st.button("Generar Mapa Mental"):
                    if texto:
                        try:
                            # Enviar a GPT para generar un esquema estructurado
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "Eres un experto en la creación de mapas mentales y resúmenes estructurados."},
                                    {"role": "user", "content": f"Genera un esquema jerárquico tipo mapa mental del siguiente texto: {texto}"}
                                ]
                            )

                            # Obtener el esquema estructurado
                            esquema = response.choices[0].message['content']
                            st.subheader("Esquema para Mapa Mental:")
                            st.write(esquema)

                            # Aquí puedes agregar un paso adicional para transformar el esquema en un gráfico visual usando librerías como Graphviz

                        except Exception as e:
                            st.error(f"Error al generar el esquema: {e}")
                    else:
                        st.warning("No se pudo extraer texto del PDF.")


            # Sección de chat
            st.subheader("Chat Interactivo")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_input = st.text_input("Escribe tu mensaje:", key=f"user_input_{idx}")
            if st.button("Enviar", key=f"send_button_{idx}"):
                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Echo: {user_input}"})

            # Mostrar el historial de chat
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f"**Tú:** {msg['content']}")
                else:
                    st.markdown(f"**Chatbot:** {msg['content']}")

            # Botón para volver a la página principal del curso
            st.button("Volver", on_click=lambda: st.session_state.update(page="business_intelligence"), key=f"back_button_{idx}")



    # Página del curso Machine Learning
    if st.session_state.page == "machine_learning":
        st.title("Machine Learning")
        st.write("Detalles del curso de Machine Learning...")
        if st.button("Volver", key="back2"):
            st.session_state.page = "home"

    # Página del curso Data Mining
    elif st.session_state.page == "data_mining":
        st.title("Data Mining")
        st.write("Detalles del curso de Data Mining...")
        if st.button("Volver", key="back3"):
            st.session_state.page = "home"


elif selected == "Calendario":
    st.title("Calendario")
    st.write("Revisa y gestiona tus eventos y fechas importantes.")



# Configuración de la sección de Mensajes
elif selected == "Mensajes":
    import openai
    import streamlit as st

    # Estilo CSS para personalizar los mensajes
    st.markdown("""
    <style>
    .user-message {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: black;
        display: inline-block;
        max-width: 80%;
    }
    .assistant-message {
        background-color: #E5E5EA;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: black;
        display: inline-block;
        max-width: 80%;
    }
    .message-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .user-container {
        display: flex;
        justify-content: flex-end;
    }
    .assistant-container {
        display: flex;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

    # Título de la sección
    st.title("LuMI Chat")

    # Configuración del cliente de OpenAI con la clave API
    openai.api_key = "sk-svcacct-a1MZI8E9hKbUpVipGsu6Vot-AHbf4e_2FZxK4mMUjxXdygT3BlbkFJSALbM3XsduooEJoHPY2dPBvNX5FlySAoyU8RQyKJJsPYgA"

    # Verificar si el modelo está configurado en la sesión
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Verificar si hay mensajes en la sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar los mensajes almacenados en el historial
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-container">
                    <div class="user-message">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-container">
                    <div class="assistant-message">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Captura de la entrada del usuario
    prompt = st.text_input("Escribe tu mensaje aquí:")

    if st.button("Enviar"):
        if prompt:
            # Agregar el mensaje del usuario al historial
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                st.markdown(f"""
                <div class="user-container">
                    <div class="user-message">{prompt}</div>
                </div>
                """, unsafe_allow_html=True)

            # Generar la respuesta del asistente
            response = openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=st.session_state.messages
            )

            # Obtener el contenido completo de la respuesta
            response_text = response.choices[0].message['content']

            # Mostrar la respuesta completa del asistente
            with chat_container:
                st.markdown(f"""
                <div class="assistant-container">
                    <div class="assistant-message">{response_text}</div>
                </div>
                """, unsafe_allow_html=True)

            # Agregar la respuesta del asistente al historial
            st.session_state.messages.append({"role": "assistant", "content": response_text})


            
elif selected == "Calificaciones":
    st.title("Calificaciones")
    st.write("Revisa tus calificaciones y retroalimentaciones.")


    # Primera tarjeta de calificación
    with st.expander("Business Inteligence"):
        st.markdown("""
        <div style='border: 1px solid #0073e6; padding: 10px; border-radius: 5px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <p style='font-size: 18px; font-weight: bold; margin: 0;'>Business Inteligence</p>
                </div>
                <div>
                    <span style='font-size: 18px; color: #555;'>--</span>
                </div>
            </div>
            <hr style='margin: 5px 0;'>
            <div style='display: flex; justify-content: space-between;'>
                <p style='margin: 0;'>Data Mart</p>
                <p style='margin: 0;'>Actividad</p>
            </div>
            <p style='color: #0073e6; text-align: right;'><a href="#" style='text-decoration: none;'>Ver todos los trabajos (3)</a></p>
        </div>
        """, unsafe_allow_html=True)

    # Segunda tarjeta de calificación
    with st.expander("Machine Learning"):
        st.markdown("""
        <div style='border: 1px solid #00b300; padding: 10px; border-radius: 5px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <p style='font-size: 18px; font-weight: bold; margin: 0;'>Machine Learning</p>
                </div>
                <div>
                    <span style='font-size: 18px; color: #555;'>--</span>
                </div>
            </div>
            <hr style='margin: 5px 0;'>
            <p style='color: #555; text-align: center;'>Cuando haya calificaciones disponibles para este curso, aparecerán aquí.</p>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Herramientas":
    st.title("Herramientas")
    st.write("Aquí puedes acceder a diversas funcionalidades avanzadas.")



    st.markdown("""
    <style>
    .tool-card {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .tool-title {
        font-size: 18px;
        color: #1E90FF;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .tool-description {
        font-size: 14px;
        color: #555;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Lista de herramientas
    tools = [
        {
            "nombre": "Gestión de Usuarios",
            "descripcion": "Administra los usuarios de la plataforma, incluyendo permisos y roles.",
            "funcion": lambda: st.write("Funcionalidad de Gestión de Usuarios aún en desarrollo."),
        },
        {
            "nombre": "Generador de Reportes",
            "descripcion": "Genera reportes personalizados a partir de los datos disponibles en la plataforma.",
            "funcion": lambda: st.write("Funcionalidad de Generador de Reportes aún en desarrollo."),
        },
        {
            "nombre": "Análisis de Datos",
            "descripcion": "Analiza datos importantes mediante gráficos y estadísticas detalladas.",
            "funcion": lambda: st.write("Funcionalidad de Análisis de Datos aún en desarrollo."),
        },
        {
            "nombre": "Importar/Exportar Datos",
            "descripcion": "Importa o exporta datos en diferentes formatos para integraciones con otros sistemas.",
            "funcion": lambda: st.write("Funcionalidad de Importar/Exportar Datos aún en desarrollo."),
        },
        {
            "nombre": "Monitor de Actividad",
            "descripcion": "Monitorea la actividad de los usuarios en tiempo real para una mejor supervisión.",
            "funcion": lambda: st.write("Funcionalidad de Monitor de Actividad aún en desarrollo."),
        },
        {
            "nombre": "Configuración del Sistema",
            "descripcion": "Accede a las configuraciones avanzadas del sistema para ajustar la plataforma según las necesidades.",
            "funcion": lambda: st.write("Funcionalidad de Configuración del Sistema aún en desarrollo."),
        },
    ]

    # Mostrar herramientas
    for tool in tools:
        st.markdown(f"<div class='tool-title'>{tool['nombre']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tool-description'>{tool['descripcion']}</div>", unsafe_allow_html=True)
        if st.button(f"Acceder a {tool['nombre']}", key=tool['nombre']):
            tool['funcion']()
        st.markdown("</div>", unsafe_allow_html=True)

    # Botón para volver a la página principal
    if st.button("Volver", key="back_tools"):
        st.session_state.page = "home"

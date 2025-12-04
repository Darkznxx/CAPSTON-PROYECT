# =========================================================
# 📘 BURNOUT WEB APP - FLASK + SESIONES + SHAP
# =========================================================
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import joblib
import numpy as np
import pandas as pd
import shap
import traceback
import os
import sqlite3
from datetime import datetime
from io import BytesIO
import base64
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import openpyxl

# =========================================================
#                  Configuración Flask
# =========================================================
app = Flask(__name__)
app.secret_key = "12345" 

# =========================================================
#                  Rutas de artefactos
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPE_PATH = os.path.join(MODELS_DIR, "best_pipeline.pkl")
META_PATH = os.path.join(MODELS_DIR, "metadata.pkl")
PREP_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
SHAP_PATH = os.path.join(MODELS_DIR, "shap_explanation.pkl")

# =========================================================
#           Carga de modelos y metadatos
# =========================================================
try:
    pipeline = joblib.load(PIPE_PATH)
    metadata = joblib.load(META_PATH)
    preprocessor = joblib.load(PREP_PATH)

    FEATURES = metadata.get("features", [])
    LABEL_MAP = metadata.get("label_map", {})
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    print("✅ Modelos y metadatos cargados correctamente.")
except Exception as e:
    print("❌ Error al cargar los artefactos:", e)
    pipeline, FEATURES, LABEL_MAP, REVERSE_LABEL_MAP = None, [], {}, {}


# =========================================================
#               Inicializar SHAP
# =========================================================
explainer = None
try:
    import shap
    import traceback
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    print("🧩 Iniciando carga de SHAP...")

    if os.path.exists(SHAP_PATH):
        try:
            explainer = joblib.load(SHAP_PATH)
            print("✅ SHAP Explainer cargado correctamente desde archivo.")
        except Exception as e_load:
            print("⚠️ Archivo shap_explanation.pkl incompatible o corrupto. Se regenerará.")
            os.remove(SHAP_PATH)
            raise e_load

    if explainer is None and pipeline is not None:
        clf = getattr(pipeline, "named_steps", {}).get("clf", pipeline)
        print("🔍 Tipo de modelo detectado:", type(clf).__name__)

        #  Si es un modelo calibrado, obtener el estimador base
        if isinstance(clf, CalibratedClassifierCV):
            base_model = getattr(clf, "base_estimator", getattr(clf, "estimator", None))
            print("🧠 Modelo base detectado dentro del calibrador:", type(base_model).__name__)
        else:
            base_model = clf

        #  Si el modelo base es un Pipeline, extraer su clasificador final
        if isinstance(base_model, Pipeline):
            inner_clf = getattr(base_model, "named_steps", {}).get("clf", base_model)
            print("🧩 Clasificador interno detectado:", type(inner_clf).__name__)
        else:
            inner_clf = base_model

        #  Generar el explainer si el clasificador interno es Random Forest
        if isinstance(inner_clf, RandomForestClassifier):
            print("⚙️ Generando nuevo SHAP TreeExplainer para Random Forest (modo compatibilidad)...")
            explainer = shap.TreeExplainer(inner_clf)
            joblib.dump(explainer, SHAP_PATH)
            print("✅ Nuevo SHAP Explainer generado y guardado.")
        else:
            print("⚠️ El clasificador interno no es Random Forest, SHAP no se inicializa.")
except Exception as e:
    print("❌ Error al cargar o inicializar SHAP:", e)
    traceback.print_exc()

# =========================================================
#          INTERPRETACIÓN TEXTUAL DE SHAP 
# =========================================================

#  Mapeo de nombres técnicos a etiquetas comprensibles
FEATURE_LABELS = {
    "edad_num": "la edad del trabajador",
    "sector": "el sector laboral en el que se desempeña",
    "antiguedad": "la antigüedad en la organización",
    "item1": "la sensación de agotamiento emocional",
    "item2": "el cansancio al final de la jornada",
    "item3": "la dificultad para empezar la jornada laboral",
    "item4": "el agotamiento tras un día de trabajo",
    "item5": "la capacidad para resolver problemas laborales",
    "item6": "la sensación de estar quemado por el trabajo",
    "item7": "la percepción de contribución valiosa",
    "item8": "la pérdida de interés por las tareas",
    "item9": "la pérdida de entusiasmo por el trabajo",
    "item10": "la percepción de competencia profesional",
    "item11": "la satisfacción por los logros alcanzados",
    "item12": "la valoración de los resultados obtenidos",
    "item13": "la preferencia por trabajar sin interrupciones",
    "item14": "el escepticismo respecto al valor del trabajo",
    "item15": "las dudas sobre la utilidad del trabajo",
    "item16": "la confianza en la propia eficacia"
}


def interpretar_shap(feature, valor, shap_value, objetivo="nivel de burnout"):
    """
    Convierte un valor SHAP individual en una frase interpretativa básica.
    """
    impacto = abs(shap_value)
    if shap_value > 0:
        tendencia = "incrementa"
    elif shap_value < 0:
        tendencia = "reduce"
    else:
        tendencia = "no afecta significativamente"

    etiqueta = FEATURE_LABELS.get(feature, feature)
    return f"El valor de {etiqueta} ({valor}) {tendencia} el {objetivo} (impacto: {impacto:.3f})."


def resumen_para_RRHH(explicaciones, pred_label="", top_n=5, objetivo="nivel de burnout"):
    """
    Genera un resumen textual narrativo y adaptado al nivel de riesgo.
    Incluye nombres de variables legibles y frases naturales.
    """
    if not explicaciones:
        return "No se generaron explicaciones SHAP válidas para esta predicción."

    # Seleccionar las características con mayor influencia
    top_features = sorted(explicaciones, key=lambda x: abs(x[2]), reverse=True)[:top_n]

    # Construcción narrativa
    oraciones = []
    for feature, valor, shap_value in top_features:
        nombre_humano = FEATURE_LABELS.get(feature, feature.replace("_", " "))
        if shap_value > 0:
            oraciones.append(f"un valor alto en {nombre_humano} parece aumentar el {objetivo}")
        else:
            oraciones.append(f"un valor bajo en {nombre_humano} parece reducir el {objetivo}")

    # Unir frases de forma fluida
    cuerpo = ", ".join(oraciones[:-1]) + " y " + oraciones[-1] + "."

    # Introducción adaptada al nivel de predicción
    if "alto" in str(pred_label).lower():
        intro = "El modelo sugiere un riesgo alto de burnout. "
    elif "moderado" in str(pred_label).lower():
        intro = "Se observa un riesgo moderado de burnout. "
    else:
        intro = "El modelo indica un nivel bajo de burnout. "

    # Redacción final más natural
    texto = (
        f"{intro} Según el análisis de las respuestas, los principales factores que influyeron en esta predicción fueron que "
        f"{cuerpo} Esto ofrece una visión general sobre los aspectos que podrían estar contribuyendo al nivel actual de agotamiento laboral."
    )

    return texto

# =========================================================
#               Predicciones confirmadas
# =========================================================
predicciones_confirmadas = []

# =========================================================
#                        Inicio Test
# =========================================================
@app.route("/")
def home():
    # Mantener la pantalla pública inicial en predict.html
    return render_template("/Test/predict.html")  



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["username"] = user["username"]  
            return redirect(url_for("user_dashboard"))  
        return render_template("login.html", error="Usuario o contraseña inválidos")

    return render_template("login.html")



# =========================================================
#                     REGISTRO
# =========================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            return render_template("register.html", error="Las contraseñas no coinciden")

        try:
            conn = sqlite3.connect("database.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                           (username, email, password, "user"))
            conn.commit()
            conn.close()

            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="El usuario o correo ya existe")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# =========================================================
#                      Panel del Usuario 
# =========================================================

@app.route("/Usuario/inicio")
def user_dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    global predicciones_confirmadas
    confirmadas = []

    for p in predicciones_confirmadas:
        fecha_valor = p.get("fecha")
        if isinstance(fecha_valor, str):
            try:
                fecha_valor = datetime.strptime(fecha_valor, "%d/%m/%Y, %H:%M:%S")
            except Exception:
                fecha_valor = datetime.now()

        confirmadas.append({
            "usuario": p.get("usuario", "N/A"),
            "resultado": p.get("resultado", ""),
            "probabilidades": p.get("probabilidades", ""),
            "fecha": fecha_valor,
            "sector": p.get("sector", "N/A")
        })

    return render_template("/Usuario/inicio.html",
                           predicciones=confirmadas,
                           username=session.get("username"))

# =========================================================
#   VISTAS DE USUARIO (INICIO - DASHBOARD - TRABAJADORES)
# =========================================================

@app.route("/usuario/inicio")
def usuario_inicio():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    return render_template("Usuario/inicio.html", username=username)


@app.route('/usuario/dashboard')
def usuario_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']

    # Conector a la base de datos
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT nombre, resultado, probabilidad, fecha, sector FROM test")
    filas = cursor.fetchall()
    conn.close()

    predicciones = []
    for fila in filas:
        predicciones.append({
            'nombre': fila[0],
            'resultado': fila[1],
            'probabilidad': fila[2],
            'fecha': fila[3],
            'sector': fila[4]
        })

    total = len(predicciones)
    alto = len([p for p in predicciones if 'alto' in p['resultado'].lower()])
    bajo = len([p for p in predicciones if 'bajo' in p['resultado'].lower()])
    moderado = len([p for p in predicciones if 'moderado' in p['resultado'].lower()])

    data = {
        'total': total,
        'alto': alto,
        'bajo': bajo,
        'moderado': moderado
    }

    return render_template(
        "Usuario/dashboard.html",
        username=username,
        data=data,
        predicciones=predicciones
    )

# =========================================================
#            DASHBOARD EN TIEMPO REAL 
# =========================================================

@app.route('/api/datos_dashboard')
def api_datos_dashboard():
    """Datos generales para KPIs y distribución general."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT LOWER(resultado) FROM test")
        resultados = [r[0] for r in cursor.fetchall()]
        conn.close()

        total = len(resultados)
        alto = sum('alto' in r for r in resultados)
        moderado = sum('moderado' in r or 'medio' in r for r in resultados)
        bajo = sum('bajo' in r for r in resultados)

        return jsonify({
            'total': total,
            'alto': alto,
            'moderado': moderado,
            'bajo': bajo,
            'labels': ['Bajo', 'Moderado', 'Alto'],
            'data': [bajo, moderado, alto]
        })
    except Exception as e:
        print("❌ Error en /api/datos_dashboard:", e)
        return jsonify({'labels': [], 'data': []})


@app.route('/api/datos_evolucion_burnout')
def api_datos_evolucion_burnout():
    """Evolución diaria del burnout por nivel."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT fecha, LOWER(resultado) FROM test")
        filas = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(filas, columns=['fecha', 'resultado'])
        if df.empty:
            return jsonify({'labels': [], 'alto': [], 'moderado': [], 'bajo': []})

        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['dia'] = df['fecha'].dt.strftime('%d/%m')
        conteo = df.groupby(['dia', 'resultado']).size().unstack(fill_value=0).tail(14)

        return jsonify({
            'labels': conteo.index.tolist(),
            'alto': conteo.get('alto', pd.Series()).tolist(),
            'moderado': conteo.get('moderado', pd.Series()).tolist(),
            'bajo': conteo.get('bajo', pd.Series()).tolist()
        })
    except Exception as e:
        print("❌ Error en /api/datos_evolucion_burnout:", e)
        return jsonify({'labels': [], 'data': []})


@app.route('/api/datos_sector_burnout')
def api_datos_sector_burnout():
    """Distribución de burnout alto por sector."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT sector, LOWER(resultado) FROM test")
        filas = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(filas, columns=['sector', 'resultado'])
        if df.empty:
            return jsonify({'labels': [], 'alto': []})

        datos = df[df['resultado'].str.contains('alto')].groupby('sector').size()

        return jsonify({
            'labels': datos.index.tolist(),
            'alto': datos.values.tolist()
        })
    except Exception as e:
        print("❌ Error en /api/datos_sector_burnout:", e)
        return jsonify({'labels': [], 'data': []})


@app.route('/api/datos_stacked_burnout')
def api_datos_stacked_burnout():
    """Comparativa apilada por sector (bajo, moderado, alto)."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT sector, LOWER(resultado) FROM test")
        filas = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(filas, columns=['sector', 'resultado'])
        if df.empty:
            return jsonify({'labels': [], 'bajo': [], 'moderado': [], 'alto': []})

        conteo = df.groupby(['sector', 'resultado']).size().unstack(fill_value=0)

        return jsonify({
            'labels': conteo.index.tolist(),
            'bajo': conteo.get('bajo', pd.Series()).tolist(),
            'moderado': conteo.get('moderado', pd.Series()).tolist(),
            'alto': conteo.get('alto', pd.Series()).tolist()
        })
    except Exception as e:
        print("❌ Error en /api/datos_stacked_burnout:", e)
        return jsonify({'labels': [], 'data': []})


@app.route("/usuario/trabajadores")
def usuario_trabajadores():
    username = session.get("username", "Usuario")
    
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, nombre, correo, resultado, probabilidad, sector, fecha,
               item1, item2, item3, item4, item5, item6, item7, item8,
               item9, item10, item11, item12, item13, item14, item15, item16
        FROM test
    """)
    
    registros = cursor.fetchall()
    conn.close()

    columnas = ["id", "nombre", "correo", "resultado", "probabilidad", "sector", "fecha",
                "item1","item2","item3","item4","item5","item6","item7","item8",
                "item9","item10","item11","item12","item13","item14","item15","item16"]
    
    trabajadores = [dict(zip(columnas, fila)) for fila in registros]
    
    return render_template("Usuario/trabajadores.html", username=username, trabajadores=trabajadores)


# =========================================================
#                 PREDICCIÓN 
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        for i in range(1, 17):
            key = f"item{i}"
            try:
                data[key] = int(data.get(key, 0))
            except:
                data[key] = None

        X_input = pd.DataFrame([{k: data.get(k, None) for k in FEATURES}])

        if "edad_num" in X_input.columns:
            if X_input["edad_num"].isna().all() or (X_input["edad_num"] == "").all():
                X_input["edad_num"] = 35  
        else:
            X_input["edad_num"] = 35

        pred = pipeline.predict(X_input)[0]
        proba = pipeline.predict_proba(X_input)[0].tolist()

        pred_label = REVERSE_LABEL_MAP.get(pred, "Clase no encontrada")

        probabilities = {REVERSE_LABEL_MAP[i]: float(prob) for i, prob in enumerate(proba)}

        # =========================================================
        # ✅ SHAP: explicación del modelo
        # =========================================================
        shap_values = None
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestClassifier

            clf = getattr(pipeline, "named_steps", {}).get("clf", pipeline)

            if isinstance(clf, CalibratedClassifierCV):
                base_model = getattr(clf, "base_estimator", getattr(clf, "estimator", None))
            else:
                base_model = clf

            if isinstance(base_model, Pipeline):
                inner_clf = getattr(base_model, "named_steps", {}).get("clf", base_model)
            else:
                inner_clf = base_model

            if isinstance(inner_clf, RandomForestClassifier) and explainer is not None:
                X_trans = (
                    pipeline.named_steps["pre"].transform(X_input)
                    if "pre" in pipeline.named_steps
                    else X_input
                )

                shap_values_raw = explainer.shap_values(X_trans)

                if isinstance(shap_values_raw, list):
                        shap_values = np.mean(np.abs(shap_values_raw), axis=0)
                        shap_values = shap_values.flatten().tolist()
                else:
                        shap_values = np.abs(shap_values_raw).flatten().tolist()

                print("✅ SHAP calculado correctamente.")

                # =========================================================
                #           Interpretación textual de SHAP
                # =========================================================
                resumen_shap = ""
                try:
                    if shap_values and isinstance(shap_values, list):
                        explicaciones = []
                        for i, f in enumerate(FEATURES):
                            val = X_input.iloc[0][f]
                            sv = shap_values[i] if i < len(shap_values) else 0
                            explicaciones.append((f, val, sv))

                        resumen_shap = resumen_para_RRHH(explicaciones, pred_label=pred_label, top_n=5)
                    else:
                        resumen_shap = "No se pudieron generar interpretaciones SHAP válidas."
                except Exception as e:
                    print("⚠️ Error interpretando SHAP:", e)
                    resumen_shap = "Error durante la interpretación de los valores SHAP."

            else:
                print("⚠️ No se encontró RandomForestClassifier válido para SHAP.")
                shap_values = None

        except Exception as shap_error:
            print("⚠️ Error SHAP:", shap_error)
            shap_values = None

        # =========================================================
        #         GUARDAR EN BD EN SQLITE 
        # =========================================================
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT,
                correo TEXT,
                sector TEXT,
                antiguedad TEXT,
                item1 INTEGER,
                item2 INTEGER,
                item3 INTEGER,
                item4 INTEGER,
                item5 INTEGER,
                item6 INTEGER,
                item7 INTEGER,
                item8 INTEGER,
                item9 INTEGER,
                item10 INTEGER,
                item11 INTEGER,
                item12 INTEGER,
                item13 INTEGER,
                item14 INTEGER,
                item15 INTEGER,
                item16 INTEGER,
                resultado TEXT,
                probabilidad TEXT,
                fecha TEXT
            )
        """)

        cursor.execute("""
           INSERT INTO test (
               nombre, correo, sector, antiguedad,
               item1, item2, item3, item4, item5, item6, item7, item8,
               item9, item10, item11, item12, item13, item14, item15, item16,
               resultado, probabilidad, fecha
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("nombre"),
            data.get("correo"),
            data.get("sector"),
            data.get("antiguedad"),
            data.get("item1"),
            data.get("item2"),
            data.get("item3"),
            data.get("item4"),
            data.get("item5"),
            data.get("item6"),
            data.get("item7"),
            data.get("item8"),
            data.get("item9"),
            data.get("item10"),
            data.get("item11"),
            data.get("item12"),
            data.get("item13"),
            data.get("item14"),
            data.get("item15"),
            data.get("item16"),
            pred_label,
            str(probabilities),
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "prediction": pred_label,
            "probabilities": probabilities,
            "shap_values": shap_values,
            "resumen_shap": resumen_shap
        })

    except Exception as e:
        print("❌ Error en predicción:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# =========================================================
#                Confirmar predicción 
# =========================================================
@app.route("/admin/confirmar_prediccion", methods=["POST"])
def confirmar_prediccion():
    try:
        data = request.json
        if not data:
            return jsonify({"ok": False, "error": "Datos no recibidos"}), 400

        nueva_pred = {
            "usuario": data.get("usuario", "Desconocido"),
            "resultado": data.get("resultado", ""),
            "probabilidades": data.get("probabilidades", ""),
            "fecha": data.get("fecha", datetime.now().strftime("%d/%m/%Y, %H:%M:%S")),
            "sector": data.get("sector", "N/A")
        }

        predicciones_confirmadas.append(nueva_pred)
        print("✅ Predicción confirmada y almacenada.")
        return jsonify({"ok": True})

    except Exception as e:
        print("❌ Error al confirmar:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================================================
#  GENERAR PDF PARA UN USUARIO 
# =========================================================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import os
from datetime import datetime


@app.route("/generar_pdf", methods=["POST"])
def generar_pdf():
    try:
        data = request.json or {}

        nombre = data.get("nombre", "Usuario")
        correo = data.get("correo", "No especificado")
        clase = data.get("clase", "Sin resultado")
        resumen_shap = data.get("resumen_shap", "")
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            title="Reporte de Predicción de Burnout",
            author=nombre
        )

        styles = getSampleStyleSheet()
        elementos = []

        # TÍTULO
        elementos.append(Paragraph("<b>Reporte de Predicción de Burnout</b>", styles["Title"]))
        elementos.append(Spacer(1, 12))

        # DATOS DEL USUARIO
        elementos.append(Paragraph(f"Nombre del usuario: <b>{nombre}</b>", styles["Normal"]))
        elementos.append(Paragraph(f"Correo: <b>{correo}</b>", styles["Normal"]))
        elementos.append(Spacer(1, 12))

        # RESULTADO
        elementos.append(Paragraph("<b>Resultado</b>", styles["Heading2"]))
        elementos.append(Paragraph(f"Nivel de burnout predicho: <b>{clase}</b>", styles["Normal"]))
        elementos.append(Spacer(1, 12))

        # SHAP
        elementos.append(Paragraph("<b>Observaciones (SHAP)</b>", styles["Heading3"]))
        elementos.append(Paragraph(resumen_shap or "No se generó una interpretación textual de SHAP.", styles["Normal"]))

        # LOGO 
        logo_path = os.path.join(BASE_DIR, "static/img/robot.png")
        if os.path.exists(logo_path):
            logo = Image(logo_path, width=130, height=130)
            logo.hAlign = "CENTER"
            elementos.append(logo)

        elementos.append(Spacer(1, 20))

        # FECHA
        elementos.append(Paragraph(f"Fecha del reporte: {fecha}", styles["Italic"]))

        doc.build(elementos)

        buffer.seek(0)

        pdf_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({
            "status": "ok",
            "filename": f"Reporte_Burnout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "pdf_base64": pdf_base64
        })

    except Exception as e:
        print("ERROR GENERAR PDF:", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
#           ENVIAR PDF POR CORREO (Brevo )
# =========================================================

# 🔹 CONFIGURACIÓN BREVO
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "TU_API_KEY_BREVO") 
BREVO_SEND_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import sqlite3
import io
import base64
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException


@app.route('/enviar_pdf_correo', methods=['POST'])
def enviar_pdf_correo():
    try:
        id_prediccion = request.form.get("id_prediccion")
        recomendaciones = request.form.get("recomendaciones", "")
        preview = request.form.get("preview")  

        if not id_prediccion:
            return jsonify({"error": "Falta el ID del registro."}), 400

        # =====================================================
        #          Conexión a la base de datos 
        # =====================================================
        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM test WHERE id = ?", (id_prediccion,))
        fila = cursor.fetchone()
        conn.close()

        if not fila:
            return jsonify({"error": "No se encontró el registro en la tabla 'test'."}), 404

        # =====================================================
        #            Datos del trabajador
        # =====================================================
        nombre = fila["nombre"] if "nombre" in fila.keys() else "Usuario"
        resultado = fila["resultado"] if "resultado" in fila.keys() else "Desconocido"
        probabilidad = fila["probabilidad"] if "probabilidad" in fila.keys() else "N/A"
        correo = fila["correo"] if "correo" in fila.keys() else None

        if not correo:
            return jsonify({"error": "El usuario no tiene correo registrado."}), 400

        # =====================================================
        #            Generar PDF con diseño limpio 
        # =====================================================
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        logo_path = "static/img/botcito.png"
        if os.path.exists(logo_path):
            elements.append(Image(logo_path, width=1.5*inch, height=1.5*inch))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Reporte de Predicción de Burnout</b>", styles['Title']))
        elements.append(Spacer(1, 20))

        fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
        elements.append(Paragraph(f"<b>Nombre del trabajador:</b> {nombre}", styles['Normal']))
        elements.append(Paragraph(f"<b>Resultado del modelo:</b> {resultado}", styles['Normal']))
        elements.append(Paragraph(f"<b>Fecha de envío:</b> {fecha}", styles['Normal']))
        elements.append(Spacer(1, 15))

        elements.append(Paragraph("<b>Recomendaciones personalizadas:</b>", styles['Heading3']))
        for linea in recomendaciones.split("\n"):
            elements.append(Paragraph(linea, styles['Normal']))
        elements.append(Spacer(1, 12))

        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()
        pdf_base64 = base64.b64encode(pdf_data).decode()

        if preview == "true":
            return jsonify({"pdf_base64": pdf_base64})

        # =====================================================
        #             Enviar correo con Brevo
        # =====================================================
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

        subject = "📄 Reporte de Predicción de Burnout"
        sender = {"name": "Sistema Capston2", "email": "tusistema@empresa.com"}

        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=to,
            sender=sender,
            subject=subject,
            html_content=f"""
                <p>Hola <b>{nombre}</b>,</p>
                <p>Adjunto encontrarás tu reporte de predicción de burnout con las recomendaciones personalizadas.</p>
                <p>Saludos,<br><b>Equipo de Bienestar Laboral</b></p>
            """,
            attachment=[
                {
                    "name": f"reporte_burnout_{id_prediccion}.pdf",
                    "content": pdf_base64
                }
            ]
        )

        api_instance.send_transac_email(send_smtp_email)
        return jsonify({"ok": True, "mensaje": f"📧 Correo enviado correctamente a {correo}."})

    except sqlite3.OperationalError as e:
        print(f"❌ Error de base de datos: {e}")
        return jsonify({"error": "Error en la base de datos. Verifica la tabla 'test'."}), 500

    except ApiException as e:
        print(f"❌ Error al enviar correo con Brevo: {e}")
        return jsonify({"error": "Error al enviar el correo. Verifica la API Key o los parámetros."}), 500

    except Exception as e:
        print(f"❌ Error inesperado en /enviar_pdf_correo: {e}")
        return jsonify({"error": str(e)}), 500




# =========================================================
#              Ejecutar servidor
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)

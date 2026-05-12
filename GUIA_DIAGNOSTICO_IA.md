# 🤖 Guía: Diagnóstico de Modelos con IA (Groq API)

## ¿Qué se agregó?

Cuando entrenes un modelo en la página **"Análisis Avanzado"**, ahora puedes generar un **diagnóstico en lenguaje no técnico** que explica cómo funciona tu modelo usando inteligencia artificial.

**Características:**
- ✅ Explicaciones simples para personas sin conocimiento técnico
- ✅ Sin términos confusos como "matriz de confusión" o "verdaderos positivos"
- ✅ Analogías del mundo real
- ✅ Recomendaciones prácticas

---

## 📋 Pasos de Configuración

### 1️⃣ Obtén una API Key de Groq (Gratis)

Groq es una plataforma de IA ultrarrápida y gratuita:

1. Ve a: **https://console.groq.com**
2. Regístrate con tu email
3. En la sección **"API Keys"**, copia tu token (comienza con `gsk_`)
4. ¡Listo! No necesitas tarjeta de crédito

---

### 2️⃣ Configura tu API Key

#### Opción A: Variables de Entorno del Sistema (Recomendado)

**En Windows (PowerShell como Administrador):**

```powershell
setx GROQ_API_KEY "gsk_tu_token_aqui"
```

Luego **reinicia Streamlit**:
```bash
streamlit run app.py
```

#### Opción B: Archivo `.env` (Más Fácil)

1. En la raíz del proyecto, crea o edita el archivo `.env`:

```
GROQ_API_KEY=gsk_tu_token_aqui
```

2. Asegúrate que `.gitignore` contenga `.env` (no subas tu API Key a GitHub)

3. Reinicia Streamlit

---

### 3️⃣ Instala Dependencias

```bash
pip install -r requirements.txt
```

Esto incluye:
- `requests` (para llamadas HTTP)
- `python-dotenv` (para cargar variables de entorno)

---

## 🚀 Cómo Usar

### En la Página "Análisis Avanzado":

1. **Carga un dataset** (CSV)
2. **Entrena un modelo** haciendo clic en "Iniciar Entrenamiento"
3. **Espera a que termine** (verás métricas: Accuracy, F1-Score, etc.)
4. **Busca la sección** "🔍 Diagnóstico del Modelo (Explicación Clara)"
5. **Haz clic en** "📊 Generar Diagnóstico"
6. **Lee la explicación IA** en lenguaje simple

---

## 📝 Ejemplo de Diagnóstico

**Entrada (Métricas técnicas):**
- Accuracy: 87%
- Precision: 0.89
- F1-Score: 0.86

**Salida (Diagnóstico IA - No técnico):**

> 🎯 **Tu modelo es muy bueno**
> 
> Funciona como un doctor que acierta el diagnóstico el 87% de las veces. No es perfecto, pero es bastante confiable para usar en decisiones reales.
> 
> **¿Por qué?** Se entrenó bien con muchos datos variados y aprendió patrones claros.
> 
> **¿Puedo confiar?** Sí, pero siempre revisa los casos dudosos. Es como confiar en un experto, pero consultar una segunda opinión en casos complejos.
> 
> **Mi recomendación:** Usa este modelo en producción, pero monitorea su desempeño cada mes.

---

## 🔐 Seguridad

⚠️ **Importante:**
- **Nunca** compartas tu API Key (`gsk_...`)
- **Nunca** lo subas a GitHub
- **Siempre** usa variables de entorno o `.env`
- `.env` está en `.gitignore` de este proyecto

---

## 🆘 Troubleshooting

### Error: "API Key no encontrada"

**Solución:**
```bash
# Verifica que está configurado
echo $env:GROQ_API_KEY  # Windows PowerShell
# o
echo $GROQ_API_KEY      # Linux/Mac
```

Si no aparece nada, sigue nuevamente los pasos de configuración.

### Error: "Timeout: La API tardó demasiado"

**Causa:** Conexión lenta
**Solución:** Intenta de nuevo. Si persiste, verifica tu conexión a internet.

### Error: "Error de Groq: Invalid API Key"

**Solución:**
1. Verifica que copiaste el token completo (sin espacios)
2. Reemplázalo con uno nuevo de: https://console.groq.com

---

## 📊 Cómo Funciona Internamente

```
1. Entrenas modelo → Obtienes métricas (Accuracy, F1-Score, etc.)
2. Haces clic en "Generar Diagnóstico"
3. Se envía a Groq API: "Explica estas métricas sin tecnicismos"
4. Groq genera texto en lenguaje natural
5. Se muestra en Streamlit
```

**Tiempo:** ~2-5 segundos

---

## 💡 Tips

- El diagnóstico es **personalizado** según tu modelo
- Funciona para **cualquier dataset** (finanzas, medicina, venta, etc.)
- Puedes generar múltiples diagnósticos para comparar modelos
- Ideal para **presentar resultados a ejecutivos/clientes**

---

## 📞 Más Información

- **Documentación Groq:** https://console.groq.com/docs
- **Modelos disponibles:** https://console.groq.com/docs/models
- **Código fuente:** `utils/groq_diagnostic.py`

---

**¡Listo! Ahora puedes obtener diagnósticos de IA para tus modelos.** 🚀

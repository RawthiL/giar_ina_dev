import json
import base64
from openai import OpenAI
from collections import Counter

# ==========================================
# 1. CARGAR LA TAXONOMÍA DESDE EL ARCHIVO JSON
# ==========================================
ruta_esquema = "taxonomia.json"
try:
    with open(ruta_esquema, "r", encoding="utf-8") as archivo:
        esquema_taxonomia = json.load(archivo)
    print(f"[✓] Taxonomía cargada exitosamente desde {ruta_esquema}")
except FileNotFoundError:
    print(f"[X] Error: No se encontró el archivo {ruta_esquema}")
    exit(1)

# ==========================================
# 2. CONFIGURAR EL CLIENTE PARA vLLM LOCAL
# ==========================================
# Apuntamos a la dirección donde está corriendo tu servidor vLLM (WSL)
client = OpenAI(
    base_url="http://localhost:9989/v1",
    api_key="none"  # vLLM local no requiere clave real
)

# ==========================================
# 3. PREPARAR LA IMAGEN Y EL PROMPT
# ==========================================
def codificar_imagen_base64(ruta_imagen):
    with open(ruta_imagen, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

ruta_imagen_celula = "cell.png"
imagen_b64 = codificar_imagen_base64(ruta_imagen_celula)

prompt_usuario = (
    "Analiza esta microfotografía de una célula individual de Allium cepa. "
    "Identifica el estado mitótico, características morfológicas, tinción y calidad. "
    "Genera el caption descriptivo al final."
)

# ==========================================
# 4. ENVIAR LA PETICIÓN ESTRUCTURADA A vLLM
# ==========================================
print("[*] Enviando imagen y taxonomía a vLLM. Esperando análisis...")
resultados_crudos = []
for i in range(1,10):
    try:
        respuesta = client.chat.completions.create(
            model="local_model", # Cambia esto por el modelo exacto que cargaste en vLLM
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en microscopía botánica. Tu salida debe ser estrictamente un JSON válido que cumpla con el esquema proporcionado."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_usuario},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_b64}"}}
                    ]
                },
                # {
                #     "role": "assistant",
                #     "content": "{"
                # }
            ],
            temperature=0.0,
            seed=42, 
            top_p=0.001, 
            #top_k=20, 
            #min_p=0.0, 
            #presence_penalty=0.0, 
            #repetition_penalty=1.0,
            # Aquí es donde ocurre la magia: forzamos el motor de vLLM a usar tu JSON
            response_format={
                "type": "json_schema",
                "json_schema": {
                "name": "taxonomia.json",
                "schema": esquema_taxonomia
                }
            },
            extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        )

    # ==========================================
    # 5. MOSTRAR EL RESULTADO
    # ==========================================
    
        texto_salida = respuesta.choices[0].message.content

        
        # Parseamos el texto a un diccionario de Python para verificar que sea JSON válido
        resultado_json = json.loads(texto_salida)

        json_ordenado = json.dumps(resultado_json, sort_keys=True)
        resultados_crudos.append(json_ordenado)
    except Exception as e:
        print(f"\n[X] Error en iteracion {i}: {e}")


conteo_respuestas = Counter(resultados_crudos)
total_variantes = len(conteo_respuestas)

if total_variantes == 1:
    print("Salida unica")
    print(json.dumps(json.loads(list(conteo_respuestas.keys())[0]), indent=2, ensure_ascii=False))
else:
    print(f"El modelo genero {total_variantes}")
    for indice, (resultado_str, cantidad) in enumerate(conteo_respuestas.most_common(), 1):
        print(f"")
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Herramienta
@tool
def contar_palabras(texto: str) -> int:
  """Cuenta el número exacto de palabras que tiene un texto dado."""
  return len(texto.split())


#Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "Eres un profesional de resúmenes de textos. Si te piden resumir,"
            " resume el texto en exactamente 3 viñetas sin añadir comentarios"
            " extra. Si te piden contar palabras, usa la herramienta"
            " contar_palabras."
        ),
    ),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{texto}"),
])

#ejemplos de textos que se pueden resumir en vinetas
texto1 = "Todo esto conduce a una conclusión: el complejo de Edipo no fue la causa de la guerra; la guerra fue la causa del complejo de Edipo (recordemos que la guerra misma no fue causa primera sino un producto del intento de controlar las presiones ecológica y reproductora). Aunque parezca un problema sin solución como el del huevo y la gallina, existen excelentes motivos científicos para rechazar las prioridades freudianas. Si comenzamos con el complejo de Edipo, no podemos explicar las variaciones de intensidad y de alcance de la guerra: ¿por qué algunos grupos son más belicosos que otros y por qué algunos practican formas externas y otros formas internas de guerra? Tampoco podemos explicar por qué el conjunto de las instituciones que forman el sistema de la supremacía masculina varía en esencia y en fuerza. Partiendo del complejo de Edipo, tampoco podemos explicar el origen de la agricultura, los caminos divergentes de las intensificaciones y los agotamientos en el Viejo y el Nuevo Mundo ni el origen del Estado. Pero si comenzamos con la presión reproductora, la intensificación y el agotamiento, podemos comprender los aspectos constantes y variables de la guerra. Y a partir de un conocimiento de las causas de las variaciones bélicas, podemos llegar a una comprensión de las causas de las variaciones de la organización familiar, las jerarquías sexuales y los papeles sexuales y, desde esta perspectiva, a una comprensión de las características constantes y variables del complejo de Edipo. Un principio admitido en la filosofía de la ciencia establece que si uno debe elegir entre dos teorías, merecerá prioridad aquella que resuelva más variables con el menor número de suposiciones independientes y no explicadas."
texto2 = "Hasta ahora nos hemos ocupado sobre todo del aspecto adquisitivo del aprendizaje, prescindiendo casi por completo de otros aspectos que, como la retención y el recuerdo, forman parte esencial del mismo, Un aprendizaje cabal comporta, no obstante, la retención de lo adquirido y la posibilidad de poder recordarlo en alguna manera, Nuestros hábitos adaptativos, la vivencia de nuestra identidad personal y la continuidad del mundo que nos rodea, procesos todos ellos relacionados con el aprendizaje, dependen esencialmente de esta capacidad de relacionar el pasado con el presente que se manifiesta en el recuerdo. Básicamente, recordar y olvidar pueden considerarse como el anverso y el reverso de un mismo proceso; el olvido consiste en la diferencia entre lo que se retiene y lo que se aprendió y, aunque no puede medirse de forma directa, no puede ser considerado como -una simple pérdida sino como el resultado de procesos activos, adquiriendo así cierta substantividad frente a la memoria. El recuerdo, a su vez, consiste en la evocación de contenidos fijados en un tiempo pasado, evocación que se apoya básicamente en procesos mentales de reconocimiento y reproducción de dichos contenidos.Para obtener una visión completa del aprendizaje debemos saber, pues, no solo cuánto retenemos de lo que hemos aprendido, sino también por qué olvidamos el resto, cómo deformamos nuestros recuerdos y por qué olvidamos precisamente unas cosas más que otras."

# 4. MODELO Y CADENAS LCEL

model = ChatOllama(model="llama3.2:3b", temperature=0)

# Modelo con herramienta enlazada para el turno de conteo
model_with_tools = model.bind_tools([contar_palabras])

parser = StrOutputParser()

# Cadena LCEL completa (prompt | modelo | parser) para Turnos 1 y 2
chain_texto = prompt | model | parser

# Cadena LCEL sin parser para detectar llamadas a herramientas en el Turno 3
chain_tools = prompt | model_with_tools

if __name__ == "__main__":
  historial = []

  # --- TURNO 1: Resumir texto1 usando parser ---
  print("=== TURNO 1: RESUMEN ===")
  entrada1 = f"Resume este texto en 3 viñetas:\n{texto2}"
  salida1 = chain_texto.invoke({"texto": entrada1, "historial": historial})
  print("SALIDA:\n" + salida1)

  # Guardamos el turno en la memoria
  historial.append(HumanMessage(content=entrada1))
  historial.append(AIMessage(content=salida1))

  # --- TURNO 2: Probar que recuerda el Turno 1 (Usa memoria + chain_texto) ---
  print("\n=== TURNO 2: PRUEBA DE MEMORIA ===")
  entrada2 = "Según el texto que acabas de resumir, ¿qué fue la causa de qué?"
  salida2 = chain_texto.invoke({"texto": entrada2, "historial": historial})
  print("PREGUNTA:", entrada2)
  print("SALIDA:\n" + salida2)

  # Guardamos el turno en la memoria
  historial.append(HumanMessage(content=entrada2))
  historial.append(AIMessage(content=salida2))

   # --- TURNO 3: Probar la herramienta real con el mismo texto de la memoria ---
  print("\n=== TURNO 3: PRUEBA DE HERRAMIENTA ===")
  entrada3 = (
      "¿Cuántas palabras exactas tiene el texto que acabas de resumir?"
  )
  respuesta3 = chain_tools.invoke({"texto": entrada3, "historial": historial})

  if respuesta3.tool_calls:
    llamada = respuesta3.tool_calls[0]
    print(f"------ Función invocada: {llamada['name']}")

    # Ejecutamos la función real en Python
    resultado_tool = contar_palabras.invoke(llamada["args"])
    print(f"----- Resultado devuelto: {resultado_tool}")

    # Actualizamos historial con la llamada y el resultado de la herramienta
    historial.append(HumanMessage(content=entrada3))
    historial.append(respuesta3)
    historial.append(
        ToolMessage(content=str(resultado_tool), tool_call_id=llamada["id"])
    )

    # El modelo genera la respuesta final usando el dato de la herramienta
    respuesta_final = model_with_tools.invoke(historial)
    print("SALIDA FINAL:\n" + parser.invoke(respuesta_final))
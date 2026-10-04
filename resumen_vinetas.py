#"Creando la cadena LCEL: prompt | modelo | parser"
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

#"escribimos la instruccion de sistema"
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesional de resumenes de textos. Resuma el texto en 3 vinetas. Devuelve SOLO las vinetas, sin comentarios."),
    ("human", "{texto}"),
])


#ejemplos de textos que se pueden resumir en vinetas
texto1 = "Todo esto conduce a una conclusión: el complejo de Edipo no fue la causa de la guerra; la guerra fue la causa del complejo de Edipo (recordemos que la guerra misma no fue causa primera sino un producto del intento de controlar las presiones ecológica y reproductora). Aunque parezca un problema sin solución como el del huevo y la gallina, existen excelentes motivos científicos para rechazar las prioridades freudianas. Si comenzamos con el complejo de Edipo, no podemos explicar las variaciones de intensidad y de alcance de la guerra: ¿por qué algunos grupos son más belicosos que otros y por qué algunos practican formas externas y otros formas internas de guerra? Tampoco podemos explicar por qué el conjunto de las instituciones que forman el sistema de la supremacía masculina varía en esencia y en fuerza. Partiendo del complejo de Edipo, tampoco podemos explicar el origen de la agricultura, los caminos divergentes de las intensificaciones y los agotamientos en el Viejo y el Nuevo Mundo ni el origen del Estado. Pero si comenzamos con la presión reproductora, la intensificación y el agotamiento, podemos comprender los aspectos constantes y variables de la guerra. Y a partir de un conocimiento de las causas de las variaciones bélicas, podemos llegar a una comprensión de las causas de las variaciones de la organización familiar, las jerarquías sexuales y los papeles sexuales y, desde esta perspectiva, a una comprensión de las características constantes y variables del complejo de Edipo. Un principio admitido en la filosofía de la ciencia establece que si uno debe elegir entre dos teorías, merecerá prioridad aquella que resuelva más variables con el menor número de suposiciones independientes y no explicadas."
texto2 = "Hasta ahora nos hemos ocupado sobre todo del aspecto adquisitivo del aprendizaje, prescindiendo casi por completo de otros aspectos que, como la retención y el recuerdo, forman parte esencial del mismo, Un aprendizaje cabal comporta, no obstante, la retención de lo adquirido y la posibilidad de poder recordarlo en alguna manera, Nuestros hábitos adaptativos, la vivencia de nuestra identidad personal y la continuidad del mundo que nos rodea, procesos todos ellos relacionados con el aprendizaje, dependen esencialmente de esta capacidad de relacionar el pasado con el presente que se manifiesta en el recuerdo. Básicamente, recordar y olvidar pueden considerarse como el anverso y el reverso de un mismo proceso; el olvido consiste en la diferencia entre lo que se retiene y lo que se aprendió y, aunque no puede medirse de forma directa, no puede ser considerado como -una simple pérdida sino como el resultado de procesos activos, adquiriendo así cierta substantividad frente a la memoria. El recuerdo, a su vez, consiste en la evocación de contenidos fijados en un tiempo pasado, evocación que se apoya básicamente en procesos mentales de reconocimiento y reproducción de dichos contenidos.Para obtener una visión completa del aprendizaje debemos saber, pues, no solo cuánto retenemos de lo que hemos aprendido, sino también por qué olvidamos el resto, cómo deformamos nuestros recuerdos y por qué olvidamos precisamente unas cosas más que otras."

#definimos el modelo que vamos a usar para resumir el texto en vinetas
model = ChatOllama(model="llama3.2:3b", temperature=0)

#"devolvemos un parser que se queda con el texto limpio"
parser = StrOutputParser()

#LCEL cadena usando el operador | para unir las tres piezas en una tubería. 
chain = prompt | model | parser

if __name__ == "__main__": 
    entrada = {"texto": texto1}
    salida = chain.invoke(entrada)
    print("ENTRADA:", entrada["texto"])
    print("SALIDA:", "\n" + salida)
   

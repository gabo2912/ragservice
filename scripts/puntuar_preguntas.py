
from src import retriever, config

preguntas = [
 "¿qué significa Jene, el agua?",
 "¿qué es Nete, el mundo o cosmos?",
 "¿qué representa Niwe, el viento?",
 "¿qué es Jain, el espacio de los espíritus?",
 "¿quién es Ronin, la anaconda?",
 "¿qué es el meraya?",
 "¿qué papel tiene la ayahuasca?",
 "¿qué hace el onaya/onanya?",
 "¿qué son los Yoshin, los espíritus?",
 "¿quiénes son los chaikoni?",
 "¿qué relación tienen los shipibo con el río Ucayali?",
 "¿qué es la cocha o laguna?",
 "¿qué importancia tiene la mujer shipiba?",
 "¿qué papel cumplen las plantas medicinales?",
 "¿cómo es la educación intercultural bilingüe?"
]

for q in preguntas:
    ch = retriever.buscar_chunks(q, k=config.RETRIEVE_K)
    s = min(c['score'] for c in ch) if ch else None
    est = 'RESPONDE' if (s is not None and s<=config.SCORE_THRESHOLD) else 'CALLA'
    print(f'  {round(s,3) if s else None}  {est}  {q}')

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.core.energia import PUNTOS_CORTE, obtener_vector_energias_promedio
from src.core.espectro import CLASES, FS, N_PUNTOS, calcularEspectro
from src.core.preprocesamiento import (
    AudioSilenciosoError, VENTANA_MEDIA_MOVIL, obtener_parametros_preprocesamiento, preparar_audio,
)

METODO_ENTRENAMIENTO = "promedio_vectores_energia"
RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data" / "processed"
RUTA_MODELO = RAIZ / "data" / "models" / "modelo_aves.json"


def preprocesar(audio):
    return preparar_audio(audio, FS)


def cargar_audios(clase):
    archivos = sorted((DIR_DATOS / clase).glob("*.wav"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron audios WAV para {clase}")
    audios = []
    for archivo in archivos:
        audio, frecuencia_muestreo = sf.read(archivo)
        if frecuencia_muestreo != FS:
            raise ValueError(f"{archivo.name}: frecuencia de {frecuencia_muestreo} Hz; se esperaba {FS} Hz")
        if len(audio) != N_PUNTOS:
            raise ValueError(f"{archivo.name}: debe tener {N_PUNTOS} muestras")
        try:
            audios.append(preprocesar(audio))
        except AudioSilenciosoError:
            print(f"Omitido por baja amplitud: {archivo.name}")
    if not audios:
        raise ValueError(f"No hay audios con nivel suficiente para {clase}")
    return audios


def entrenar():
    vectores_por_clase = []
    for clase in CLASES:
        audios = cargar_audios(clase)
        espectros = [calcularEspectro(audio) for audio in audios]
        vector_energias_promedio = obtener_vector_energias_promedio(espectros)
        vectores_por_clase.append(vector_energias_promedio)
        print(f"{clase}: {len(audios)} clips -> energías {vector_energias_promedio}")
    guardar_modelo(np.array(vectores_por_clase))


def guardar_modelo(vectores_por_clase):
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    limites_bandas = [0, *PUNTOS_CORTE, N_PUNTOS]
    modelo = {
        "fs": FS,
        "n_puntos": N_PUNTOS,
        "ventana_media_movil": VENTANA_MEDIA_MOVIL,
        "metodo_entrenamiento": METODO_ENTRENAMIENTO,
        "preprocesamiento": obtener_parametros_preprocesamiento(),
        "subbandas": [
            {
                "banda": indice_banda + 1,
                "bin_inicio": inicio_banda,
                "bin_fin": fin_banda,
                "hz_inicio": inicio_banda * FS / N_PUNTOS,
                "hz_fin": fin_banda * FS / N_PUNTOS,
            }
            for indice_banda, (inicio_banda, fin_banda)
            in enumerate(zip(limites_bandas[:-1], limites_bandas[1:]))
        ],
        "puntos_corte": list(PUNTOS_CORTE),
        "vectores_energia": {
            clase: vector_energias.tolist()
            for clase, vector_energias in zip(CLASES, vectores_por_clase)
        },
    }
    with open(RUTA_MODELO, "w", encoding="utf-8") as archivo_modelo:
        json.dump(modelo, archivo_modelo, indent=2, ensure_ascii=False)
    print(f"Modelo guardado en {RUTA_MODELO}")


def cargar_modelo(ruta=RUTA_MODELO):
    with open(ruta, encoding="utf-8") as archivo_modelo:
        modelo = json.load(archivo_modelo)
    if modelo.get("metodo_entrenamiento") != METODO_ENTRENAMIENTO:
        raise ValueError("El modelo usa otro procedimiento de entrenamiento. Vuelve a entrenar.")
    if modelo.get("preprocesamiento") != obtener_parametros_preprocesamiento():
        raise ValueError("El modelo usa otro preprocesamiento. Vuelve a entrenar.")
    if modelo["fs"] != FS or modelo["n_puntos"] != N_PUNTOS:
        raise ValueError("El modelo usa otra frecuencia de muestreo o longitud de señal")
    if modelo["puntos_corte"] != PUNTOS_CORTE:
        raise ValueError("Los puntos de corte del modelo no coinciden con energia.py")
    if modelo["ventana_media_movil"] != VENTANA_MEDIA_MOVIL:
        raise ValueError("El modelo usa otra ventana de media móvil")
    clases = list(modelo["vectores_energia"])
    vectores_por_clase = np.array([modelo["vectores_energia"][clase] for clase in clases])
    if set(clases) != set(CLASES) or vectores_por_clase.shape != (len(CLASES), len(PUNTOS_CORTE) + 1):
        raise ValueError("Las clases o las bandas del modelo no coinciden con la configuración actual")
    return clases, vectores_por_clase


if __name__ == "__main__":
    entrenar()

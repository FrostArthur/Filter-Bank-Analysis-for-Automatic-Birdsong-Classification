"""Fase de entrenamiento: genera models/modelo_aves.json

Ejecutar desde la raíz del proyecto:
    python -m src.core.entrenamiento
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.core.energia import PUNTOS_CORTE, obtenerVectorEnergias
from src.core.espectro import CLASES, FS, N_PUNTOS, EspectroPromedio, calcularEspectro
from src.scripts.filtro_media_movil import aplicar_filtro_media_movil

VENTANA_MM = 5
RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data" / "processed"
RUTA_MODELO = RAIZ / "data" / "models" / "modelo_aves.json"


def preprocesar(audio):
    """MISMO preprocesado para entrenamiento y tiempo real: mono -> media móvil."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return aplicar_filtro_media_movil(audio, VENTANA_MM)


def cargar_audios(clase):
    archivos = sorted((DIR_DATOS / clase).glob("*.wav"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron audios WAV para {clase}")
    audios = []
    for archivo in archivos:
        audio, fs = sf.read(archivo)
        if fs != FS:
            raise ValueError(f"{archivo.name}: fs={fs} Hz, se esperaba {FS} Hz")
        audios.append(preprocesar(audio))
    return audios


def entrenar():
    vectores = []
    for clase in CLASES:
        audios = cargar_audios(clase)
        espectro_prom = EspectroPromedio(audios)
        vector = obtenerVectorEnergias(espectro_prom)
        vectores.append(vector)
        print(f"{clase}: {len(audios)} clips -> energías {vector}")
    guardar_modelo(np.array(vectores))


def guardar_modelo(vectores):
    """Guarda el modelo en JSON legible: un vector de 4 energías por clase."""
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    limites = [0, *PUNTOS_CORTE, N_PUNTOS // 2]
    modelo = {
        "fs": FS,
        "n_puntos": N_PUNTOS,
        "ventana_media_movil": VENTANA_MM,
        "subbandas": [
            {
                "banda": i + 1,
                "bin_inicio": a,
                "bin_fin": b,
                "hz_inicio": a * FS / N_PUNTOS,
                "hz_fin": b * FS / N_PUNTOS,
            }
            for i, (a, b) in enumerate(zip(limites[:-1], limites[1:]))
        ],
        "puntos_corte": list(PUNTOS_CORTE),
        "vectores_energia": {
            clase: [float(e) for e in v] for clase, v in zip(CLASES, vectores)
        },
    }
    with open(RUTA_MODELO, "w", encoding="utf-8") as f:
        json.dump(modelo, f, indent=2, ensure_ascii=False)
    print(f"Modelo guardado en {RUTA_MODELO}")


# ----------------------------- Tiempo real -----------------------------
def cargar_modelo(ruta=RUTA_MODELO):
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    assert d["fs"] == FS and d["n_puntos"] == N_PUNTOS, \
        "El modelo se entrenó con otra fs o longitud de señal"
    assert tuple(d["puntos_corte"]) == tuple(PUNTOS_CORTE), \
        "Los puntos de corte del modelo no coinciden con energia.py"
    clases = list(d["vectores_energia"].keys())
    vectores = np.array([d["vectores_energia"][c] for c in clases])
    return clases, vectores


def clasificar(senal_mic, clases, vectores):
    """senal_mic: exactamente N_PUNTOS muestras (2 s a 44100 Hz)."""
    espectro = calcularEspectro(preprocesar(senal_mic))
    v = obtenerVectorEnergias(espectro)
    distancias = np.linalg.norm(vectores - v, axis=1)   # o np.abs(...).sum(axis=1)
    return clases[int(np.argmin(distancias))], distancias


if __name__ == "__main__":
    entrenar()
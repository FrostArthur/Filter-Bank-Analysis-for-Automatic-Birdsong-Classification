"""Clasificador por mínima distancia (fase de tiempo real)."""
import numpy as np
 
from src.core.energia import obtenerVectorEnergias
from src.core.entrenamiento import preprocesar
from src.core.espectro import N_PUNTOS, calcularEspectro
 
# Métricas de distancia que menciona el docente: euclidiana o resta absoluta
METRICAS = {
    "Euclidiana": lambda vectores, v: np.linalg.norm(vectores - v, axis=1),
    "Resta absoluta": lambda vectores, v: np.abs(vectores - v).sum(axis=1),
}
 
 
def analizar(senal, clases, vectores, metrica="Euclidiana"):
    """Procesa una señal de exactamente N_PUNTOS muestras (2 s a 44100 Hz).
 
    Aplica el MISMO procedimiento del entrenamiento (mono -> media móvil ->
    FFT -> vector de energías) y devuelve todo lo necesario para graficar.
    """
    senal = np.asarray(senal, dtype=np.float64)
    if senal.ndim == 2:
        senal = senal.mean(axis=1)
    if len(senal) != N_PUNTOS:
        raise ValueError(f"La señal debe tener {N_PUNTOS} muestras, tiene {len(senal)}")
 
    espectro = calcularEspectro(preprocesar(senal))
    vector = obtenerVectorEnergias(espectro)
    distancias = METRICAS[metrica](vectores, vector)
    return {
        "clase": clases[int(np.argmin(distancias))],
        "distancias": distancias,
        "vector": vector,
        "espectro": espectro,
        "senal": senal,
    }
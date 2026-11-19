import numpy as np
 
from src.core.energia import obtener_vector_energias
from src.core.entrenamiento import preprocesar
from src.core.espectro import N_PUNTOS, calcularEspectro
 

def distancia_euclidiana(vectores_clases, vector_entrada):
    diferencias = vectores_clases - vector_entrada
    suma_cuadrados = np.sum(diferencias ** 2, axis=1)
    return np.sqrt(suma_cuadrados)


def distancia_absoluta(vectores_clases, vector_entrada):
    diferencias = vectores_clases - vector_entrada
    return np.sum(np.abs(diferencias), axis=1)


METRICAS = {
    "Euclidiana": distancia_euclidiana,
    "Resta absoluta": distancia_absoluta,
}
 
 
def analizar(senal_entrada, clases, vectores_clases, metrica="Euclidiana"):
    senal_entrada = np.asarray(senal_entrada, dtype=np.float64)
    if senal_entrada.ndim == 2:
        senal_entrada = senal_entrada.mean(axis=1)
    if len(senal_entrada) != N_PUNTOS:
        raise ValueError(f"La señal debe tener {N_PUNTOS} muestras, tiene {len(senal_entrada)}")
 
    senal_filtrada = preprocesar(senal_entrada)
    espectro_entrada = calcularEspectro(senal_filtrada)
    vector_energias_entrada = obtener_vector_energias(espectro_entrada)
    distancias_por_clase = METRICAS[metrica](vectores_clases, vector_energias_entrada)
    indice_clase_mas_cercana = int(np.argmin(distancias_por_clase))
    return {
        "clase": clases[indice_clase_mas_cercana],
        "distancias": distancias_por_clase,
        "vector": vector_energias_entrada,
        "espectro": espectro_entrada,
        "senal": senal_entrada,
    }

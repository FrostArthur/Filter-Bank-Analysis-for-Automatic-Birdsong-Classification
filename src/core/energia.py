import numpy as np

PRIMER_PUNTO_CORTE = 238
SEGUNDO_PUNTO_CORTE = 4552
TERCER_PUNTO_CORTE = 14000


PUNTOS_CORTE = [PRIMER_PUNTO_CORTE, SEGUNDO_PUNTO_CORTE, TERCER_PUNTO_CORTE]


def obtener_vector_energias(espectro):
    energia_primera_banda = np.mean(espectro[:PRIMER_PUNTO_CORTE] ** 2)
    energia_segunda_banda = np.mean(espectro[PRIMER_PUNTO_CORTE:SEGUNDO_PUNTO_CORTE] ** 2)
    energia_tercera_banda = np.mean(espectro[SEGUNDO_PUNTO_CORTE:TERCER_PUNTO_CORTE] ** 2)
    energia_cuarta_banda = np.mean(espectro[TERCER_PUNTO_CORTE:] ** 2)

    return np.array([
        energia_primera_banda,
        energia_segunda_banda,
        energia_tercera_banda,
        energia_cuarta_banda,
    ])

def obtener_vector_energias_promedio(espectros):
    vectores_energias = [obtener_vector_energias(espectro) for espectro in espectros]
    if not vectores_energias:
        raise ValueError("Se necesita al menos un espectro para calcular el promedio.")
    return np.mean(vectores_energias, axis=0)

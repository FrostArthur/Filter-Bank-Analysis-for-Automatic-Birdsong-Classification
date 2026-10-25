#4 sub-bandas

import numpy as np

PRIMER_PUNTO_CORTE = 238
SEGUNDO_PUNTO_CORTE = 1283
TERCERA_PUNTO_CORTE = 4552

#Obtener vector de energías de la señal de audio
def obtenerVectorEnergias(senal):
    energia1 = np.mean(senal[:PRIMER_PUNTO_CORTE]**2)
    energia2 = np.mean(senal[PRIMER_PUNTO_CORTE:SEGUNDO_PUNTO_CORTE]**2)
    energia3 = np.mean(senal[SEGUNDO_PUNTO_CORTE:TERCERA_PUNTO_CORTE]**2)
    energia4 = np.mean(senal[TERCERA_PUNTO_CORTE:44100]**2)
    return np.array([energia1, energia2, energia3, energia4])

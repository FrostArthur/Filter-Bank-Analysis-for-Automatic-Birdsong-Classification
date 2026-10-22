import numpy as np
N_PUNTOS = 88200
freq = 44100

#Calcular el espectro de un audio (según el pseudocódigo del docente)
def calcularEspectro(audio):
    assert len(audio) == N_PUNTOS, "El audio debe ser exactamente de 2 segundos"
    espectro = np.fft.fft(audio)
    return np.abs(espectro, axis=0)

#Calcular el espectro promedio de la lista de audios de cada clase del banco de señales
def EspectroPromedio(lista_audios):
    magnitudes = [calcularEspectro(audio) for audio in lista_audios]
    return np.mean(magnitudes, axis=0)
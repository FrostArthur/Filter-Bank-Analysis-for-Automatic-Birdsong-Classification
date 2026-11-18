import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.scripts.filtro_media_movil import aplicar_filtro_media_movil

FRECUENCIA_CORTE_PASAALTOS = 300
ORDEN_FILTRO = 2
VENTANA_MEDIA_MOVIL = 5
UMBRAL_RMS = 0.001
RMS_OBJETIVO = 0.05
GANANCIA_MAXIMA = 50.0
PICO_MAXIMO = 0.95


class AudioSilenciosoError(ValueError):
    def __init__(self, rms):
        self.rms = rms
        super().__init__(f"Audio de baja amplitud: RMS = {rms:.6f}")


def convertir_a_mono(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("El audio debe contener muestras finitas de una señal mono o estéreo")
    return audio


def calcular_rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def medir_nivel_entrada(audio):
    audio_mono = convertir_a_mono(audio)
    return calcular_rms(audio_mono - audio_mono.mean())


def preparar_audio(audio, frecuencia_muestreo):
    audio_mono = convertir_a_mono(audio)
    audio_centrado = audio_mono - audio_mono.mean()
    rms_entrada = calcular_rms(audio_centrado)
    if rms_entrada < UMBRAL_RMS:
        raise AudioSilenciosoError(rms_entrada)

    filtro_pasaaltos = butter(ORDEN_FILTRO, FRECUENCIA_CORTE_PASAALTOS,
                             btype="highpass", fs=frecuencia_muestreo, output="sos")
    audio_sin_graves_de_fondo = sosfiltfilt(filtro_pasaaltos, audio_centrado)
    audio_filtrado = aplicar_filtro_media_movil(audio_sin_graves_de_fondo, VENTANA_MEDIA_MOVIL)
    rms_filtrado = calcular_rms(audio_filtrado)
    if rms_filtrado <= np.finfo(np.float64).eps:
        raise AudioSilenciosoError(rms_filtrado)

    pico_filtrado = float(np.max(np.abs(audio_filtrado)))
    ganancia = min(RMS_OBJETIVO / rms_filtrado, GANANCIA_MAXIMA, PICO_MAXIMO / pico_filtrado)
    return audio_filtrado * ganancia


def obtener_parametros_preprocesamiento():
    return {
        "version": 1,
        "quitar_dc": True,
        "frecuencia_corte_pasaaltos": FRECUENCIA_CORTE_PASAALTOS,
        "orden_filtro": ORDEN_FILTRO,
        "ventana_media_movil": VENTANA_MEDIA_MOVIL,
        "umbral_rms": UMBRAL_RMS,
        "rms_objetivo": RMS_OBJETIVO,
        "ganancia_maxima": GANANCIA_MAXIMA,
        "pico_maximo": PICO_MAXIMO,
    }

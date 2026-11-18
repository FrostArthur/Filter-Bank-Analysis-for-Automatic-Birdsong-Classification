import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf

from src.core.energia import PUNTOS_CORTE
from src.core.preprocesamiento import AudioSilenciosoError, preparar_audio

N_PUNTOS = 88200
FS = 44100
CLASES = ("brant_goose", "common_snipe")

#Calcular el espectro de un audio (según el pseudocódigo del docente)
def calcularEspectro(audio):
    assert len(audio) == N_PUNTOS, "El audio debe ser exactamente de 2 segundos"
    espectro = np.fft.fft(audio)
    return np.abs(espectro)

#Calcular el espectro promedio de la lista de audios de cada clase del banco de señales
def EspectroPromedio(lista_audios):
    magnitudes = [calcularEspectro(audio) for audio in lista_audios]
    return np.mean(magnitudes, axis=0)

#Graficar espectros promedio de cada clase del banco de señales
def graficar_espectros_promedio(k_max=None):
    directorio_datos = Path(__file__).resolve().parents[2] / "data" / "processed"
    k_fin = k_max if k_max else N_PUNTOS // 2
    k = np.arange(k_fin + 1)
    colores = {
        "brant_goose": "tab:blue",
        "common_snipe": "tab:orange",
    }

    figura, eje = plt.subplots(figsize=(12, 6))

    for clase in CLASES:
        archivos = sorted((directorio_datos / clase).glob("*.wav"))
        if not archivos:
            raise FileNotFoundError(f"No se encontraron audios WAV para {clase}")

        audios = []
        for archivo in archivos:
            audio, frecuencia_muestreo = sf.read(archivo)

            if frecuencia_muestreo != FS:
                raise ValueError(
                    f"{archivo.name} tiene frecuencia de muestreo "
                    f"{frecuencia_muestreo} Hz; se esperaba {FS} Hz"
                )

            try:
                audios.append(preparar_audio(audio, FS))
            except AudioSilenciosoError:
                continue

        if not audios:
            raise ValueError(f"No hay audios con nivel suficiente para {clase}")

        espectro_promedio = EspectroPromedio(audios)
        eje.plot(
            k,
            espectro_promedio[:k_fin + 1],
            label=clase,
            color=colores[clase],
        )

    for punto_corte in PUNTOS_CORTE:
        if punto_corte <= k_fin:
            eje.axvline(punto_corte, color="crimson", linestyle="--", linewidth=1.2)

    eje.set_title("Espectros promedio por tipo de ave")
    eje.set_xlabel("k")
    eje.set_ylabel("Magnitud")
    eje.set_xlim(0, k_fin)
    eje.grid(True)
    eje.legend(title="Tipo de ave")

    # Eje superior en Hz: f = k * FS / N
    eje_hz = eje.secondary_xaxis(
        "top",
        functions=(lambda k_: k_ * FS / N_PUNTOS, lambda f_: f_ * N_PUNTOS / FS),
    )
    eje_hz.set_xlabel("Frecuencia (Hz)")

    figura.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_espectros_promedio()

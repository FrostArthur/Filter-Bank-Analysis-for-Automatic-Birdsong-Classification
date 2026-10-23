import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf

from src.scripts.filtro_media_movil import aplicar_filtro_media_movil

N_PUNTOS = 88200
FS = 44100
CLASES = ("brant_goose", "common_snipe", "red_knot")

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
        "red_knot": "tab:green",
    }

    figura, eje = plt.subplots(figsize=(12, 6))

    for clase in CLASES:
        archivos = sorted((directorio_datos / clase).glob("*.wav"))
        if not archivos:
            raise FileNotFoundError(f"No se encontraron audios WAV para {clase}")

        audios = []
        for archivo in archivos:
            audio, frecuencia_muestreo = sf.read(archivo)
            audio = aplicar_filtro_media_movil(audio, 5)

            if frecuencia_muestreo != FS:
                raise ValueError(
                    f"{archivo.name} tiene frecuencia de muestreo "
                    f"{frecuencia_muestreo} Hz; se esperaba {FS} Hz"
                )

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            audios.append(audio)

        espectro_promedio = EspectroPromedio(audios)
        eje.plot(
            k,
            espectro_promedio[:k_fin + 1],
            label=clase,
            color=colores[clase],
        )

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
    
graficar_espectros_promedio()
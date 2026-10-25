"""Gráficas de la interfaz (solo matplotlib, sin dependencia de tkinter)."""
import numpy as np
 
from src.core.energia import PUNTOS_CORTE
from src.core.espectro import FS, N_PUNTOS
 
NOMBRES = {
    "brant_goose": "Barnacla carinegra",
    "common_snipe": "Agachadiza común",
    "red_knot": "Correlimos gordo",
}
COLORES_CLASE = {
    "brant_goose": "tab:blue",
    "common_snipe": "tab:orange",
    "red_knot": "tab:green",
}
COLORES_BANDA = ("tab:gray", "tab:purple", "tab:cyan", "tab:olive")
 
 
def nombre_clase(clase):
    return NOMBRES.get(clase, clase)
 
 
def _limites():
    bins = [0, *PUNTOS_CORTE, N_PUNTOS // 2]
    hz = [k * FS / N_PUNTOS for k in bins]
    return bins, hz
 
 
def crear_ejes(figura):
    return figura.subplots(2, 2).ravel()
 
 
def _titulos(ejes):
    ejes[0].set_title("Señal capturada (2 s)")
    ejes[0].set_xlabel("Tiempo (s)")
    ejes[0].set_ylabel("Amplitud")
    ejes[1].set_title("Espectro |X(f)| de la señal filtrada")
    ejes[1].set_xlabel("Frecuencia (Hz, escala log)")
    ejes[1].set_ylabel("Magnitud")
    ejes[2].set_title("Vector de energías por sub-banda")
    ejes[2].set_ylabel("Energía (escala log)")
    ejes[3].set_title("Distancia a cada clase (menor = más parecida)")
 
 
def dibujar_vacio(ejes):
    for ax in ejes:
        ax.clear()
        ax.grid(True, alpha=0.3)
    _titulos(ejes)
 
 
def dibujar(ejes, resultado, clases, vectores):
    for ax in ejes:
        ax.clear()
        ax.grid(True, alpha=0.3)
    _titulos(ejes)
    _, hz = _limites()
 
    # 1) Señal en el tiempo
    senal = resultado["senal"]
    ejes[0].plot(np.arange(len(senal)) / FS, senal, lw=0.6, color="tab:blue")
    tope = max(float(np.max(np.abs(senal))), 1e-3) * 1.1
    ejes[0].set_ylim(-tope, tope)
    ejes[0].set_xlim(0, len(senal) / FS)
 
    # 2) Espectro (mitad positiva) con las sub-bandas sombreadas
    mitad = N_PUNTOS // 2
    f = np.arange(1, mitad) * FS / N_PUNTOS
    ejes[1].plot(f, resultado["espectro"][1:mitad], lw=0.6, color="black")
    ejes[1].set_xscale("log")
    for i in range(4):
        ini = max(hz[i], 5.0)
        ejes[1].axvspan(ini, hz[i + 1], alpha=0.15, color=COLORES_BANDA[i],
                        label=f"Banda {i + 1}")
    ejes[1].set_xlim(5, FS / 2)
    ejes[1].legend(loc="upper right", fontsize=8)
 
    # 3) Vector de energías: captura vs. cada clase
    series = [("Captura", resultado["vector"], "black")]
    series += [(nombre_clase(c), v, COLORES_CLASE.get(c))
               for c, v in zip(clases, vectores)]
    ancho = 0.8 / len(series)
    x = np.arange(4)
    for i, (etiqueta, valores, color) in enumerate(series):
        desplazamiento = (i - (len(series) - 1) / 2) * ancho
        ejes[2].bar(x + desplazamiento, np.maximum(valores, 1e-6), ancho,
                    label=etiqueta, color=color)
    ejes[2].set_yscale("log")
    ejes[2].set_xticks(x)
    ejes[2].set_xticklabels(
        [f"B{i + 1}\n{hz[i]:.0f}–{hz[i + 1]:.0f} Hz" for i in range(4)], fontsize=8)
    ejes[2].legend(fontsize=8)
 
    # 4) Distancias
    d = resultado["distancias"]
    mejor = int(np.argmin(d))
    barras = ejes[3].bar(
        [nombre_clase(c) for c in clases], d,
        color=[COLORES_CLASE.get(c, "gray") for c in clases],
        edgecolor=["black" if i == mejor else "none" for i in range(len(clases))],
        linewidth=2.5,
    )
    ejes[3].bar_label(barras, fmt="%.1f", fontsize=9)
    ejes[3].margins(y=0.12)
    ejes[3].tick_params(axis="x", labelsize=9)
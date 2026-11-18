# Reconocimiento de aves por análisis de bancos de audio

Clasificador de cantos de aves en Python. A partir de un clip de audio de 2 s calcula su espectro con la FFT, lo resume en un **vector de energías por sub-bandas** y lo compara, por **mínima distancia**, con el vector de cada clase obtenido en el entrenamiento. Incluye una interfaz sencilla que graba desde el micrófono, clasifica y muestra las gráficas del proceso.

> **Estado:** dos especies: barnacla carinegra y agachadiza común. El entrenamiento promedia los vectores de energía calculados individualmente para cada audio.

---

## Cómo funciona

El mismo procesamiento se aplica en las dos fases, para que los vectores sean comparables:

```
audio de 2 s (88200 muestras a 44100 Hz)
   └─ mono
      └─ quitar desplazamiento DC y rechazar RMS < 0,001
         └─ filtro pasaaltos Butterworth de 300 Hz (orden 2, ida y vuelta)
            └─ filtro de media móvil (ventana M = 5)
               └─ ajustar amplitud hacia RMS = 0,05, con ganancia máxima 50 y pico máximo 0,95
                  └─ FFT  →  |X(k)|
                     └─ energía por sub-banda: E = (1/N) · Σ |X(k)|² (N = bins de la banda)
                        └─ vector de 4 componentes
```

- **Entrenamiento:** cada audio se convierte a mono, se filtra y se transforma con la FFT. Se calcula su vector de cuatro energías con la media de |X(k)|² de cada banda. Después se promedian esos vectores por clase, componente a componente, y los 2 vectores resultantes se guardan en `data/models/modelo_aves.json`.
- **Tiempo real:** se captura una señal de exactamente 2 s del micrófono, se procesa igual y se asigna la clase cuyo vector esté a **menor distancia** (euclidiana o resta absoluta, a elección en la interfaz).

El preprocesamiento está compartido en `src/core/preprocesamiento.py`. Se filtra y se ajusta la amplitud del audio antes de la FFT; no se normalizan las energías ni se cambian las distancias o el promedio por clase. El nivel se mide sin DC antes de amplificar: si el RMS es menor que `0.001`, se muestra «Sin clasificar: audio de baja amplitud» y se limpian los resultados anteriores. El mismo control excluye clips demasiado débiles del entrenamiento. El umbral y los límites de ganancia se configuran en ese archivo.

La ganancia intenta llevar el audio filtrado a RMS `0.05`, pero se reduce si superaría el factor `50` o un pico absoluto de `0.95`. Esos límites evitan amplificar sin límite o saturar el audio. El control de amplitud no distingue por sí solo ruido fuerte de canto.

### Sub-bandas

Con N = 88200 y fs = 44100 Hz, cada bin de la FFT equivale a 0,5 Hz (f = k · fs / N). Se conserva el cálculo actual: las primeras tres bandas llegan hasta el bin 14000 y la cuarta incluye todo el resto de la FFT, también las frecuencias negativas. La gráfica muestra la mitad positiva y marca los cortes con líneas verticales rojas.

| Banda | Bins (k), fin excluido | Frecuencia periódica (Hz), fin excluido |
|-------|---------------|-----------------|
| 1     | 0 – 238       | 0 – 119         |
| 2     | 238 – 4552    | 119 – 2276     |
| 3     | 4552 – 14000  | 2276 – 7000    |
| 4     | 14000 – 88200 | 7000 – 44100   |

### Especies

Los audios provienen de [xeno-canto.org](https://xeno-canto.org) (calidad A).

| Carpeta (`CLASES`) | Especie            | Nombre científico   |
|--------------------|--------------------|---------------------|
| `brant_goose`      | Barnacla carinegra | *Branta bernicla*   |
| `common_snipe`     | Agachadiza común   | *Gallinago gallinago* |

---

## Estructura del proyecto

```
.
├── .env                         # XENO_CANTO_API_KEY (no se sube a git)
├── .gitignore
├── README.md
├── data/                        # ignorado por git: cada quien lo descarga
│   ├── audio/<clase>/           #   grabaciones originales XC<id>.mp3 + metadata.csv
│   └── processed/<clase>/       #   clips de 2 s: XC<id>_<nn>.wav
├── models/
│   └── modelo_aves.json         # generado por el entrenamiento
└── src/
    ├── app.py                   # interfaz de tiempo real (tkinter)
    ├── graficas.py              # gráficas de la interfaz (matplotlib)
    ├── core/
    │   ├── energia.py           # vector de energías por sub-bandas
    │   ├── espectro.py          # FFT, espectro promedio y gráfica de espectros por clase
    │   ├── entrenamiento.py     # fase de entrenamiento + cargar_modelo / preprocesar
    │   └── clasificador.py      # análisis de una señal y mínima distancia
    └── scripts/
        ├── xenocantodownload.py # descarga de audios y fragmentación en clips de 2 s
        └── filtro_media_movil.py
```

---

## Requisitos

- Python 3.10 o superior (desarrollado con 3.14).
- Tkinter (viene con el instalador oficial de Python en Windows y macOS; en Linux: `sudo apt install python3-tk`).
- Un micrófono, para la fase de tiempo real.
- Una API key de xeno-canto (se obtiene desde tu cuenta en xeno-canto.org).

```bash
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz del proyecto:

```
XENO_CANTO_API_KEY=tu_clave_aqui
```

---

## Orden de ejecución

La carpeta `data/` está en `.gitignore`, así que después de clonar el repositorio **hay que generarla**. Todos los comandos se ejecutan **desde la raíz del proyecto**.

### 1. Descargar los audios

```bash
python -m src.scripts.xenocantodownload
```

Para cada especie busca grabaciones de calidad A, las descarga en `data/audio/<clase>/`, las parte en clips de 2 s sin solapamiento y conserva los más energéticos (los que más probablemente contienen canto). Los clips quedan en `data/processed/<clase>/`.

Parámetros ajustables al inicio del script:

| Constante             | Valor | Significado                                              |
|-----------------------|-------|----------------------------------------------------------|
| `MAX_GRABACIONES`     | 20    | grabaciones usadas por especie                           |
| `CLIPS_POR_GRABACION` | 5     | clips conservados por grabación                          |
| `UMBRAL_REL`          | 0,5   | un clip se descarta si su RMS < 50 % del mayor de la grabación |

Con los valores por defecto salen hasta 100 clips por clase (el mínimo exigido es 50).

### 2. (Opcional) Ver los espectros promedio

```bash
python -m src.core.espectro
```

Grafica el espectro promedio de cada clase. Con el zoom de la figura se pueden revisar los puntos de corte de las sub-bandas.

### 3. Entrenar

```bash
python -m src.core.entrenamiento
```

Lee los clips de `data/processed/`, calcula el vector de energías de cada clase y escribe `models/modelo_aves.json`. Ese archivo es legible: contiene `fs`, `n_puntos`, la ventana del filtro, las sub-bandas (en bins y en Hz) y un vector de 4 energías por clase.

### 4. Ejecutar la aplicación (tiempo real)

```bash
python main.py
```

- **Grabar 2 s y clasificar:** captura del micrófono y clasifica.
- **Cargar WAV…:** clasifica un archivo (44100 Hz, mínimo 2 s; usa los primeros 2 s). Útil para probar sin micrófono.
- **Distancia:** euclidiana o resta absoluta (reclasifica al cambiarla).
- **Gráficas:** señal en el tiempo, espectro con las sub-bandas, energías de la captura frente a las de cada clase y distancia a cada clase.

Si el modelo no existe, la aplicación avisa que hay que ejecutar primero el paso 3.

---

## Formato del modelo

```json
{
  "fs": 44100,
  "n_puntos": 88200,
  "ventana_media_movil": 5,
  "metodo_entrenamiento": "promedio_vectores_energia",
  "subbandas": [{"banda": 1, "bin_inicio": 0, "bin_fin": 238, "hz_inicio": 0.0, "hz_fin": 119.0}, "..."],
  "preprocesamiento": {"version": 1, "quitar_dc": true, "frecuencia_corte_pasaaltos": 300, "orden_filtro": 2, "ventana_media_movil": 5, "umbral_rms": 0.001, "rms_objetivo": 0.05, "ganancia_maxima": 50.0, "pico_maximo": 0.95},
  "puntos_corte": [238, 4552, 14000],
  "vectores_energia": {
    "brant_goose": [e1, e2, e3, e4],
    "common_snipe": [e1, e2, e3, e4]
  }
}
```

Al cargarlo, `cargar_modelo` verifica el método de entrenamiento, los parámetros de preprocesamiento, la frecuencia de muestreo, el número de muestras, el filtro, los cortes y las dimensiones de los vectores. Los modelos anteriores deben regenerarse con `python -m src.core.entrenamiento`.

---

## Cambiar una especie

1. En `src/scripts/xenocantodownload.py`, cambia la entrada correspondiente de `ESPECIES` (carpeta → género, especie).
2. En `src/core/espectro.py`, actualiza `CLASES` (y los colores de `graficar_espectros_promedio`).
3. En `src/graficas.py`, actualiza `NOMBRES` y `COLORES_CLASE`.
4. Vuelve a descargar (paso 1) y revisa los espectros promedio (paso 2): **los puntos de corte dependen de las clases**, así que hay que reescoger los de `PUNTOS_CORTE` en `src/core/energia.py` si las nuevas especies se separan en otras frecuencias.
5. Vuelve a entrenar (paso 3).

---

## Estado y pendientes

- **Volumen y graves de fondo:** se quita DC, se atenúan los graves con el pasaaltos y se ajusta el nivel del audio antes de la FFT. Los vectores de energía conservan su cálculo original.
- **Validación:** `python -m src.scripts.evaluar_preprocesamiento` compara cinco preparaciones mediante validación anidada: cinco pliegues externos y tres internos, separando por grabación XC. En 161 clips de 40 grabaciones, la selección pasó del 60,2 % al 87,6 % con distancia euclidiana y del 50,9 % al 87,6 % con resta absoluta. Hubo 20 errores con ambas métricas. Los parámetros se seleccionan usando únicamente los pliegues internos; el informe está en `output/evaluacion_preprocesamiento.json`.
- **Límite de la evaluación:** es validación interna de un banco ya explorado. Las grabaciones nuevas, el micrófono, la saturación y el ruido de fondo pueden producir resultados distintos. Los silencios se rechazan antes de calcular FFT y distancias.

---

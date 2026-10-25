# Reconocimiento de aves por análisis de bancos de audio

Clasificador de cantos de aves en Python. A partir de un clip de audio de 2 s calcula su espectro con la FFT, lo resume en un **vector de energías por sub-bandas** y lo compara, por **mínima distancia**, con el vector de cada clase obtenido en el entrenamiento. Incluye una interfaz sencilla que graba desde el micrófono, clasifica y muestra las gráficas del proceso.

> **Estado:** versión inicial. El pipeline completo (descarga, entrenamiento y tiempo real) funciona, pero una de las tres especies debe reemplazarse (ver [Estado y pendientes](#estado-y-pendientes)).

---

## Cómo funciona

El mismo procesamiento se aplica en las dos fases, para que los vectores sean comparables:

```
audio de 2 s (88200 muestras a 44100 Hz)
   └─ mono
      └─ filtro de media móvil (ventana M = 5)
         └─ FFT  →  |X(k)|
            └─ energía por sub-banda:  E = (1/N) · Σ |X(k)|²   (N = bins de la banda)
               └─ vector de 4 componentes
```

- **Entrenamiento:** para cada clase se promedian los espectros |X(k)| de todos sus clips y, del espectro promedio, se calcula el vector de energías. Los 3 vectores se guardan en `models/modelo_aves.json`.
- **Tiempo real:** se captura una señal de exactamente 2 s del micrófono, se procesa igual y se asigna la clase cuyo vector esté a **menor distancia** (euclidiana o resta absoluta, a elección en la interfaz).

### Sub-bandas

Con N = 88200 y fs = 44100 Hz, cada bin de la FFT equivale a 0,5 Hz (f = k · fs / N). Solo se usa la mitad positiva del espectro (bins 0 a 44100). Los puntos de corte se escogieron observando dónde se separan los espectros promedio de las clases.

| Banda | Bins (k)      | Frecuencia (Hz) |
|-------|---------------|-----------------|
| 1     | 0 – 238       | 0 – 119         |
| 2     | 238 – 1283    | 119 – 641,5     |
| 3     | 1283 – 4552   | 641,5 – 2276    |
| 4     | 4552 – 44100  | 2276 – 22050    |

### Especies

Los audios provienen de [xeno-canto.org](https://xeno-canto.org) (calidad A).

| Carpeta (`CLASES`) | Especie            | Nombre científico   |
|--------------------|--------------------|---------------------|
| `brant_goose`      | Barnacla carinegra | *Branta bernicla*   |
| `common_snipe`     | Agachadiza común   | *Gallinago gallinago* |
| `red_knot`         | Correlimos gordo   | *Calidris canutus*  |

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
pip install numpy matplotlib soundfile librosa requests python-dotenv sounddevice
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
python -m src.app
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
  "subbandas": [{"banda": 1, "bin_inicio": 0, "bin_fin": 238, "hz_inicio": 0.0, "hz_fin": 119.0}, "..."],
  "puntos_corte": [238, 1283, 4552],
  "vectores_energia": {
    "brant_goose": [e1, e2, e3, e4],
    "common_snipe": [e1, e2, e3, e4],
    "red_knot": [e1, e2, e3, e4]
  }
}
```

Al cargarlo, `cargar_modelo` verifica que `fs`, `n_puntos` y los puntos de corte coincidan con el código; si no, falla con un mensaje claro en lugar de clasificar mal.

---

## Cambiar una especie

1. En `src/scripts/xenocantodownload.py`, cambia la entrada correspondiente de `ESPECIES` (carpeta → género, especie).
2. En `src/core/espectro.py`, actualiza `CLASES` (y los colores de `graficar_espectros_promedio`).
3. En `src/graficas.py`, actualiza `NOMBRES` y `COLORES_CLASE`.
4. Vuelve a descargar (paso 1) y revisa los espectros promedio (paso 2): **los puntos de corte dependen de las clases**, así que hay que reescoger los de `PUNTOS_CORTE` en `src/core/energia.py` si las nuevas especies se separan en otras frecuencias.
5. Vuelve a entrenar (paso 3).

---

## Estado y pendientes

- **Correlimos gordo (`red_knot`):** la mayoría de sus grabaciones son ruido blanco, por lo que no sirve como clase. Hay que reemplazarlo por otra especie (ver [Cambiar una especie](#cambiar-una-especie)).
- **Sensibilidad al volumen:** las energías son absolutas, así que la distancia al micrófono, la ganancia y el volumen del parlante afectan el resultado. Posible mejora: normalizar los vectores (por ejemplo, dividirlos entre su suma) en entrenamiento y en tiempo real.
- **Ruido de baja frecuencia y offset DC:** afectan la banda 1. Posible mejora: restar la media de la señal en `preprocesar`.
- **Validación:** falta medir la precisión con clips que no hayan entrado al promedio de entrenamiento.

---

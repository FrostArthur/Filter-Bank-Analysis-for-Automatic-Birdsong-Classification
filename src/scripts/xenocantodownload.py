import csv
import os
import time
from dotenv import load_dotenv
from pathlib import Path
 
# pyrefly: ignore[missing-import]
import librosa
import numpy as np
import requests
import soundfile as sf
 
# ----------------------------- CONFIGURACIÓN -----------------------------
load_dotenv()
API_KEY = os.environ.get("XENO_CANTO_API_KEY")
URL_API = "https://xeno-canto.org/api/3/recordings"
 
FS = 44100
DURACION = 2
N = FS * DURACION
 
ESPECIES = {
    "common_snipe": ("Gallinago", "gallinago"),
    "red_knot": ("Calidris", "canutus"),
    "brant_goose": ("Branta", "bernicla"),
}
 
CALIDAD = "A"               
MAX_GRABACIONES = 20         
CLIPS_POR_GRABACION = 5      
UMBRAL_REL = 0.5            
PAUSA = 1.0 
 
RAIZ = Path(__file__).resolve().parent.parent.parent
DIR_RAW = RAIZ / "data" / "audio"
DIR_PROC = RAIZ / "data" / "processed"
 
 
def buscar(genero, especie):
    """Devuelve la lista de grabaciones (metadatos) de una especie."""
    consulta = f"gen:{genero} sp:{especie} q:{CALIDAD}"
    grabaciones, pagina = [], 1
    while len(grabaciones) < MAX_GRABACIONES * 3:   # margen por si alguna falla
        r = requests.get(
            URL_API,
            params={"query": consulta, "key": API_KEY, "page": pagina},
            timeout=30,
        )
        r.raise_for_status()
        datos = r.json()
        grabaciones += datos.get("recordings", [])
        if pagina >= int(datos.get("numPages", 1)):
            break
        pagina += 1
    return grabaciones
 
 
def descargar(url, destino):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    destino.write_bytes(r.content)
 
 
def fragmentar(y):
    """Parte la señal en clips de N muestras sin solapamiento y se queda con
    los CLIPS_POR_GRABACION más energéticos (los que más probablemente
    contienen canto y no silencio o ruido de fondo)."""
    n_clips = len(y) // N
    if n_clips == 0:
        return []
    clips = [y[i * N:(i + 1) * N] for i in range(n_clips)]
    rms = np.array([np.sqrt(np.mean(c ** 2)) for c in clips])
    if rms.max() == 0:
        return []
    validos = [i for i in range(n_clips) if rms[i] >= UMBRAL_REL * rms.max()]
    validos = sorted(validos, key=lambda i: -rms[i])[:CLIPS_POR_GRABACION]
    return [clips[i] for i in sorted(validos)]   # en orden temporal
 
 
def procesar_especie(carpeta, genero, especie):
    dir_raw = DIR_RAW / carpeta
    dir_proc = DIR_PROC / carpeta
    dir_raw.mkdir(parents=True, exist_ok=True)
    dir_proc.mkdir(parents=True, exist_ok=True)
 
    grabaciones = buscar(genero, especie)
    print(f"[{carpeta}] {len(grabaciones)} grabaciones encontradas")
 
    usadas, total_clips = 0, 0
    with open(dir_raw / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "autor", "licencia", "calidad", "duracion", "url"])
        for g in grabaciones:
            if usadas >= MAX_GRABACIONES:
                break
            xc_id = g["id"]
            ruta = dir_raw / f"XC{xc_id}.mp3"
            try:
                if not ruta.exists():
                    descargar(g["file"], ruta)
                    time.sleep(PAUSA)
                y, _ = librosa.load(ruta, sr=FS, mono=True)
            except Exception as e:
                print(f"  XC{xc_id}: omitida ({e})")
                continue
 
            clips = fragmentar(y)
            if not clips:
                print(f"  XC{xc_id}: sin clips útiles")
                continue
            for k, c in enumerate(clips):
                sf.write(dir_proc / f"XC{xc_id}_{k:02d}.wav", c, FS)
            w.writerow([xc_id, g.get("rec"), g.get("lic"), g.get("q"),
                        g.get("length"), g.get("url")])
            usadas += 1
            total_clips += len(clips)
            print(f"  XC{xc_id}: {len(clips)} clips")
 
    print(f"[{carpeta}] {usadas} grabaciones -> {total_clips} clips de 2 s\n")
 
 
if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Define la variable de entorno XENO_CANTO_API_KEY")
    for carpeta, (gen, sp) in ESPECIES.items():
        procesar_especie(carpeta, gen, sp)
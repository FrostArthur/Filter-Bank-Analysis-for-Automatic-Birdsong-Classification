"""Interfaz de tiempo real: graba 2 s del micrófono, clasifica y grafica.
 
Ejecutar desde la raíz del proyecto (después de entrenar):
    python -m src.app
 
Requiere:  pip install sounddevice
"""
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
 
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
 
from src.core.clasificador import METRICAS, analizar
from src.core.entrenamiento import cargar_modelo
from src.core.espectro import FS, N_PUNTOS
from src.graficas import crear_ejes, dibujar, dibujar_vacio, nombre_clase
 
UMBRAL_RMS = 1e-3   # por debajo de esto la señal es prácticamente silencio
 
class App:
    def __init__(self, raiz, clases, vectores):
        self.raiz = raiz
        self.clases = clases
        self.vectores = vectores
        self.cola = queue.Queue()
        self.ultima_senal = None   # (señal, origen) para reclasificar al cambiar la métrica
 
        raiz.title("Clasificador de aves por energía de sub-bandas")
 
        barra = ttk.Frame(raiz, padding=8)
        barra.pack(fill="x")
        self.btn_grabar = ttk.Button(barra, text="Grabar 2 s y clasificar", command=self.grabar)
        self.btn_grabar.pack(side="left")
        self.btn_wav = ttk.Button(barra, text="Cargar WAV…", command=self.cargar_wav)
        self.btn_wav.pack(side="left", padx=8)
        ttk.Label(barra, text="Distancia:").pack(side="left", padx=(16, 4))
        self.metrica = tk.StringVar(value="Euclidiana")
        combo = ttk.Combobox(barra, textvariable=self.metrica, values=list(METRICAS),
                             state="readonly", width=16)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _e: self.reclasificar())
        self.lbl_mic = ttk.Label(barra, text="")
        self.lbl_mic.pack(side="right")
 
        self.lbl_resultado = ttk.Label(raiz, text="—", font=("Segoe UI", 24, "bold"),
                                       anchor="center")
        self.lbl_resultado.pack(fill="x", pady=(4, 0))
        self.lbl_estado = ttk.Label(raiz, text="Pulsa «Grabar» para empezar.", anchor="center")
        self.lbl_estado.pack(fill="x")
 
        self.figura = Figure(figsize=(11, 7), constrained_layout=True)
        self.ejes = crear_ejes(self.figura)
        dibujar_vacio(self.ejes)
        self.canvas = FigureCanvasTkAgg(self.figura, master=raiz)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
 
        self._verificar_microfono()
 
    # ------------------------------------------------------------------ micrófono
    def _verificar_microfono(self):
        try:
            sd.check_input_settings(samplerate=FS, channels=1)
            nombre = sd.query_devices(kind="input")["name"]
            self.lbl_mic.config(text=f"Micrófono: {nombre}")
        except Exception as e:
            self.lbl_mic.config(text="Micrófono no disponible")
            self.lbl_estado.config(text=f"Problema con el micrófono a {FS} Hz: {e}")
 
    def grabar(self):
        self._bloquear(True)
        self.lbl_estado.config(text="Grabando 2 s… reproduce o emite el canto ahora")
        threading.Thread(target=self._capturar, daemon=True).start()
        self.raiz.after(100, self._revisar_cola)
 
    def _capturar(self):
        # Corre en un hilo aparte para no congelar la ventana durante los 2 s
        try:
            audio = sd.rec(N_PUNTOS, samplerate=FS, channels=1, dtype="float32")
            sd.wait()
            self.cola.put(("ok", audio[:, 0].astype(np.float64)))
        except Exception as e:
            self.cola.put(("error", str(e)))
 
    def _revisar_cola(self):
        try:
            tipo, dato = self.cola.get_nowait()
        except queue.Empty:
            self.raiz.after(100, self._revisar_cola)
            return
        self._bloquear(False)
        if tipo == "error":
            self.lbl_estado.config(text="Error al grabar")
            messagebox.showerror("Micrófono", f"No se pudo grabar:\n{dato}")
            return
        self._procesar(dato, "micrófono")
 
    # ------------------------------------------------------------------ archivos
    def cargar_wav(self):
        ruta = filedialog.askopenfilename(filetypes=[("Audio WAV", "*.wav")])
        if not ruta:
            return
        try:
            audio, fs = sf.read(ruta)
        except Exception as e:
            messagebox.showerror("WAV", f"No se pudo leer el archivo:\n{e}")
            return
        if fs != FS:
            messagebox.showerror("WAV", f"El archivo está a {fs} Hz; se esperaba {FS} Hz.")
            return
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if len(audio) < N_PUNTOS:
            messagebox.showerror("WAV", "El audio dura menos de 2 s.")
            return
        self._procesar(audio[:N_PUNTOS], Path(ruta).name)
 
    # ------------------------------------------------------------------ análisis
    def _procesar(self, senal, origen):
        self.ultima_senal = (senal, origen)
        rms = float(np.sqrt(np.mean(senal ** 2)))
        try:
            res = analizar(senal, self.clases, self.vectores, self.metrica.get())
        except Exception as e:
            messagebox.showerror("Análisis", str(e))
            return
        dibujar(self.ejes, res, self.clases, self.vectores)
        self.canvas.draw_idle()
 
        d = np.sort(res["distancias"])
        margen = (d[1] - d[0]) / d[1] if d[1] > 0 else 0.0
        self.lbl_resultado.config(text=nombre_clase(res["clase"]))
        estado = f"Fuente: {origen}  |  RMS = {rms:.4f}  |  margen sobre la 2.ª clase: {margen:.0%}"
        if rms < UMBRAL_RMS:
            estado += "  |  ATENCIÓN: señal muy débil (¿micrófono silenciado o muy lejos?)"
        self.lbl_estado.config(text=estado)
 
    def reclasificar(self):
        if self.ultima_senal is not None:
            self._procesar(*self.ultima_senal)
 
    def _bloquear(self, bloqueado):
        estado = "disabled" if bloqueado else "normal"
        self.btn_grabar.config(state=estado)
        self.btn_wav.config(state=estado)
 
 
def iniciar_app():
    raiz = tk.Tk()
    try:
        clases, vectores = cargar_modelo()
    except Exception as e:
        messagebox.showerror(
            "Modelo",
            f"No se pudo cargar el modelo:\n{e}\n\n"
            "Ejecuta primero:  python -m src.core.entrenamiento")
        raiz.destroy()
        return
    App(raiz, clases, vectores)
    raiz.mainloop()
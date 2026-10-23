import numpy as np

#Algoritmo del filtro de media móvil (peine)
def filtro_media_movil_matematico(x, M):
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("La señal debe ser un vector mono.")
    if M > len(x):
        raise ValueError("M no puede ser mayor que la cantidad de muestras.")

    y = x.astype(np.result_type(x.dtype, np.float64), copy=True)

    if M < len(x):
        acumulado = np.concatenate((
            np.array([0.0]),
            np.cumsum(x, dtype=np.result_type(x.dtype, np.float64)),
        ))
        y[M:] = (acumulado[M:len(x)] - acumulado[:len(x) - M]) / M

    return y

def aplicar_filtro_media_movil(x_entrada, M_puntos):
    M_puntos = int(M_puntos)
    if M_puntos < 1:
        raise ValueError("M debe ser un entero mayor o igual a 1.")
    return filtro_media_movil_matematico(x_entrada, M_puntos)
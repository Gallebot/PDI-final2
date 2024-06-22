import cv2
import tkinter as tk
from tkinter import ttk
import random
import numpy as np

# Función para pixelar la imagen
def pixelar(image, block_size):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            avg_color = block.mean(axis=(0, 1))
            new_image[y:y+block_size, x:x+block_size] = avg_color
    
    return new_image

# Función para aplicar el efecto de cristal a cuadros
def efecto_cristal_cuadros(image, block_size):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            random_y = y + random.randint(0, block_size - 1)
            random_x = x + random.randint(0, block_size - 1)

            random_y = min(random_y, height - 1)
            random_x = min(random_x, width - 1)

            new_image[y:y+block_size, x:x+block_size] = image[random_y, random_x]
    
    return new_image

# Funciones para la selección de canales
def escala_grises(image):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(height):
        for x in range(width):
            gray = int(0.299 * image[y, x][2] + 0.587 * image[y, x][1] + 0.114 * image[y, x][0])
            new_image[y, x] = [gray, gray, gray]

    return new_image

def canal_r(image):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(height):
        for x in range(width):
            new_image[y, x] = [0, 0, image[y, x][2]]

    return new_image

def canal_g(image):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(height):
        for x in range(width):
            new_image[y, x] = [0, image[y, x][1], 0]

    return new_image

def canal_b(image):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(height):
        for x in range(width):
            new_image[y, x] = [image[y, x][0], 0, 0]

    return new_image

# Función para calcular y mostrar el histograma
def calcular_histograma(image):
    height, width, _ = image.shape
    hist_height = 300
    hist_width = 256
    histogram_image = np.zeros((hist_height, hist_width * 3, 3), dtype=np.uint8)

    # Separar los canales
    b_hist = [0] * 256
    g_hist = [0] * 256
    r_hist = [0] * 256

    for y in range(height):
        for x in range(width):
            b_hist[image[y, x][0]] += 1
            g_hist[image[y, x][1]] += 1
            r_hist[image[y, x][2]] += 1

    # Normalizar el histograma
    b_hist = [x / max(b_hist) * hist_height for x in b_hist]
    g_hist = [x / max(g_hist) * hist_height for x in g_hist]
    r_hist = [x / max(r_hist) * hist_height for x in r_hist]

    # Dibujar el histograma
    for i in range(1, 256):
        cv2.line(histogram_image, (i - 1, hist_height - int(b_hist[i - 1])), (i, hist_height - int(b_hist[i])), (255, 0, 0), 1)
        cv2.line(histogram_image, (hist_width + i - 1, hist_height - int(g_hist[i - 1])), (hist_width + i, hist_height - int(g_hist[i])), (0, 255, 0), 1)
        cv2.line(histogram_image, (2 * hist_width + i - 1, hist_height - int(r_hist[i - 1])), (2 * hist_width + i, hist_height - int(r_hist[i])), (0, 0, 255), 1)

    return histogram_image

# Función para aplicar el filtro de sharpening (afilado)
def sharpen(image):
    height, width, _ = image.shape
    new_image = np.zeros_like(image)

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    for y in range(1, height-1):
        for x in range(1, width-1):
            region = image[y-1:y+2, x-1:x+2]
            new_image[y, x] = np.clip(np.sum(kernel[..., np.newaxis] * region, axis=(0, 1)), 0, 255)

    return new_image

# Función para actualizar la imagen según el efecto seleccionado
def update_image(effect, channel, image):
    if effect == "Pixelar":
        image = pixelar(image, 10)
    elif effect == "Cristal a cuadros":
        image = efecto_cristal_cuadros(image, 10)
    elif effect == "Afilado":
        image = sharpen(image)
    
    if channel == "Escala de grises":
        image = escala_grises(image)
    elif channel == "Canal Rojo":
        image = canal_r(image)
    elif channel == "Canal Verde":
        image = canal_g(image)
    elif channel == "Canal Azul":
        image = canal_b(image)

    return image

# Configuración de la captura de la cámara
cap = cv2.VideoCapture(1)

# Función para el bucle principal de la cámara
def video_loop():
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (320, 240))  # Reducir resolución para mejorar rendimiento
    effect = effect_var.get()
    channel = channel_var.get()
    frame = update_image(effect, channel, frame)
    
    histogram_image = calcular_histograma(frame)
    
    combined_frame = np.vstack((frame, cv2.resize(histogram_image, (320, 100))))
    
    cv2.imshow("Webcam", combined_frame)
    window.after(10, video_loop)

# Configuración de la interfaz gráfica
window = tk.Tk()
window.title("Efectos en Tiempo Real")

effect_var = tk.StringVar(value="None")
ttk.Label(window, text="Seleccione un efecto:").pack(pady=10)
effects = ["None", "Pixelate", "Glass", "Sharpen"]
for effect in effects:
    ttk.Radiobutton(window, text=effect, variable=effect_var, value=effect).pack(anchor=tk.W)

channel_var = tk.StringVar(value="None")
ttk.Label(window, text="Seleccione un canal:").pack(pady=10)
channels = ["None", "Grayscale", "Red", "Green", "Blue"]
for channel in channels:
    ttk.Radiobutton(window, text=channel, variable=channel_var, value=channel).pack(anchor=tk.W)

window.after(10, video_loop)
window.mainloop()

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
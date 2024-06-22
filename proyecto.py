import cv2
import tkinter as tk
from tkinter import ttk
import random
import math
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

# Función para aplicar la transformación cilíndrica
def transformacion_cilindrica(image, focal_length, axis='x'):
    height, width, _ = image.shape
    new_image = image.copy()

    if axis == 'x':
        for y in range(height):
            for x in range(width):
                theta = (x - width / 2) / focal_length
                h_ = (y - height / 2) / focal_length

                X = math.sin(theta)
                Y = h_
                Z = math.cos(theta)

                x_ = int(focal_length * X / Z + width / 2)
                y_ = int(focal_length * Y / Z + height / 2)

                if 0 <= x_ < width and 0 <= y_ < height:
                    new_image[y, x] = image[y_, x_]
    elif axis == 'y':
        for y in range(height):
            for x in range(width):
                theta = (y - height / 2) / focal_length
                w_ = (x - width / 2) / focal_length

                Y = math.sin(theta)
                X = w_
                Z = math.cos(theta)

                y_ = int(focal_length * Y / Z + height / 2)
                x_ = int(focal_length * X / Z + width / 2)

                if 0 <= x_ < width and 0 <= y_ < height:
                    new_image[y, x] = image[y_, x_]

    return new_image

# Función para la transformación elíptica
def transformacion_eliptica(image, a, axis='x'):
    height, width, _ = image.shape
    new_image = image.copy()

    if axis == 'x':
        for y in range(height):
            for x in range(width):
                x_norm = 2 * (x / width) - 1
                factor = math.sqrt(max(0, 1 - (a * x_norm) ** 2))
                new_x = int((factor * x_norm + 1) * width / 2)
                if 0 <= new_x < width:
                    new_image[y, new_x] = image[y, x]
    elif axis == 'y':
        for y in range(height):
            for x in range(width):
                y_norm = 2 * (y / height) - 1
                factor = math.sqrt(max(0, 1 - (a * y_norm) ** 2))
                new_y = int((factor * y_norm + 1) * height / 2)
                if 0 <= new_y < height:
                    new_image[new_y, x] = image[y, x]

    return new_image

# Función para la transformación abombada
def transformacion_abombada(image, strength):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    max_radius = math.sqrt(center_x**2 + center_y**2)
    new_image = image.copy()

    def abombar(x, y):
        dx, dy = x - center_x, y - center_y
        distance = math.sqrt(dx**2 + dy**2)
        if distance == 0:
            return x, y
        dist_abombada = (distance / max_radius) ** strength * max_radius
        ratio = dist_abombada / distance
        return int(center_x + dx * ratio), int(center_y + dy * ratio)

    for y in range(height):
        for x in range(width):
            new_x, new_y = abombar(x, y)
            if 0 <= new_x < width and 0 <= new_y < height:
                new_image[y, x] = image[new_y, new_x]

    return new_image

# Función para la transformación ondulada
def transformacion_ondulacion(image, frequency=10, amplitude=5):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    new_image = image.copy()

    def ondulacion(x, y):
        dx, dy = x - center_x, y - center_y
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        wave_effect = amplitude * math.sin(frequency * distance / max(width, height) * 2 * math.pi)
        new_x = int(center_x + (distance + wave_effect) * math.cos(angle))
        new_y = int(center_y + (distance + wave_effect) * math.sin(angle))
        return new_x, new_y

    for y in range(height):
        for x in range(width):
            new_x, new_y = ondulacion(x, y)
            if 0 <= new_x < width and 0 <= new_y < height:
                new_image[y, x] = image[new_y, new_x]

    return new_image

# Función para el efecto de ola
def efecto_onda(image, amplitude, frequency):
    height, width, _ = image.shape
    new_image = image.copy()

    for y in range(height):
        for x in range(width):
            new_x = int(x + amplitude * math.sin(2 * math.pi * y / frequency))
            new_y = int(y + amplitude * math.sin(2 * math.pi * x / frequency))
            new_x = min(max(new_x, 0), width - 1)
            new_y = min(max(new_y, 0), height - 1)
            new_image[y, x] = image[new_y, new_x]

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

# Función para aplicar el filtro de Sobel
def filtro_sobel(image):
    height, width, _ = image.shape
    new_image = np.zeros_like(image)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for y in range(1, height-1):
        for x in range(1, width-1):
            region = image[y-1:y+2, x-1:x+2, 0]
            gx = np.sum(sobel_x * region)
            gy = np.sum(sobel_y * region)
            magnitude = min(255, math.sqrt(gx**2 + gy**2))
            new_image[y, x] = [magnitude, magnitude, magnitude]

    return new_image

# Función para aplicar desenfoque gaussiano
def desenfoque(image, kernel_size=5):
    height, width, _ = image.shape
    new_image = np.zeros_like(image)

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    for y in range(kernel_size // 2, height - kernel_size // 2):
        for x in range(kernel_size // 2, width - kernel_size // 2):
            region = image[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]
            new_image[y, x] = np.sum(kernel[..., np.newaxis] * region, axis=(0, 1))

    return new_image

# Función para aplicar el efecto negativo
def negativo(image):
    return 255 - image

# Función para aplicar el filtro sepia
def sepia(image):
    height, width, _ = image.shape
    new_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            new_r = min(255, 0.393 * r + 0.769 * g + 0.189 * b)
            new_g = min(255, 0.349 * r + 0.686 * g + 0.168 * b)
            new_b = min(255, 0.272 * r + 0.534 * g + 0.131 * b)
            new_image[y, x] = [new_b, new_g, new_r]

    return new_image

# Función para actualizar la imagen según el efecto seleccionado
def update_image(effect, channel, image):
    if effect == "Pixelate":
        image = pixelar(image, 10)
    elif effect == "Glass":
        image = efecto_cristal_cuadros(image, 10)
    elif effect == "Cylindrical X":
        image = transformacion_cilindrica(image, 300, 'x')
    elif effect == "Cylindrical Y":
        image = transformacion_cilindrica(image, 300, 'y')
    elif effect == "Elliptical X":
        image = transformacion_eliptica(image, 0.75, 'x')
    elif effect == "Elliptical Y":
        image = transformacion_eliptica(image, 0.75, 'y')
    elif effect == "Bulge":
        image = transformacion_abombada(image, 1.5)
    elif effect == "Ripple":
        image = transformacion_ondulacion(image, 20, 15)
    elif effect == "Wave":
        image = efecto_onda(image, 15, 20)
    elif effect == "Sobel":
        image = filtro_sobel(image)
    elif effect == "Blur":
        image = desenfoque(image)
    elif effect == "Negative":
        image = negativo(image)
    elif effect == "Sepia":
        image = sepia(image)
    
    if channel == "Grayscale":
        image = escala_grises(image)
    elif channel == "Red":
        image = canal_r(image)
    elif channel == "Green":
        image = canal_g(image)
    elif channel == "Blue":
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
effects = ["None", "Pixelate", "Glass", "Cylindrical X", "Cylindrical Y", "Elliptical X", "Elliptical Y", "Bulge", "Ripple", "Wave", "Sobel", "Blur", "Negative", "Sepia"]
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

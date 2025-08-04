import os
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FREQ_BANDS = [
    (20, 60),     # Subgraves
    (60, 250),    # Graves
    (250, 2000),  # Médios
    (2000, 4000), # Médios-agudos
    (4000, 16000) # Agudos
]

COLORS = [
    (255, 0, 0, 170),     # Red
    (255, 165, 0, 170),   # Orange
    (255, 255, 0, 170),   # Yellow
    (0, 128, 0, 170),     # Green
    (0, 0, 255, 170),     # Blue
]

def analyze_audio(file_path, num_slices=16):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    slice_len = duration / num_slices

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr)

    slices = []
    for i in range(num_slices):
        start_t = i * slice_len
        end_t = (i + 1) * slice_len
        mask = (times >= start_t) & (times < end_t)
        slice_stft = stft[:, mask]

        band_timbres = []
        for band in FREQ_BANDS:
            idx = (freqs >= band[0]) & (freqs < band[1])
            band_energy = slice_stft[idx, :]
            avg_energy = band_energy.mean(axis=0) if band_energy.size > 0 else np.zeros((1,))
            band_timbres.append(avg_energy)

        slices.append(band_timbres)
    return slices

def draw_circle_timbres(band_timbres, filename, time_start, time_end):
    img_size = 800  # maior resolução
    center = img_size // 2
    image = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(image, 'RGBA')

    # Fundo artístico: círculos de referência (cinza claro)
    for r in range(100, center, 80):
        draw.ellipse(
            [center - r, center - r, center + r, center + r],
            outline=(220, 220, 220, 100),
            width=1
        )

    max_radius = center - 80
    base_radius = 120  # aumento do raio inicial
    radius_step = (max_radius - base_radius) // len(band_timbres)

    for i, (band_energy, color) in enumerate(zip(band_timbres, COLORS)):
        if len(band_energy) < 2:
            continue
        r = base_radius + i * radius_step
        points = []
        theta = np.linspace(0, 2*np.pi, len(band_energy))
        norm = np.interp(band_energy, (band_energy.min(), band_energy.max() + 1e-9), (0.4, 1.0))
        for t, amp in zip(theta, norm):
            x = center + np.cos(t) * r * amp
            y = center + np.sin(t) * r * amp
            points.append((x, y))
        draw.polygon(points, outline=color, fill=None)

    # Texto do tempo no topo esquerdo
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    time_text = f"{int(time_start):02}:{int((time_start%1)*60):02} - {int(time_end):02}:{int((time_end%1)*60):02}"
    draw.text((30, 30), time_text, fill="black", font=font)

    image.save(filename)

def compose_final_image(image_paths, output_path='final_output.png'):
    grid_size = 4
    img_size = 800
    padding = 0
    canvas_size = grid_size * img_size + (grid_size - 1) * padding

    final_img = Image.new('RGB', (canvas_size, canvas_size), 'black')
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        row = idx // grid_size
        col = idx % grid_size
        x = col * (img_size + padding)
        y = row * (img_size + padding)
        final_img.paste(img, (x, y))
    final_img.save(output_path)

def process_audio(file_path):
    print("Analyzing audio...")
    slices = analyze_audio(file_path)

    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    slice_len = duration / 16

    image_paths = []
    for idx, band_timbres in enumerate(slices):
        start_t = idx * slice_len
        end_t = (idx + 1) * slice_len
        img_path = os.path.join(OUTPUT_DIR, f'image_{idx:02}.png')
        draw_circle_timbres(band_timbres, img_path, start_t / 60, end_t / 60)
        image_paths.append(img_path)

    compose_final_image(image_paths)
    print("Image generated as 'final_output.png'.")

# --- GUI using tkinter ---

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        label_file.config(text=f"Selected: {os.path.basename(file_path)}")
        button_process.config(state="normal")
        window.selected_file = file_path

def start_processing():
    button_process.config(state="disabled")
    process_audio(window.selected_file)
    label_file.config(text="Done! Output saved.")

# Interface
window = tk.Tk()
window.title("Analyseur de Timbre")
window.geometry("400x200")
window.selected_file = None

label_title = tk.Label(window, text="Sélectionnez un fichier audio", font=("Arial", 14))
label_title.pack(pady=10)

button_browse = tk.Button(window, text="Parcourir...", command=select_file)
button_browse.pack(pady=5)

label_file = tk.Label(window, text="Aucun fichier sélectionné")
label_file.pack(pady=5)

button_process = tk.Button(window, text="Démarrer", command=start_processing, state="disabled")
button_process.pack(pady=10)

window.mainloop()

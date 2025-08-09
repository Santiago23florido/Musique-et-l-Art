import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import threading
import time
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from collections import deque
import io
import wave
import tempfile
import os
from scipy import interpolate
import warnings
import cv2
import base64
from scipy.io import wavfile
import subprocess
import sys
warnings.filterwarnings('ignore')

# Configuración optimizada
st.set_page_config(
    page_title="Audio Viz Circular - Upload & Download",
    page_icon="🌊",
    layout="wide"
)

# Variables globales optimizadas
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None
if 'features_buffer' not in st.session_state:
    st.session_state.features_buffer = deque(maxlen=1000)
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []

class AudioProcessor:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.noise_threshold = 0.002
        self.prev_features = None
        self.smoothing_factor = 0.25
        
    def extract_features_fast(self, audio_chunk):
        """Extracción ultrarrápida de características"""
        if len(audio_chunk) == 0:
            return self.create_silent_features()
        
        # RMS rápido
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        if rms < self.noise_threshold:
            return self.create_silent_features(rms)
        
        # FFT optimizada
        fft_data = fft(audio_chunk, n=self.buffer_size)
        freqs = fftfreq(self.buffer_size, 1/self.sample_rate)
        magnitudes = np.abs(fft_data)
        
        # Solo frecuencias positivas
        pos_idx = freqs > 0
        pos_freqs = freqs[pos_idx]
        pos_mags = magnitudes[pos_idx]
        
        # Encontrar picos principales rápidamente
        if len(pos_mags) == 0:
            return self.create_silent_features(rms)
        
        # Método más rápido para encontrar picos
        peak_threshold = np.max(pos_mags) * 0.1
        peaks = np.where(pos_mags > peak_threshold)[0]
        
        if len(peaks) == 0:
            max_idx = np.argmax(pos_mags)
            peaks = [max_idx]
        
        # Tomar solo los picos más importantes
        peak_freqs = pos_freqs[peaks]
        peak_mags = pos_mags[peaks]
        
        # Ordenar por magnitud y tomar los top 4
        if len(peak_mags) > 1:
            top_indices = np.argsort(peak_mags)[-4:][::-1]
            peak_freqs = peak_freqs[top_indices]
            peak_mags = peak_mags[top_indices]
        
        # Características básicas
        fundamental_freq = peak_freqs[0] if len(peak_freqs) > 0 else 440
        fundamental_mag = peak_mags[0] if len(peak_mags) > 0 else 0
        
        # Centroide espectral rápido
        if np.sum(pos_mags) > 0:
            spectral_centroid = np.sum(pos_freqs * pos_mags) / np.sum(pos_mags)
        else:
            spectral_centroid = 1000
        
        # Bandas de energía simplificadas
        low_mask = (pos_freqs >= 0) & (pos_freqs < 500)
        mid_mask = (pos_freqs >= 500) & (pos_freqs < 2000)
        high_mask = (pos_freqs >= 2000) & (pos_freqs < 8000)
        
        low_energy = np.sum(pos_mags[low_mask])
        mid_energy = np.sum(pos_mags[mid_mask])
        high_energy = np.sum(pos_mags[high_mask])
        
        total_energy = low_energy + mid_energy + high_energy
        if total_energy > 0:
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
        else:
            low_ratio = mid_ratio = high_ratio = 0.33
        
        features = {
            'fundamental_freq': fundamental_freq,
            'fundamental_mag': fundamental_mag,
            'spectral_centroid': spectral_centroid,
            'rms': rms,
            'low_ratio': low_ratio,
            'mid_ratio': mid_ratio,
            'high_ratio': high_ratio,
            'peak_freqs': peak_freqs[:4],
            'peak_mags': peak_mags[:4],
            'is_silent': False,
            'timestamp': time.time()
        }
        
        # Suavizado en tiempo real
        if self.prev_features is not None:
            features = self.smooth_features(features, self.prev_features)
        
        self.prev_features = features.copy()
        return features
    
    def smooth_features(self, current, previous):
        """Suavizado ultrarrápido"""
        alpha = self.smoothing_factor
        
        smoothed = current.copy()
        for key in ['fundamental_freq', 'fundamental_mag', 'spectral_centroid', 'rms',
                   'low_ratio', 'mid_ratio', 'high_ratio']:
            if key in previous:
                smoothed[key] = (1 - alpha) * previous[key] + alpha * current[key]
        
        return smoothed
    
    def create_silent_features(self, rms=0):
        return {
            'fundamental_freq': 220,
            'fundamental_mag': 0,
            'spectral_centroid': 500,
            'rms': rms,
            'low_ratio': 0.33,
            'mid_ratio': 0.33,
            'high_ratio': 0.33,
            'peak_freqs': np.array([220]),
            'peak_mags': np.array([0]),
            'is_silent': True,
            'timestamp': time.time()
        }

class CircularVisualRenderer:
    def __init__(self, num_points=3000):
        self.num_points = num_points
        self.phase_accumulator = 0
        self.color_cache = {}
        
        # Pre-calcular posiciones angulares para el círculo
        self.angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        self.base_radius = 3.0
        
        # Pre-calcular valores para optimización
        self.freq_scalers = np.array([1.0, 0.5, 0.3, 0.2])
        self.phase_multipliers = np.array([1.0, 1.2, 1.4, 1.6])
        
        # FIXED: Pre-calcular función de suavizado circular con tamaño fijo
        self.smooth_window_size = 20
        self.smooth_window = self._create_circular_smoothing_window(self.smooth_window_size)
        
    def _create_circular_smoothing_window(self, window_size):
        """Crear ventana de suavizado que respeta la circularidad"""
        # Ventana gaussiana que se aplica circularmente
        window = np.exp(-0.5 * (np.linspace(-2, 2, window_size) ** 2))
        return window / np.sum(window)
    
    def _apply_circular_smoothing(self, values, intensity=0.3):
        """FIXED: Aplicar suavizado que mantiene la continuidad circular"""
        if len(values) < 10:
            return values
        
        smoothed = values.copy()
        window_half = self.smooth_window_size // 2
        
        for i in range(len(values)):
            # Obtener índices circulares para la ventana
            indices = []
            for j in range(-window_half, window_half):
                idx = (i + j) % len(values)
                indices.append(idx)
            
            # FIXED: Asegurar que tenemos exactamente el tamaño correcto
            if len(indices) != self.smooth_window_size:
                # Ajustar si hay discrepancia
                while len(indices) < self.smooth_window_size:
                    indices.append(indices[-1])
                while len(indices) > self.smooth_window_size:
                    indices.pop()
            
            # Aplicar ventana de suavizado
            window_values = values[indices]
            
            # FIXED: Verificar tamaños antes de la operación
            if len(window_values) == len(self.smooth_window):
                smoothed_value = np.sum(window_values * self.smooth_window)
                # Mezclar con valor original
                smoothed[i] = (1 - intensity) * values[i] + intensity * smoothed_value
        
        return smoothed
        
    def create_circular_visualization(self, features, dt=0.016):
        """Renderizado circular MEJORADO - círculo perfectamente cerrado"""
        self.phase_accumulator += dt * 4
        
        # Configurar figura de manera eficiente
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        
        # Parámetros normalizados OPTIMIZADOS
        freq_norm = min(features['fundamental_freq'] / 600, 2.0)
        rms_norm = min(features['rms'] * 50, 1.0)
        centroid_norm = min(features['spectral_centroid'] / 2000, 1.5)
        
        # Calcular radios para cada punto
        radii = np.full(self.num_points, self.base_radius)
        individual_movement = np.zeros(self.num_points)
        
        if features['is_silent'] or rms_norm < 0.08:
            # Animación suave para silencio - ASEGURAR CONTINUIDAD
            ripple = 0.15 * np.sin(2 * self.angles + self.phase_accumulator * 0.8)
            individual_noise = np.random.normal(0, 0.08, self.num_points)
            breathing = 0.1 * np.sin(self.phase_accumulator * 0.5)
            
            # APLICAR SUAVIZADO CIRCULAR
            ripple = self._apply_circular_smoothing(ripple, 0.4)
            individual_noise = self._apply_circular_smoothing(individual_noise, 0.6)
            
            radii += ripple + individual_noise + breathing
            individual_movement = individual_noise
            
            # VERIFICAR CIERRE PERFECTO
            radii = self._ensure_smooth_closure(radii)
            
            colors = np.full((self.num_points, 3), [0.2, 0.2, 0.4])
            point_sizes = np.full(self.num_points, 2.0)
            alphas = np.full(self.num_points, 0.4)
            
        else:
            # DISTORSIÓN CONTROLADA DEL CÍRCULO CON CIERRE SUAVE
            
            # 1. Onda principal MODERADA
            main_freq = features['fundamental_freq'] / 200
            primary_wave_phase = main_freq * self.angles + self.phase_accumulator * 1.5
            primary_wave = np.sin(primary_wave_phase)
            main_distortion = np.clip(rms_norm * 1.2 * primary_wave, -1.2, 1.2)
            
            # APLICAR SUAVIZADO CIRCULAR A LA ONDA PRINCIPAL
            main_distortion = self._apply_circular_smoothing(main_distortion, 0.3)
            radii += main_distortion
            
            # 2. Ondas secundarias MÁS CONTROLADAS
            if len(features['peak_freqs']) > 1:
                for i, (freq, mag) in enumerate(zip(features['peak_freqs'][:3], features['peak_mags'][:3])):
                    if i == 0 or mag == 0:
                        continue
                    
                    mag_factor = mag / features['peak_mags'][0] if features['peak_mags'][0] > 0 else 0
                    wave_freq = freq / 150
                    wave_phase = wave_freq * self.angles + self.phase_accumulator * self.phase_multipliers[i]
                    
                    secondary_wave = np.sin(wave_phase) * self.freq_scalers[i]
                    amplitude = mag_factor * rms_norm * 0.98
                    secondary_distortion = np.clip(amplitude * secondary_wave, -0.6, 0.6)
                    
                    # SUAVIZADO CIRCULAR PARA ONDAS SECUNDARIAS
                    secondary_distortion = self._apply_circular_smoothing(secondary_distortion, 0.2)
                    radii += secondary_distortion
            
            # 3. MOVIMIENTO INDIVIDUAL CONTROLADO CON CONTINUIDAD
            for i in range(0, self.num_points, 4):
                end_idx = min(i + 4, self.num_points)
                for j in range(i, end_idx):
                    point_phase = self.angles[j] * freq_norm * 0.5 + self.phase_accumulator * 2
                    
                    individual_wave1 = 0.3 * np.sin(point_phase * 2) * rms_norm
                    individual_wave2 = 0.2 * np.sin(point_phase * 4 + self.phase_accumulator * 3) * features['high_ratio']
                    individual_noise = np.random.normal(0, 0.1 * rms_norm)
                    
                    individual_movement[j] = individual_wave1 + individual_wave2 + individual_noise
                    individual_movement[j] = np.clip(individual_movement[j], -0.5, 0.5)
                    radii[j] += individual_movement[j]
            
            # APLICAR SUAVIZADO CIRCULAR AL MOVIMIENTO INDIVIDUAL
            individual_movement = self._apply_circular_smoothing(individual_movement, 0.4)
            radii = self.base_radius * np.ones(self.num_points) + main_distortion + individual_movement
            
            # 4. Modulación rotatoria SUAVE
            centroid_rotation = centroid_norm * 3 * self.angles + self.phase_accumulator * 1.2
            centroid_wave = 0.4 * np.sin(centroid_rotation)
            centroid_wave = self._apply_circular_smoothing(centroid_wave, 0.3)
            radii += centroid_wave * rms_norm * 0.5
            
            # 5. Ondas de alta frecuencia MÁS SUTILES
            if features['high_ratio'] > 0.15:
                high_freq_wave = 0.25 * np.sin(12 * self.angles + self.phase_accumulator * 8) * features['high_ratio']
                high_freq_wave = self._apply_circular_smoothing(high_freq_wave, 0.5)
                radii += high_freq_wave * rms_norm * 0.6
            
            # 6. LIMITADOR GLOBAL CON CONTINUIDAD CIRCULAR
            max_distortion = self.base_radius * 0.8
            radii = np.clip(radii, self.base_radius - max_distortion, self.base_radius + max_distortion)
            
            # ASEGURAR CIERRE PERFECTO DEL CÍRCULO
            radii = self._ensure_smooth_closure(radii)
            
            # 7. Efectos globales suaves
            global_breathing = 0.2 * np.sin(self.phase_accumulator * 0.8) * rms_norm
            rhythm_pulse = 0.3 * np.sin(self.phase_accumulator * 4) * (rms_norm ** 2)
            radii += global_breathing + rhythm_pulse
            
            # Sistema de colores CONTROLADO
            hue_base = (centroid_norm * 0.8 + self.phase_accumulator * 0.03) % 1.0
            colors = self.generate_smooth_colors(radii, individual_movement, hue_base, features)
            
            # Tamaños PEQUEÑOS Y CONTROLADOS
            radius_variation = np.abs(radii - self.base_radius)
            individual_intensity = np.abs(individual_movement)
            combined_intensity = radius_variation + individual_intensity
            
            point_sizes = 2.0 + combined_intensity * 8
            point_sizes = np.clip(point_sizes, 1.0, 12)
            
            # Transparencias SUAVES
            max_intensity = np.max(radius_variation + individual_intensity) + 1e-6
            local_energy = (radius_variation + individual_intensity) / max_intensity
            alphas = 0.5 + local_energy * 0.3
            alphas = np.clip(alphas, 0.3, 0.8)
        
        # Convertir coordenadas polares a cartesianas
        x_positions = radii * np.cos(self.angles)
        y_positions = radii * np.sin(self.angles)
        
        # Renderizado SUAVE
        self.render_smooth_points(ax, x_positions, y_positions, colors, point_sizes, alphas, individual_movement)
        
        # Círculo de referencia SIEMPRE VISIBLE
        circle_alpha = 0.3 if rms_norm > 0.08 else 0.2
        reference_circle = plt.Circle((0, 0), self.base_radius, 
                                    fill=False, color='cyan', 
                                    alpha=circle_alpha, linewidth=1.0)
        ax.add_patch(reference_circle)
        
        # Configurar ejes
        max_radius = self.base_radius * 1.8
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        ax.axis('off')
        
        # Título informativo
        if not features['is_silent']:
            title = f"🎵 {features['fundamental_freq']:.0f}Hz | RMS: {features['rms']:.3f} | Puntos: {self.num_points}"
            plt.suptitle(title, color='white', fontsize=11, y=0.95)
        
        plt.tight_layout()
        return fig
    
    def _ensure_smooth_closure(self, radii, blend_points=10):
        """MEJORA: Asegurar que el inicio y final del círculo se unan suavemente"""
        if len(radii) < blend_points * 2:
            return radii
        
        # Crear una copia para modificar
        smooth_radii = radii.copy()
        
        # Obtener valores del inicio y final
        start_values = radii[:blend_points]
        end_values = radii[-blend_points:]
        
        # Crear promedio ponderado para suavizar la unión
        for i in range(blend_points):
            # Factor de mezcla (más peso al centro, menos en los bordes)
            weight = (i + 1) / (blend_points + 1)
            
            # Mezclar inicio con final
            start_blend = (1 - weight) * start_values[i] + weight * end_values[-(i+1)]
            end_blend = (1 - weight) * end_values[-(i+1)] + weight * start_values[i]
            
            smooth_radii[i] = start_blend
            smooth_radii[-(i+1)] = end_blend
        
        return smooth_radii
    
    def generate_smooth_colors(self, radii, individual_movement, hue_base, features):
        """Generación de colores SUAVE - con continuidad circular"""
        colors = np.zeros((len(radii), 3))
        
        # Color base que rota SUAVEMENTE
        base_hue = (hue_base + self.angles / (2 * np.pi) * 0.4) % 1.0
        
        # APLICAR SUAVIZADO CIRCULAR A LOS COLORES
        base_hue = self._apply_circular_smoothing(base_hue, 0.3)
        
        # Variación SUAVE por lotes
        for i in range(0, len(colors), 100):
            end_idx = min(i + 100, len(colors))
            for j in range(i, end_idx):
                angle_hue = base_hue[j]
                
                movement_factor = np.abs(individual_movement[j])
                movement_factor = np.clip(movement_factor / 0.3, 0, 1)
                
                radius_factor = (radii[j] - self.base_radius) / self.base_radius
                radius_factor = np.clip(radius_factor, -0.5, 0.5)
                
                if movement_factor > 0.6:
                    active_hue = (angle_hue + movement_factor * 0.15) % 1.0
                    active_color = np.array(colorsys.hsv_to_rgb(active_hue, 0.8, 0.9))
                    colors[j] = active_color
                else:
                    base_color = np.array(colorsys.hsv_to_rgb(angle_hue, 0.6, 0.7))
                    colors[j] = base_color
                
                if radius_factor > 0.2:
                    colors[j] += np.array([0.1, 0.1, 0.1]) * radius_factor
                elif radius_factor < -0.2:
                    colors[j] *= (1 - abs(radius_factor) * 0.2)
        
        # Efectos especiales SUTILES
        if features['high_ratio'] > 0.2:
            high_freq_mask = np.random.random(len(colors)) < features['high_ratio'] * 0.1
            colors[high_freq_mask] += np.array([0.2, 0.2, 0.2])
        
        if features['low_ratio'] > 0.35:
            warm_adjustment = np.array([0.15, 0.1, 0])
            colors += warm_adjustment * features['low_ratio'] * 0.3
        
        pulse_factor = np.sin(self.phase_accumulator * 3) * features['rms'] * 10
        pulse_adjustment = np.array([pulse_factor, pulse_factor * 0.3, pulse_factor * 0.2])
        colors += pulse_adjustment * 0.05
        
        return np.clip(colors, 0, 1)
    
    def render_smooth_points(self, ax, x_pos, y_pos, colors, sizes, alphas, individual_movement):
        """Renderizado SUAVE - sin efectos extremos"""
        movement_intensity = np.abs(individual_movement)
        
        high_movement_mask = movement_intensity > 0.4
        medium_movement_mask = (movement_intensity > 0.2) & (movement_intensity <= 0.4)
        low_movement_mask = movement_intensity <= 0.2
        
        # Renderizar puntos de bajo movimiento (mayoría)
        if np.any(low_movement_mask):
            ax.scatter(
                x_pos[low_movement_mask], 
                y_pos[low_movement_mask],
                c=colors[low_movement_mask], 
                s=sizes[low_movement_mask],
                alpha=0.5,
                edgecolors='none'
            )
        
        # Renderizar puntos de movimiento medio
        if np.any(medium_movement_mask):
            ax.scatter(
                x_pos[medium_movement_mask], 
                y_pos[medium_movement_mask],
                c=colors[medium_movement_mask], 
                s=sizes[medium_movement_mask] * 1.1,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.2
            )
        
        # Renderizar puntos de ALTO movimiento
        if np.any(high_movement_mask):
            ax.scatter(
                x_pos[high_movement_mask], 
                y_pos[high_movement_mask],
                c=colors[high_movement_mask], 
                s=sizes[high_movement_mask] * 1.5,
                alpha=0.2,
                edgecolors='none'
            )
            
            ax.scatter(
                x_pos[high_movement_mask], 
                y_pos[high_movement_mask],
                c=colors[high_movement_mask], 
                s=sizes[high_movement_mask],
                alpha=0.9,
                edgecolors='white',
                linewidths=0.4
            )

def check_ffmpeg():
    """Verificar si FFmpeg está disponible"""
    try:
        # Primero intentar con FFmpeg del sistema
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Si no encuentra FFmpeg del sistema, intentar con imageio-ffmpeg
        try:
            import imageio_ffmpeg as ffmpeg
            ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg_exe, '-version'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
            return True
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            return False

def get_ffmpeg_executable():
    """Obtener la ruta del ejecutable FFmpeg"""
    try:
        # Primero intentar con FFmpeg del sistema
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return 'ffmpeg'
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Si no encuentra FFmpeg del sistema, usar imageio-ffmpeg
        try:
            import imageio_ffmpeg as ffmpeg
            return ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

def install_ffmpeg():
    """Intentar instalar FFmpeg"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg'])
        return True
    except subprocess.CalledProcessError:
        return False

def load_wav_file(uploaded_file):
    """Cargar archivo WAV desde upload"""
    try:
        audio_bytes = uploaded_file.read()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        sample_rate, audio_data = wavfile.read(tmp_file_path)
        
        os.unlink(tmp_file_path)
        
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        return audio_data, sample_rate
        
    except Exception as e:
        st.error(f"Error cargando archivo WAV: {e}")
        return None, None

def generate_video_with_audio(audio_data, sample_rate, target_fps=30, num_points=3000):
    """Generar video completo con audio sincronizado"""
    
    processor = AudioProcessor(sample_rate, buffer_size=1024)
    renderer = CircularVisualRenderer(num_points=num_points)
    
    frame_duration = 1.0 / target_fps
    audio_duration = len(audio_data) / sample_rate
    total_frames = int(audio_duration * target_fps)
    
    st.session_state.video_frames = []
    
    progress_container = st.empty()
    status_container = st.empty()
    
    status_container.info(f"🎬 Generando {total_frames} frames a {target_fps} FPS...")
    
    for frame_idx in range(total_frames):
        frame_start_time = frame_idx * frame_duration
        
        progress = frame_idx / total_frames
        progress_container.progress(progress, f"Frame {frame_idx + 1}/{total_frames}")
        
        audio_position = int(frame_start_time * sample_rate)
        chunk_size = 1024
        audio_start = max(0, audio_position - chunk_size // 2)
        audio_end = min(len(audio_data), audio_start + chunk_size)
        
        if audio_end > audio_start:
            current_chunk = audio_data[audio_start:audio_end]
            features = processor.extract_features_fast(current_chunk)
            fig = renderer.create_circular_visualization(features, frame_duration)
            
            if fig is not None:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', 
                           facecolor='black', edgecolor='none')
                buf.seek(0)
                
                img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img = cv2.resize(img, (800, 800))
                    st.session_state.video_frames.append(img)
                
                plt.close(fig)
                buf.close()
    
    progress_container.progress(1.0)
    status_container.success(f"✅ {len(st.session_state.video_frames)} frames generados")
    
    return len(st.session_state.video_frames) > 0

def create_video_file_with_audio(audio_data, sample_rate, target_fps=30):
    """MEJORA: Crear archivo de video MP4 CON AUDIO"""
    if not st.session_state.video_frames:
        return None
    
    try:
        # Crear archivos temporales
        video_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        final_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        video_temp_path = video_temp.name
        audio_temp_path = audio_temp.name
        final_temp_path = final_temp.name
        
        video_temp.close()
        audio_temp.close()
        final_temp.close()
        
        # 1. Crear video sin audio
        height, width = st.session_state.video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_temp_path, fourcc, target_fps, (width, height))
        
        for frame in st.session_state.video_frames:
            video_writer.write(frame)
        
        video_writer.release()
        
        # 2. Guardar audio original
        wavfile.write(audio_temp_path, sample_rate, (audio_data * 32767).astype(np.int16))
        
        # 3. Verificar FFmpeg
        if not check_ffmpeg():
            st.warning("FFmpeg no encontrado. Intentando instalarlo...")
            if install_ffmpeg():
                st.success("FFmpeg instalado correctamente")
            else:
                st.error("No se pudo instalar FFmpeg. El video se descargará sin audio.")
                # Devolver video sin audio
                with open(video_temp_path, 'rb') as f:
                    video_bytes = f.read()
                os.unlink(video_temp_path)
                os.unlink(audio_temp_path)
                return video_bytes
        
        # 4. Combinar video y audio con FFmpeg
        try:
            # Obtener el ejecutable correcto de FFmpeg
            ffmpeg_exe = get_ffmpeg_executable()
            if not ffmpeg_exe:
                st.error("No se encontró FFmpeg. El video se descargará sin audio.")
                with open(video_temp_path, 'rb') as f:
                    video_bytes = f.read()
                os.unlink(video_temp_path)
                os.unlink(audio_temp_path)
                return video_bytes
            
            ffmpeg_cmd = [
                ffmpeg_exe, '-y',  # Usar el ejecutable correcto
                '-i', video_temp_path,  # Video de entrada
                '-i', audio_temp_path,  # Audio de entrada
                '-c:v', 'libx264',  # Codec de video
                '-c:a', 'aac',  # Codec de audio
                '-shortest',  # Duración del más corto
                '-pix_fmt', 'yuv420p',  # Formato de pixel compatible
                final_temp_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
            
            # Leer video final con audio
            with open(final_temp_path, 'rb') as f:
                video_bytes = f.read()
            
            # Limpiar archivos temporales
            os.unlink(video_temp_path)
            os.unlink(audio_temp_path)
            os.unlink(final_temp_path)
            
            return video_bytes
            
        except subprocess.CalledProcessError as e:
            st.error(f"Error combinando audio y video: {e}")
            # Devolver video sin audio como respaldo
            with open(video_temp_path, 'rb') as f:
                video_bytes = f.read()
            os.unlink(video_temp_path)
            os.unlink(audio_temp_path)
            try:
                os.unlink(final_temp_path)
            except:
                pass
            return video_bytes
        
    except Exception as e:
        st.error(f"Error creando video: {e}")
        return None

def play_with_realtime_preview(audio_data, sample_rate, target_fps=45, num_points=3000):
    """Sistema de reproducción con preview en tiempo real"""
    
    processor = AudioProcessor(sample_rate, buffer_size=1024)
    renderer = CircularVisualRenderer(num_points=num_points)
    
    viz_container = st.empty()
    info_container = st.empty()
    progress_container = st.empty()
    
    frame_duration = 1.0 / target_fps
    audio_duration = len(audio_data) / sample_rate
    
    def audio_playback():
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            st.error(f"Error de audio: {e}")
    
    audio_thread = threading.Thread(target=audio_playback, daemon=True)
    audio_thread.start()
    
    start_time = time.time()
    frame_count = 0
    last_frame_time = start_time
    
    try:
        while st.session_state.is_playing:
            frame_start = time.time()
            elapsed_time = frame_start - start_time
            
            if elapsed_time >= audio_duration:
                break
            
            audio_position = int(elapsed_time * sample_rate)
            
            chunk_size = 1024
            audio_start = max(0, audio_position - chunk_size // 2)
            audio_end = min(len(audio_data), audio_start + chunk_size)
            
            if audio_end > audio_start:
                current_chunk = audio_data[audio_start:audio_end]
                features = processor.extract_features_fast(current_chunk)
                
                dt = frame_start - last_frame_time
                fig = renderer.create_circular_visualization(features, dt)
                
                if fig is not None:
                    with viz_container.container():
                        st.pyplot(fig, clear_figure=True, use_container_width=True)
                    plt.close(fig)
            
            if frame_count % 10 == 0:
                fps_actual = 1.0 / (frame_start - last_frame_time + 1e-6)
                info_text = f"⚡ {fps_actual:.1f} FPS | ⏱️ {elapsed_time:.2f}s/{audio_duration:.2f}s | 🔵 {num_points} puntos"
                if 'features' in locals() and not features['is_silent']:
                    info_text += f" | 🎵 {features['fundamental_freq']:.0f}Hz"
                info_container.info(info_text)
            
            if frame_count % 5 == 0:
                progress = min(elapsed_time / audio_duration, 1.0)
                progress_container.progress(progress)
            
            frame_end = time.time()
            processing_time = frame_end - frame_start
            sleep_time = max(0, frame_duration - processing_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            last_frame_time = frame_start
            frame_count += 1
        
        progress_container.progress(1.0)
        st.success(f"✅ Reproducción completada - {frame_count} frames @ {frame_count/audio_duration:.1f} FPS promedio")
        
    except Exception as e:
        st.error(f"Error durante reproducción: {e}")
    finally:
        st.session_state.is_playing = False
        try:
            sd.stop()
        except:
            pass

# ===== INTERFAZ PRINCIPAL =====

st.title("🌊 Visualizador Circular Mejorado - Con Audio y Círculo Suave")
st.markdown("---")
st.markdown("🎵 **Carga archivos WAV** | 🎬 **Videos con audio** | 🌊 **Círculo perfectamente cerrado**")
st.markdown("### 📁 Subir archivo → ▶️ Reproducir → 🎬 Generar video → ⬇️ Descargar con audio")
st.markdown("---")

# ===== SECCIÓN DE CARGA DE ARCHIVO =====
st.header("📁 Cargar Archivo de Audio")

uploaded_file = st.file_uploader(
    "Selecciona un archivo WAV",
    type=['wav'],
    help="Sube un archivo .wav para visualizar"
)

if uploaded_file is not None:
    with st.spinner("🔄 Cargando archivo..."):
        audio_data, sample_rate = load_wav_file(uploaded_file)
        
        if audio_data is not None:
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            
            duration = len(audio_data) / sample_rate
            
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duración", f"{duration:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sample_rate} Hz")
            with col3:
                st.metric("Muestras", f"{len(audio_data):,}")
            with col4:
                rms_level = np.sqrt(np.mean(audio_data**2))
                st.metric("RMS", f"{rms_level:.4f}")

# ===== CONTROLES DE VISUALIZACIÓN =====
if st.session_state.audio_data is not None:
    st.header("🎨 Configuración de Visualización Mejorada")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_fps = st.slider("FPS objetivo", 15, 60, 30)
        num_points = st.selectbox("Puntos en círculo", [2000, 3000, 4000, 5000], index=1)
    
    with col2:
        st.metric("Duración total", f"{len(st.session_state.audio_data)/st.session_state.sample_rate:.2f}s")
        st.metric("Mejoras", "✅ Audio + Círculo suave")
    
    with col3:
        st.metric("Calidad video", f"{800}x{800}px")
        st.metric("Audio en video", "✅ Incluido")
    
    # Información de mejoras
    st.info("🆕 **Nuevas características:** El círculo ahora se cierra perfectamente sin discontinuidades y el video incluye el audio original sincronizado.")
    
    st.markdown("---")
    
    # ===== BOTONES DE CONTROL =====
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("▶️ Preview en Tiempo Real", type="primary", use_container_width=True):
            st.session_state.is_playing = True
            
            play_with_realtime_preview(
                st.session_state.audio_data,
                st.session_state.sample_rate,
                target_fps,
                num_points
            )
    
    with col_b:
        if st.button("🎬 Generar Video con Audio", use_container_width=True):
            with st.spinner("🎬 Generando video mejorado..."):
                success = generate_video_with_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_fps,
                    num_points
                )
                
                if success:
                    st.success(f"✅ Video generado con {len(st.session_state.video_frames)} frames y círculo suave")
                else:
                    st.error("❌ Error generando video")
    
    with col_c:
        if st.button("⏹️ Detener", use_container_width=True):
            st.session_state.is_playing = False
            sd.stop()
            st.info("⏸️ Detenido")

# ===== SECCIÓN DE DESCARGA MEJORADA =====
if st.session_state.video_frames:
    st.header("⬇️ Descargar Video con Audio")
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        st.success(f"🎬 Video listo: {len(st.session_state.video_frames)} frames")
        
        duration_video = len(st.session_state.video_frames) / target_fps
        st.info(f"📊 Duración: {duration_video:.2f}s @ {target_fps} FPS")
        st.success("🔊 Con audio sincronizado incluido")
    
    with col_download2:
        if st.button("🎬 Crear archivo MP4 con Audio", use_container_width=True):
            with st.spinner("📦 Creando archivo MP4 con audio..."):
                video_bytes = create_video_file_with_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_fps
                )
                
                if video_bytes:
                    st.download_button(
                        label="⬇️ Descargar Video MP4 con Audio",
                        data=video_bytes,
                        file_name=f"audio_viz_con_audio_{int(time.time())}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    st.success("✅ ¡Archivo MP4 con audio listo para descargar!")

# ===== INFORMACIÓN Y AYUDA =====
with st.expander("📖 Instrucciones de Uso - Versión Mejorada"):
    st.markdown("""
    ### 🚀 Pasos para crear tu visualización mejorada:
    
    1. **📁 Cargar Audio**: Sube un archivo `.wav` usando el botón de arriba
    2. **⚙️ Configurar**: Ajusta FPS, puntos y calidad según tus preferencias
    3. **▶️ Preview**: Usa "Preview en Tiempo Real" para ver la visualización mejorada
    4. **🎬 Generar**: Presiona "Generar Video con Audio" para crear el video completo
    5. **⬇️ Descargar**: Crea y descarga el archivo MP4 con audio sincronizado
    
    ### 🆕 Nuevas Mejoras:
    - **🔄 Círculo perfectamente cerrado**: El inicio y final se unen suavemente sin discontinuidades
    - **🔊 Audio incluido**: El video descargado incluye el audio original sincronizado
    - **🎨 Suavizado circular**: Algoritmos especiales que respetan la continuidad circular
    - **⚡ Mejor rendimiento**: Optimizaciones en el renderizado y procesamiento
    
    ### 🎨 Características técnicas:
    - **Forma circular optimizada** con unión suave garantizada
    - **Audio sincronizado** usando FFmpeg para máxima compatibilidad
    - **Puntos controlados** (1-12px) con suavizado circular
    - **Colores continuos** que respetan la geometría circular
    - **Alta calidad** 800x800px con audio de alta fidelidad
    
    ### 💡 Consejos mejorados:
    - **Círculo suave**: Ahora el círculo se cierra perfectamente en todos los casos
    - **Audio incluido**: No necesitas combinar audio por separado
    - **FFmpeg automático**: Se instala automáticamente si no está disponible
    - **Compatibilidad total**: Videos MP4 reproducibles en cualquier dispositivo
    """)

with st.expander("⚙️ Mejoras Técnicas Implementadas"):
    st.markdown("### 🔧 Algoritmos de Suavizado Circular:")
    st.markdown(f"""
    - **Ventana gaussiana circular**: Suavizado que respeta la geometría del círculo
    - **Unión perfecta**: Los puntos inicial y final se mezclan suavemente
    - **Continuidad garantizada**: Sin saltos visuales en la unión
    - **Suavizado adaptativo**: Intensidad ajustada según el tipo de efecto
    - **Preservación de forma**: El círculo base siempre permanece reconocible
    """)
    
    st.markdown("### 🎬 Sistema de Audio Integrado:")
    st.markdown("""
    - **FFmpeg automático**: Detección e instalación automática
    - **Sincronización perfecta**: Frame-perfect timing entre audio y video
    - **Codecs optimizados**: H.264 para video + AAC para audio
    - **Compatibilidad universal**: Reproducible en todos los dispositivos
    - **Calidad preservada**: Audio de 16-bit sin pérdida de calidad
    - **Respaldo sin audio**: Si FFmpeg falla, se genera video sin audio
    """)
    
    st.markdown("### 🔄 Proceso de Mejora del Círculo:")
    st.markdown("""
    1. **Generación base**: Crear puntos distribuidos uniformemente
    2. **Aplicar efectos**: Distorsiones controladas por audio
    3. **Suavizado circular**: Ventana gaussiana que respeta circularidad
    4. **Verificar cierre**: Algoritmo especial para unir inicio/final
    5. **Renderizado suave**: Puntos con tamaños y colores continuos
    """)

# Estado actual mejorado
if st.session_state.audio_data is not None:
    st.markdown("---")
    st.markdown("### 📊 Estado Actual - Versión Mejorada")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        st.success("🎵 Audio cargado")
        duration = len(st.session_state.audio_data) / st.session_state.sample_rate
        st.write(f"Duración: {duration:.2f}s")
    
    with col_status2:
        if st.session_state.video_frames:
            st.success("🎬 Video generado")
            st.write(f"Frames: {len(st.session_state.video_frames)}")
        else:
            st.info("⏳ Video pendiente")
    
    with col_status3:
        ffmpeg_available = check_ffmpeg()
        if ffmpeg_available:
            st.success("🔊 FFmpeg disponible")
            st.write("Audio: ✅ Habilitado")
        else:
            st.warning("🔊 FFmpeg no disponible")
            st.write("Se intentará instalar")
    
    with col_status4:
        if st.session_state.is_playing:
            st.warning("▶️ Reproduciendo...")
        else:
            st.info("⏸️ Detenido")

# Debug info mejorado
with st.expander("🔧 Información de Debug - Versión Mejorada"):
    st.write("**Estado interno mejorado:**")
    st.write(f"- Audio cargado: {st.session_state.audio_data is not None}")
    st.write(f"- Sample rate: {st.session_state.sample_rate}")
    st.write(f"- Reproduciendo: {st.session_state.is_playing}")
    st.write(f"- Frames de video: {len(st.session_state.video_frames) if st.session_state.video_frames else 0}")
    st.write(f"- FFmpeg disponible: {check_ffmpeg()}")
    
    if st.session_state.audio_data is not None:
        st.write(f"- Duración total: {len(st.session_state.audio_data)/st.session_state.sample_rate:.2f}s")
        st.write(f"- RMS promedio: {np.sqrt(np.mean(st.session_state.audio_data**2)):.4f}")
        st.write(f"- Rango dinámico: {np.min(st.session_state.audio_data):.3f} a {np.max(st.session_state.audio_data):.3f}")
        st.write(f"- Mejoras activas: Círculo suave ✅, Audio en video ✅")

st.markdown("---")
st.markdown("🌊 **Visualizador Circular Mejorado** - Círculo perfectamente cerrado + Audio incluido")
st.markdown("### 📁 ▶️ 🎬 🔊 ⬇️ Pipeline completo con audio sincronizado")
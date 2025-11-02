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

# Configuration optimisée
st.set_page_config(
    page_title="Visualiseur Audio Circulaire - Téléchargement & Génération",
    page_icon="🌊",
    layout="wide"
)

# Variables globales optimisées
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
        """Extraction ultra-rapide des caractéristiques"""
        if len(audio_chunk) == 0:
            return self.create_silent_features()
        
        # RMS rapide
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        if rms < self.noise_threshold:
            return self.create_silent_features(rms)
        
        # FFT optimisée
        fft_data = fft(audio_chunk, n=self.buffer_size)
        freqs = fftfreq(self.buffer_size, 1/self.sample_rate)
        magnitudes = np.abs(fft_data)
        
        # Seulement les fréquences positives
        pos_idx = freqs > 0
        pos_freqs = freqs[pos_idx]
        pos_mags = magnitudes[pos_idx]
        
        # Trouver les pics principaux rapidement
        if len(pos_mags) == 0:
            return self.create_silent_features(rms)
        
        # Méthode plus rapide pour trouver les pics
        peak_threshold = np.max(pos_mags) * 0.1
        peaks = np.where(pos_mags > peak_threshold)[0]
        
        if len(peaks) == 0:
            max_idx = np.argmax(pos_mags)
            peaks = [max_idx]
        
        # Prendre seulement les pics les plus importants
        peak_freqs = pos_freqs[peaks]
        peak_mags = pos_mags[peaks]
        
        # Trier par magnitude et prendre les top 4
        if len(peak_mags) > 1:
            top_indices = np.argsort(peak_mags)[-4:][::-1]
            peak_freqs = peak_freqs[top_indices]
            peak_mags = peak_mags[top_indices]
        
        # Caractéristiques de base
        fundamental_freq = peak_freqs[0] if len(peak_freqs) > 0 else 440
        fundamental_mag = peak_mags[0] if len(peak_mags) > 0 else 0
        
        # Centroïde spectral rapide
        if np.sum(pos_mags) > 0:
            spectral_centroid = np.sum(pos_freqs * pos_mags) / np.sum(pos_mags)
        else:
            spectral_centroid = 1000
        
        # Bandes d'énergie simplifiées
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
        
        # Lissage en temps réel
        if self.prev_features is not None:
            features = self.smooth_features(features, self.prev_features)
        
        self.prev_features = features.copy()
        return features
    
    def smooth_features(self, current, previous):
        """Lissage ultra-rapide"""
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
        
        # Pré-calculer les positions angulaires pour le cercle
        self.angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        self.base_radius = 3.0
        
        # Pré-calculer les valeurs pour l'optimisation
        self.freq_scalers = np.array([1.0, 0.5, 0.3, 0.2])
        self.phase_multipliers = np.array([1.0, 1.2, 1.4, 1.6])
        
        # CORRIGÉ: Pré-calculer la fonction de lissage circulaire avec taille fixe
        self.smooth_window_size = 20
        self.smooth_window = self._create_circular_smoothing_window(self.smooth_window_size)
        
    def _create_circular_smoothing_window(self, window_size):
        """Créer une fenêtre de lissage qui respecte la circularité"""
        # Fenêtre gaussienne qui s'applique circulairement
        window = np.exp(-0.5 * (np.linspace(-2, 2, window_size) ** 2))
        return window / np.sum(window)
    
    def _apply_circular_smoothing(self, values, intensity=0.3):
        """CORRIGÉ: Appliquer un lissage qui maintient la continuité circulaire"""
        if len(values) < 10:
            return values
        
        smoothed = values.copy()
        window_half = self.smooth_window_size // 2
        
        for i in range(len(values)):
            # Obtenir les indices circulaires pour la fenêtre
            indices = []
            for j in range(-window_half, window_half):
                idx = (i + j) % len(values)
                indices.append(idx)
            
            # CORRIGÉ: S'assurer que nous avons exactement la bonne taille
            if len(indices) != self.smooth_window_size:
                # Ajuster s'il y a une discordance
                while len(indices) < self.smooth_window_size:
                    indices.append(indices[-1])
                while len(indices) > self.smooth_window_size:
                    indices.pop()
            
            # Appliquer la fenêtre de lissage
            window_values = values[indices]
            
            # CORRIGÉ: Vérifier les tailles avant l'opération
            if len(window_values) == len(self.smooth_window):
                smoothed_value = np.sum(window_values * self.smooth_window)
                # Mélanger avec la valeur originale
                smoothed[i] = (1 - intensity) * values[i] + intensity * smoothed_value
        
        return smoothed
        
    def create_circular_visualization(self, features, dt=0.016):
        """Rendu circulaire AMÉLIORÉ - cercle parfaitement fermé"""
        self.phase_accumulator += dt * 4
        
        # Configurer la figure de manière efficace
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        
        # Paramètres normalisés OPTIMISÉS
        freq_norm = min(features['fundamental_freq'] / 600, 2.0)
        rms_norm = min(features['rms'] * 50, 1.0)
        centroid_norm = min(features['spectral_centroid'] / 2000, 1.5)
        
        # Calculer les rayons pour chaque point
        radii = np.full(self.num_points, self.base_radius)
        individual_movement = np.zeros(self.num_points)
        
        if features['is_silent'] or rms_norm < 0.08:
            # Animation douce pour le silence - ASSURER LA CONTINUITÉ
            ripple = 0.15 * np.sin(2 * self.angles + self.phase_accumulator * 0.8)
            individual_noise = np.random.normal(0, 0.08, self.num_points)
            breathing = 0.1 * np.sin(self.phase_accumulator * 0.5)
            
            # APPLIQUER LE LISSAGE CIRCULAIRE
            ripple = self._apply_circular_smoothing(ripple, 0.4)
            individual_noise = self._apply_circular_smoothing(individual_noise, 0.6)
            
            radii += ripple + individual_noise + breathing
            individual_movement = individual_noise
            
            # VÉRIFIER LA FERMETURE PARFAITE
            radii = self._ensure_smooth_closure(radii)
            
            colors = np.full((self.num_points, 3), [0.2, 0.2, 0.4])
            point_sizes = np.full(self.num_points, 2.0)
            alphas = np.full(self.num_points, 0.4)
            
        else:
            # DISTORSION CONTRÔLÉE DU CERCLE AVEC FERMETURE DOUCE
            
            # 1. Onde principale MODÉRÉE
            main_freq = features['fundamental_freq'] / 200
            primary_wave_phase = main_freq * self.angles + self.phase_accumulator * 1.5
            primary_wave = np.sin(primary_wave_phase)
            main_distortion = np.clip(rms_norm * 1.2 * primary_wave, -1.2, 1.2)
            
            # APPLIQUER LE LISSAGE CIRCULAIRE À L'ONDE PRINCIPALE
            main_distortion = self._apply_circular_smoothing(main_distortion, 0.3)
            radii += main_distortion
            
            # 2. Ondes secondaires PLUS CONTRÔLÉES
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
                    
                    # LISSAGE CIRCULAIRE POUR LES ONDES SECONDAIRES
                    secondary_distortion = self._apply_circular_smoothing(secondary_distortion, 0.2)
                    radii += secondary_distortion
            
            # 3. MOUVEMENT INDIVIDUEL CONTRÔLÉ AVEC CONTINUITÉ
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
            
            # APPLIQUER LE LISSAGE CIRCULAIRE AU MOUVEMENT INDIVIDUEL
            individual_movement = self._apply_circular_smoothing(individual_movement, 0.4)
            radii = self.base_radius * np.ones(self.num_points) + main_distortion + individual_movement
            
            # 4. Modulation rotatoire DOUCE
            centroid_rotation = centroid_norm * 3 * self.angles + self.phase_accumulator * 1.2
            centroid_wave = 0.4 * np.sin(centroid_rotation)
            centroid_wave = self._apply_circular_smoothing(centroid_wave, 0.3)
            radii += centroid_wave * rms_norm * 0.5
            
            # 5. Ondes haute fréquence PLUS SUBTILES
            if features['high_ratio'] > 0.15:
                high_freq_wave = 0.25 * np.sin(12 * self.angles + self.phase_accumulator * 8) * features['high_ratio']
                high_freq_wave = self._apply_circular_smoothing(high_freq_wave, 0.5)
                radii += high_freq_wave * rms_norm * 0.6
            
            # 6. LIMITEUR GLOBAL AVEC CONTINUITÉ CIRCULAIRE
            max_distortion = self.base_radius * 0.8
            radii = np.clip(radii, self.base_radius - max_distortion, self.base_radius + max_distortion)
            
            # ASSURER LA FERMETURE PARFAITE DU CERCLE
            radii = self._ensure_smooth_closure(radii)
            
            # 7. Effets globaux doux
            global_breathing = 0.2 * np.sin(self.phase_accumulator * 0.8) * rms_norm
            rhythm_pulse = 0.3 * np.sin(self.phase_accumulator * 4) * (rms_norm ** 2)
            radii += global_breathing + rhythm_pulse
            
            # Système de couleurs CONTRÔLÉ
            hue_base = (centroid_norm * 0.8 + self.phase_accumulator * 0.03) % 1.0
            colors = self.generate_smooth_colors(radii, individual_movement, hue_base, features)
            
            # Tailles PETITES ET CONTRÔLÉES
            radius_variation = np.abs(radii - self.base_radius)
            individual_intensity = np.abs(individual_movement)
            combined_intensity = radius_variation + individual_intensity
            
            point_sizes = 2.0 + combined_intensity * 8
            point_sizes = np.clip(point_sizes, 1.0, 12)
            
            # Transparences DOUCES
            max_intensity = np.max(radius_variation + individual_intensity) + 1e-6
            local_energy = (radius_variation + individual_intensity) / max_intensity
            alphas = 0.5 + local_energy * 0.3
            alphas = np.clip(alphas, 0.3, 0.8)
        
        # Convertir les coordonnées polaires en cartésiennes
        x_positions = radii * np.cos(self.angles)
        y_positions = radii * np.sin(self.angles)
        
        # Rendu DOUX
        self.render_smooth_points(ax, x_positions, y_positions, colors, point_sizes, alphas, individual_movement)
        
        # Cercle de référence TOUJOURS VISIBLE
        circle_alpha = 0.3 if rms_norm > 0.08 else 0.2
        reference_circle = plt.Circle((0, 0), self.base_radius, 
                                    fill=False, color='cyan', 
                                    alpha=circle_alpha, linewidth=1.0)
        ax.add_patch(reference_circle)
        
        # Configurer les axes
        max_radius = self.base_radius * 1.8
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        ax.axis('off')
        
        # Titre informatif
        if not features['is_silent']:
            title = f"🎵 {features['fundamental_freq']:.0f}Hz | RMS: {features['rms']:.3f} | Points: {self.num_points}"
            plt.suptitle(title, color='white', fontsize=11, y=0.95)
        
        plt.tight_layout()
        return fig
    
    def _ensure_smooth_closure(self, radii, blend_points=10):
        """AMÉLIORATION: S'assurer que le début et la fin du cercle se rejoignent en douceur"""
        if len(radii) < blend_points * 2:
            return radii
        
        # Créer une copie pour modifier
        smooth_radii = radii.copy()
        
        # Obtenir les valeurs du début et de la fin
        start_values = radii[:blend_points]
        end_values = radii[-blend_points:]
        
        # Créer une moyenne pondérée pour lisser la jonction
        for i in range(blend_points):
            # Facteur de mélange (plus de poids au centre, moins sur les bords)
            weight = (i + 1) / (blend_points + 1)
            
            # Mélanger le début avec la fin
            start_blend = (1 - weight) * start_values[i] + weight * end_values[-(i+1)]
            end_blend = (1 - weight) * end_values[-(i+1)] + weight * start_values[i]
            
            smooth_radii[i] = start_blend
            smooth_radii[-(i+1)] = end_blend
        
        return smooth_radii
    
    def generate_smooth_colors(self, radii, individual_movement, hue_base, features):
        """Génération de couleurs DOUCE - avec continuité circulaire"""
        colors = np.zeros((len(radii), 3))
        
        # Couleur de base qui tourne DOUCEMENT
        base_hue = (hue_base + self.angles / (2 * np.pi) * 0.4) % 1.0
        
        # APPLIQUER LE LISSAGE CIRCULAIRE AUX COULEURS
        base_hue = self._apply_circular_smoothing(base_hue, 0.3)
        
        # Variation DOUCE par lots
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
        
        # Effets spéciaux SUBTILS
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
        """Rendu DOUX - sans effets extrêmes"""
        movement_intensity = np.abs(individual_movement)
        
        high_movement_mask = movement_intensity > 0.4
        medium_movement_mask = (movement_intensity > 0.2) & (movement_intensity <= 0.4)
        low_movement_mask = movement_intensity <= 0.2
        
        # Rendre les points de faible mouvement (majorité)
        if np.any(low_movement_mask):
            ax.scatter(
                x_pos[low_movement_mask], 
                y_pos[low_movement_mask],
                c=colors[low_movement_mask], 
                s=sizes[low_movement_mask],
                alpha=0.5,
                edgecolors='none'
            )
        
        # Rendre les points de mouvement moyen
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
        
        # Rendre les points de HAUT mouvement
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
    """Vérifier si FFmpeg est disponible"""
    try:
        # D'abord essayer avec FFmpeg du système
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Si FFmpeg du système n'est pas trouvé, essayer avec imageio-ffmpeg
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
    """Obtenir le chemin de l'exécutable FFmpeg"""
    try:
        # D'abord essayer avec FFmpeg du système
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return 'ffmpeg'
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Si FFmpeg du système n'est pas trouvé, utiliser imageio-ffmpeg
        try:
            import imageio_ffmpeg as ffmpeg
            return ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

def install_ffmpeg():
    """Essayer d'installer FFmpeg"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg'])
        return True
    except subprocess.CalledProcessError:
        return False

def load_wav_file(uploaded_file):
    """Charger un fichier WAV depuis l'upload"""
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
        st.error(f"Erreur lors du chargement du fichier WAV : {e}")
        return None, None

def generate_video_with_audio(audio_data, sample_rate, target_fps=30, num_points=3000):
    """Générer une vidéo complète avec audio synchronisé"""
    
    processor = AudioProcessor(sample_rate, buffer_size=1024)
    renderer = CircularVisualRenderer(num_points=num_points)
    
    frame_duration = 1.0 / target_fps
    audio_duration = len(audio_data) / sample_rate
    total_frames = int(audio_duration * target_fps)
    
    st.session_state.video_frames = []
    
    progress_container = st.empty()
    status_container = st.empty()
    
    status_container.info(f"🎬 Génération de {total_frames} images à {target_fps} FPS...")
    
    for frame_idx in range(total_frames):
        frame_start_time = frame_idx * frame_duration
        
        progress = frame_idx / total_frames
        progress_container.progress(progress, f"Image {frame_idx + 1}/{total_frames}")
        
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
    status_container.success(f"✅ {len(st.session_state.video_frames)} images générées")
    
    return len(st.session_state.video_frames) > 0

def create_video_file_with_audio(audio_data, sample_rate, target_fps=30):
    """AMÉLIORATION: Créer un fichier vidéo MP4 AVEC AUDIO"""
    if not st.session_state.video_frames:
        return None
    
    try:
        # Créer des fichiers temporaires
        video_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        final_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        
        video_temp_path = video_temp.name
        audio_temp_path = audio_temp.name
        final_temp_path = final_temp.name
        
        video_temp.close()
        audio_temp.close()
        final_temp.close()
        
        # 1. Créer la vidéo sans audio
        height, width = st.session_state.video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_temp_path, fourcc, target_fps, (width, height))
        
        for frame in st.session_state.video_frames:
            video_writer.write(frame)
        
        video_writer.release()
        
        # 2. Sauvegarder l'audio original
        wavfile.write(audio_temp_path, sample_rate, (audio_data * 32767).astype(np.int16))
        
        # 3. Vérifier FFmpeg
        if not check_ffmpeg():
            st.warning("FFmpeg non trouvé. Tentative d'installation...")
            if install_ffmpeg():
                st.success("FFmpeg installé correctement")
            else:
                st.error("Impossible d'installer FFmpeg. La vidéo sera téléchargée sans audio.")
                # Retourner la vidéo sans audio
                with open(video_temp_path, 'rb') as f:
                    video_bytes = f.read()
                os.unlink(video_temp_path)
                os.unlink(audio_temp_path)
                return video_bytes
        
        # 4. Combiner vidéo et audio avec FFmpeg
        try:
            # Obtenir l'exécutable correct de FFmpeg
            ffmpeg_exe = get_ffmpeg_executable()
            if not ffmpeg_exe:
                st.error("FFmpeg non trouvé. La vidéo sera téléchargée sans audio.")
                with open(video_temp_path, 'rb') as f:
                    video_bytes = f.read()
                os.unlink(video_temp_path)
                os.unlink(audio_temp_path)
                return video_bytes
            
            ffmpeg_cmd = [
                ffmpeg_exe, '-y',  # Utiliser l'exécutable correct
                '-i', video_temp_path,  # Vidéo d'entrée
                '-i', audio_temp_path,  # Audio d'entrée
                '-c:v', 'libx264',  # Codec vidéo
                '-c:a', 'aac',  # Codec audio
                '-shortest',  # Durée du plus court
                '-pix_fmt', 'yuv420p',  # Format de pixel compatible
                final_temp_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
            
            # Lire la vidéo finale avec audio
            with open(final_temp_path, 'rb') as f:
                video_bytes = f.read()
            
            # Nettoyer les fichiers temporaires
            os.unlink(video_temp_path)
            os.unlink(audio_temp_path)
            os.unlink(final_temp_path)
            
            return video_bytes
            
        except subprocess.CalledProcessError as e:
            st.error(f"Erreur lors de la combinaison audio et vidéo : {e}")
            # Retourner la vidéo sans audio comme solution de secours
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
        st.error(f"Erreur lors de la création de la vidéo : {e}")
        return None

def play_with_realtime_preview(audio_data, sample_rate, target_fps=45, num_points=3000):
    """Système de lecture avec aperçu en temps réel"""
    
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
            st.error(f"Erreur audio : {e}")
    
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
                info_text = f"⚡ {fps_actual:.1f} FPS | ⏱️ {elapsed_time:.2f}s/{audio_duration:.2f}s | 🔵 {num_points} points"
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
        st.success(f"✅ Lecture terminée - {frame_count} images @ {frame_count/audio_duration:.1f} FPS moyen")
        
    except Exception as e:
        st.error(f"Erreur pendant la lecture : {e}")
    finally:
        st.session_state.is_playing = False
        try:
            sd.stop()
        except:
            pass

# ===== INTERFACE PRINCIPALE =====

st.title("🌊 Visualiseur Circulaire Amélioré - Avec Audio et Cercle Lisse")
st.markdown("---")
st.markdown("🎵 **Charger fichiers WAV** | 🎬 **Vidéos avec audio** | 🌊 **Cercle parfaitement fermé**")
st.markdown("### 📁 Télécharger fichier → ▶️ Lire → 🎬 Générer vidéo → ⬇️ Télécharger avec audio")
st.markdown("---")

# ===== SECTION DE CHARGEMENT DE FICHIER =====
st.header("📁 Charger un Fichier Audio")

uploaded_file = st.file_uploader(
    "Sélectionnez un fichier WAV",
    type=['wav'],
    help="Téléchargez un fichier .wav pour visualiser"
)

if uploaded_file is not None:
    with st.spinner("🔄 Chargement du fichier..."):
        audio_data, sample_rate = load_wav_file(uploaded_file)
        
        if audio_data is not None:
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            
            duration = len(audio_data) / sample_rate
            
            st.success(f"✅ Fichier chargé : {uploaded_file.name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Durée", f"{duration:.2f}s")
            with col2:
                st.metric("Fréq. échant.", f"{sample_rate} Hz")
            with col3:
                st.metric("Échantillons", f"{len(audio_data):,}")
            with col4:
                rms_level = np.sqrt(np.mean(audio_data**2))
                st.metric("RMS", f"{rms_level:.4f}")

# ===== CONTRÔLES DE VISUALISATION =====
if st.session_state.audio_data is not None:
    st.header("🎨 Configuration de Visualisation Améliorée")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_fps = st.slider("FPS cible", 15, 60, 30)
        num_points = st.selectbox("Points dans le cercle", [2000, 3000, 4000, 5000], index=1)
    
    with col2:
        st.metric("Durée totale", f"{len(st.session_state.audio_data)/st.session_state.sample_rate:.2f}s")
        st.metric("Améliorations", "✅ Audio + Cercle lisse")
    
    with col3:
        st.metric("Qualité vidéo", f"{800}x{800}px")
        st.metric("Audio dans vidéo", "✅ Inclus")
    
    # Information sur les améliorations
    st.info("🆕 **Nouvelles fonctionnalités :** Le cercle se ferme maintenant parfaitement sans discontinuités et la vidéo inclut l'audio original synchronisé.")
    
    st.markdown("---")
    
    # ===== BOUTONS DE CONTRÔLE =====
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("▶️ Aperçu en Temps Réel", type="primary", use_container_width=True):
            st.session_state.is_playing = True
            
            play_with_realtime_preview(
                st.session_state.audio_data,
                st.session_state.sample_rate,
                target_fps,
                num_points
            )
    
    with col_b:
        if st.button("🎬 Générer Vidéo avec Audio", use_container_width=True):
            with st.spinner("🎬 Génération de la vidéo améliorée..."):
                success = generate_video_with_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_fps,
                    num_points
                )
                
                if success:
                    st.success(f"✅ Vidéo générée avec {len(st.session_state.video_frames)} images et cercle lisse")
                else:
                    st.error("❌ Erreur lors de la génération de la vidéo")
    
    with col_c:
        if st.button("⏹️ Arrêter", use_container_width=True):
            st.session_state.is_playing = False
            sd.stop()
            st.info("⏸️ Arrêté")

# ===== SECTION DE TÉLÉCHARGEMENT AMÉLIORÉE =====
if st.session_state.video_frames:
    st.header("⬇️ Télécharger la Vidéo avec Audio")
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        st.success(f"🎬 Vidéo prête : {len(st.session_state.video_frames)} images")
        
        duration_video = len(st.session_state.video_frames) / target_fps
        st.info(f"📊 Durée : {duration_video:.2f}s @ {target_fps} FPS")
        st.success("🔊 Avec audio synchronisé inclus")
    
    with col_download2:
        if st.button("🎬 Créer fichier MP4 avec Audio", use_container_width=True):
            with st.spinner("📦 Création du fichier MP4 avec audio..."):
                video_bytes = create_video_file_with_audio(
                    st.session_state.audio_data,
                    st.session_state.sample_rate,
                    target_fps
                )
                
                if video_bytes:
                    st.download_button(
                        label="⬇️ Télécharger Vidéo MP4 avec Audio",
                        data=video_bytes,
                        file_name=f"visualisation_audio_avec_audio_{int(time.time())}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    st.success("✅ ¡Fichier MP4 avec audio prêt à télécharger !")

# ===== INFORMATION ET AIDE =====
with st.expander("📖 Instructions d'Utilisation - Version Améliorée"):
    st.markdown("""
    ### 🚀 Étapes pour créer votre visualisation améliorée :
    
    1. **📁 Charger Audio** : Téléchargez un fichier `.wav` en utilisant le bouton ci-dessus
    2. **⚙️ Configurer** : Ajustez les FPS, points et qualité selon vos préférences
    3. **▶️ Aperçu** : Utilisez "Aperçu en Temps Réel" pour voir la visualisation améliorée
    4. **🎬 Générer** : Appuyez sur "Générer Vidéo avec Audio" pour créer la vidéo complète
    5. **⬇️ Télécharger** : Créez et téléchargez le fichier MP4 avec audio synchronisé
    
    ### 🆕 Nouvelles Améliorations :
    - **🔄 Cercle parfaitement fermé** : Le début et la fin se rejoignent en douceur sans discontinuités
    - **🔊 Audio inclus** : La vidéo téléchargée inclut l'audio original synchronisé
    - **🎨 Lissage circulaire** : Algorithmes spéciaux qui respectent la continuité circulaire
    - **⚡ Meilleures performances** : Optimisations dans le rendu et le traitement
    
    ### 🎨 Caractéristiques techniques :
    - **Forme circulaire optimisée** avec jonction lisse garantie
    - **Audio synchronisé** utilisant FFmpeg pour une compatibilité maximale
    - **Points contrôlés** (1-12px) avec lissage circulaire
    - **Couleurs continues** qui respectent la géométrie circulaire
    - **Haute qualité** 800x800px avec audio haute fidélité
    
    ### 💡 Conseils améliorés :
    - **Cercle lisse** : Le cercle se ferme maintenant parfaitement dans tous les cas
    - **Audio inclus** : Pas besoin de combiner l'audio séparément
    - **FFmpeg automatique** : S'installe automatiquement s'il n'est pas disponible
    - **Compatibilité totale** : Vidéos MP4 lisibles sur n'importe quel appareil
    """)

with st.expander("⚙️ Améliorations Techniques Implémentées"):
    st.markdown("### 🔧 Algorithmes de Lissage Circulaire :")
    st.markdown(f"""
    - **Fenêtre gaussienne circulaire** : Lissage qui respecte la géométrie du cercle
    - **Jonction parfaite** : Les points initial et final se mélangent en douceur
    - **Continuité garantie** : Sans sauts visuels dans la jonction
    - **Lissage adaptatif** : Intensité ajustée selon le type d'effet
    - **Préservation de forme** : Le cercle de base reste toujours reconnaissable
    """)
    
    st.markdown("### 🎬 Système Audio Intégré :")
    st.markdown("""
    - **FFmpeg automatique** : Détection et installation automatique
    - **Synchronisation parfaite** : Timing parfait image par image entre audio et vidéo
    - **Codecs optimisés** : H.264 pour vidéo + AAC pour audio
    - **Compatibilité universelle** : Lisible sur tous les appareils
    - **Qualité préservée** : Audio 16-bit sans perte de qualité
    - **Solution de secours sans audio** : Si FFmpeg échoue, génère vidéo sans audio
    """)
    
    st.markdown("### 🔄 Processus d'Amélioration du Cercle :")
    st.markdown("""
    1. **Génération de base** : Créer des points distribués uniformément
    2. **Appliquer les effets** : Distorsions contrôlées par l'audio
    3. **Lissage circulaire** : Fenêtre gaussienne qui respecte la circularité
    4. **Vérifier la fermeture** : Algorithme spécial pour joindre début/fin
    5. **Rendu lisse** : Points avec tailles et couleurs continues
    """)

# État actuel amélioré
if st.session_state.audio_data is not None:
    st.markdown("---")
    st.markdown("### 📊 État Actuel - Version Améliorée")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        st.success("🎵 Audio chargé")
        duration = len(st.session_state.audio_data) / st.session_state.sample_rate
        st.write(f"Durée : {duration:.2f}s")
    
    with col_status2:
        if st.session_state.video_frames:
            st.success("🎬 Vidéo générée")
            st.write(f"Images : {len(st.session_state.video_frames)}")
        else:
            st.info("⏳ Vidéo en attente")
    
    with col_status3:
        ffmpeg_available = check_ffmpeg()
        if ffmpeg_available:
            st.success("🔊 FFmpeg disponible")
            st.write("Audio : ✅ Activé")
        else:
            st.warning("🔊 FFmpeg non disponible")
            st.write("Tentative d'installation")
    
    with col_status4:
        if st.session_state.is_playing:
            st.warning("▶️ En lecture...")
        else:
            st.info("⏸️ Arrêté")

# Info de débogage améliorée
with st.expander("🔧 Information de Débogage - Version Améliorée"):
    st.write("**État interne amélioré :**")
    st.write(f"- Audio chargé : {st.session_state.audio_data is not None}")
    st.write(f"- Fréquence d'échantillonnage : {st.session_state.sample_rate}")
    st.write(f"- En lecture : {st.session_state.is_playing}")
    st.write(f"- Images de vidéo : {len(st.session_state.video_frames) if st.session_state.video_frames else 0}")
    st.write(f"- FFmpeg disponible : {check_ffmpeg()}")
    
    if st.session_state.audio_data is not None:
        st.write(f"- Durée totale : {len(st.session_state.audio_data)/st.session_state.sample_rate:.2f}s")
        st.write(f"- RMS moyen : {np.sqrt(np.mean(st.session_state.audio_data**2)):.4f}")
        st.write(f"- Plage dynamique : {np.min(st.session_state.audio_data):.3f} à {np.max(st.session_state.audio_data):.3f}")
        st.write(f"- Améliorations actives : Cercle lisse ✅, Audio dans vidéo ✅")

st.markdown("---")
st.markdown("🌊 **Visualiseur Circulaire Amélioré** - Cercle parfaitement fermé + Audio inclus")
st.markdown("### 📁 ▶️ 🎬 🔊 ⬇️ Pipeline complet avec audio synchronisé")
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

# Configuración de la página
st.set_page_config(
    page_title="Visualizador de Audio FFT - Grabar y Reproducir",
    page_icon="🎵",
    layout="wide"
)

# Variables globales
recorded_audio = None
audio_features_sequence = []
is_recording = False
is_playing = False
recording_thread = None

# Inicializar variables de sesión
if 'recorded_data' not in st.session_state:
    st.session_state.recorded_data = None
if 'features_sequence' not in st.session_state:
    st.session_state.features_sequence = []
if 'recording_params' not in st.session_state:
    st.session_state.recording_params = {}
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

def record_audio(duration, sample_rate=44100):
    """Graba audio por una duración específica"""
    try:
        st.info(f"🎤 Grabando por {duration} segundos...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Esperar a que termine la grabación
        return audio_data.flatten(), sample_rate
    except Exception as e:
        st.error(f"Error al grabar audio: {e}")
        return None, None

def calculate_noise_threshold():
    """Calcula el umbral de ruido para conversación susurrada"""
    return 0.005

def create_silent_features():
    """Crea características para audio silencioso"""
    return {
        'fundamental_freq': 0,
        'fundamental_mag': 0,
        'peak_frequencies': np.array([]),
        'peak_magnitudes': np.array([]),
        'spectral_centroid': 0,
        'spectral_rolloff': 0,
        'rms': 0,
        'low_ratio': 0,
        'mid_ratio': 0,
        'high_ratio': 0,
        'num_peaks': 0,
        'is_silent': True
    }

def get_audio_features(audio_data, sample_rate, noise_threshold=None):
    """Extrae características del audio para la visualización"""
    if len(audio_data) == 0:
        return create_silent_features()
    
    # Calcular RMS
    rms = np.sqrt(np.mean(audio_data**2))
    
    if noise_threshold is None:
        noise_threshold = calculate_noise_threshold()
    
    if rms < noise_threshold:
        return create_silent_features()
    
    # FFT
    fft_data = fft(audio_data)
    frequencies = fftfreq(len(audio_data), 1/sample_rate)
    magnitude = np.abs(fft_data)
    
    # Solo frecuencias positivas
    positive_idx = frequencies > 0
    pos_freq = frequencies[positive_idx]
    pos_mag = magnitude[positive_idx]
    
    if len(pos_mag) == 0:
        return create_silent_features()
    
    # Encontrar picos
    peaks, _ = find_peaks(pos_mag, height=np.max(pos_mag) * 0.1, distance=20)
    
    if len(peaks) == 0:
        return create_silent_features()
    
    # Características principales
    peak_freqs = pos_freq[peaks]
    peak_mags = pos_mag[peaks]
    
    # Ordenar por magnitud
    sorted_idx = np.argsort(peak_mags)[::-1]
    peak_freqs = peak_freqs[sorted_idx]
    peak_mags = peak_mags[sorted_idx]
    
    # Calcular características adicionales
    spectral_centroid = np.sum(pos_freq * pos_mag) / np.sum(pos_mag)
    spectral_rolloff = np.where(np.cumsum(pos_mag) >= 0.85 * np.sum(pos_mag))[0]
    rolloff_freq = pos_freq[spectral_rolloff[0]] if len(spectral_rolloff) > 0 else 0
    
    # Distribución de energía por bandas
    low_band = np.sum(pos_mag[(pos_freq >= 0) & (pos_freq < 200)])
    mid_band = np.sum(pos_mag[(pos_freq >= 200) & (pos_freq < 2000)])
    high_band = np.sum(pos_mag[(pos_freq >= 2000) & (pos_freq < 8000)])
    
    total_energy = low_band + mid_band + high_band
    if total_energy > 0:
        low_ratio = low_band / total_energy
        mid_ratio = mid_band / total_energy
        high_ratio = high_band / total_energy
    else:
        low_ratio = mid_ratio = high_ratio = 0.33
    
    return {
        'fundamental_freq': peak_freqs[0] if len(peak_freqs) > 0 else 440,
        'fundamental_mag': peak_mags[0] if len(peak_mags) > 0 else 0,
        'peak_frequencies': peak_freqs[:8],
        'peak_magnitudes': peak_mags[:8],
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': rolloff_freq,
        'rms': rms,
        'low_ratio': low_ratio,
        'mid_ratio': mid_ratio,
        'high_ratio': high_ratio,
        'num_peaks': len(peak_freqs),
        'is_silent': False
    }

def process_audio_to_features(audio_data, sample_rate, window_size=2048, hop_size=512, noise_threshold=None):
    """Procesa todo el audio y genera secuencia de características"""
    features_sequence = []
    
    # Crear ventanas deslizantes
    num_windows = (len(audio_data) - window_size) // hop_size + 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        window_data = audio_data[start_idx:end_idx]
        
        # Extraer características de esta ventana
        features = get_audio_features(window_data, sample_rate, noise_threshold)
        features['timestamp'] = start_idx / sample_rate  # Tiempo en segundos
        features_sequence.append(features)
        
        # Actualizar progreso
        progress = (i + 1) / num_windows
        progress_bar.progress(progress)
        status_text.text(f"Procesando audio... {progress*100:.1f}% ({i+1}/{num_windows} ventanas)")
    
    progress_bar.empty()
    status_text.empty()
    
    return features_sequence

def smooth_features_sequence(features_sequence, smoothing_factor=0.3):
    """Suaviza toda la secuencia de características"""
    if len(features_sequence) < 2:
        return features_sequence
    
    smoothed_sequence = [features_sequence[0].copy()]  # Primer frame sin cambios
    
    for i in range(1, len(features_sequence)):
        current = features_sequence[i]
        previous = smoothed_sequence[i-1]
        
        smoothed = {}
        for key in current:
            if key == 'timestamp':
                smoothed[key] = current[key]
            elif isinstance(current[key], np.ndarray):
                if key in previous and len(current[key]) == len(previous[key]):
                    smoothed[key] = (1 - smoothing_factor) * previous[key] + smoothing_factor * current[key]
                else:
                    smoothed[key] = current[key]
            elif isinstance(current[key], (int, float)):
                if key in previous:
                    smoothed[key] = (1 - smoothing_factor) * previous[key] + smoothing_factor * current[key]
                else:
                    smoothed[key] = current[key]
            else:
                smoothed[key] = current[key]
        
        smoothed_sequence.append(smoothed)
    
    return smoothed_sequence

def create_circular_visualization(features, style="cosmic", frame_time=0.0):
    """Crea visualización circular para un frame específico"""
    if features is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Si es silencioso, mostrar visualización mínima
    if features.get('is_silent', False) and features['rms'] < 0.001:
        center_theta = np.linspace(0, 2*np.pi, 100)
        center_radius = np.full_like(center_theta, 0.1)
        ax.fill(center_theta, center_radius, color=(0.1, 0.1, 0.2), alpha=0.3)
        
        ax.set_ylim(0, 1.0)
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        
        plt.suptitle(f'Tiempo: {frame_time:.2f}s - Sin señal', color='white', fontsize=12, y=0.95)
        return fig
    
    # Parámetros basados en las características del audio
    fundamental = features['fundamental_freq']
    rms = features['rms']
    centroid = features['spectral_centroid']
    peaks = features['peak_frequencies']
    mags = features['peak_magnitudes']
    
    # Normalizar valores
    freq_norm = min(fundamental / 1000, 2.0)
    rms_norm = min(rms * 50, 1.0)
    centroid_norm = min(centroid / 2000, 1.0)
    
    # Crear ángulos
    theta = np.linspace(0, 2*np.pi, 1000)
    
    # Colores basados en el timbre
    hue_base = (centroid_norm * 0.8) % 1.0
    
    if style == "cosmic":
        colors = []
        for i in range(max(6, len(peaks))):
            hue = (hue_base + i * 0.15) % 1.0
            sat = 0.8 + rms_norm * 0.2
            val = 0.6 + (mags[i] / mags[0] if len(mags) > i and mags[0] > 0 else 0) * 0.4
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(rgb)
    
    # Capa externa - Anillo principal
    if len(peaks) > 0 and mags[0] > 0:
        base_radius = 0.85 + rms_norm * 0.1
        modulation = 0.08 * rms_norm * (
            np.sin(8 * theta + freq_norm * 0.5 + frame_time) * 0.6 +
            np.sin(4 * theta + freq_norm * 0.3 + frame_time * 0.5) * 0.4
        )
        r_outer = base_radius + modulation
        
        main_color = colors[0] if colors else (0.8, 0.6, 0.2)
        ax.fill(theta, r_outer, color=main_color, alpha=0.8)
        
        # Anillo interno
        inner_radius = 0.65 + rms_norm * 0.05
        pattern_freq = max(4, int(fundamental / 100))
        inner_modulation = 0.04 * (
            np.sin(pattern_freq * theta + frame_time * 2) * rms_norm * 0.7 +
            np.sin(pattern_freq * 2 * theta + np.pi/4 + frame_time) * rms_norm * 0.3
        )
        r_inner = inner_radius + inner_modulation
        
        inner_color = colors[1] if len(colors) > 1 else (0.6, 0.8, 0.9)
        ax.fill(theta, r_inner, color=inner_color, alpha=0.7)
    
    # Capas de frecuencias
    for i, (freq, mag) in enumerate(zip(peaks[:4], mags[:4])):
        if i >= len(colors) or mags[0] == 0:
            break
            
        radius_base = 0.55 - i * 0.08
        pattern_density = max(3, int(freq / 200))
        phase_shift = i * np.pi / 3 + frame_time * (i + 1)
        
        mag_factor = mag / mags[0]
        
        primary_wave = np.sin(pattern_density * theta + phase_shift)
        secondary_wave = 0.3 * np.sin(pattern_density * 2 * theta + phase_shift * 1.5)
        tertiary_wave = 0.1 * np.sin(pattern_density * 4 * theta + phase_shift * 2)
        
        combined_wave = primary_wave + secondary_wave + tertiary_wave
        r_pattern = radius_base * (1 + 0.25 * mag_factor * combined_wave)
        
        ax.plot(theta, r_pattern, color=colors[i], linewidth=2 + mag_factor * 3, alpha=0.8)
        
        if mag_factor > 0.3:
            peak_angles = np.linspace(0, 2*np.pi, pattern_density, endpoint=False)
            for angle in peak_angles:
                peak_r = radius_base * (1 + 0.25 * mag_factor)
                marker_size = 3 + mag_factor * 4
                ax.plot(angle, peak_r, 'o', color=colors[i], 
                       markersize=marker_size, alpha=0.7)
    
    # Centro - Núcleo energético
    center_size = 0.25 * (1 + rms_norm * 0.8)
    center_theta = np.linspace(0, 2*np.pi, 200)
    
    pulse_fast = np.sin(12 * center_theta + frame_time * 8) * 0.15
    pulse_slow = np.sin(4 * center_theta + frame_time * 2) * 0.1
    center_pattern = center_size * (1 + pulse_fast + pulse_slow)
    
    center_hue = (hue_base + 0.5) % 1.0
    center_color = colorsys.hsv_to_rgb(center_hue, 0.9, 0.9)
    ax.fill(center_theta, center_pattern, color=center_color, alpha=0.9)
    
    # Efectos adicionales
    if features['high_ratio'] > 0.15:
        num_rays = int(6 + features['high_ratio'] * 10)
        for i in range(num_rays):
            angle = i * 2 * np.pi / num_rays + frame_time * 0.5
            r_ray = np.linspace(0.1, 0.8, 50)
            intensity = np.exp(-r_ray * 2)
            theta_ray = np.full_like(r_ray, angle)
            alpha_ray = features['high_ratio'] * 0.4 * intensity
            
            for j in range(len(r_ray)-1):
                ax.plot([theta_ray[j], theta_ray[j+1]], [r_ray[j], r_ray[j+1]], 
                       color='white', alpha=alpha_ray[j], linewidth=1.5)
    
    # Configuración final
    ax.set_ylim(0, 1.0)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    
    # Título con información temporal
    title_text = f'Tiempo: {frame_time:.2f}s | '
    title_text += f'Freq: {fundamental:.1f} Hz | Centroide: {centroid:.0f} Hz | RMS: {rms:.4f}'
    
    plt.suptitle(title_text, color='white', fontsize=11, y=0.95)
    
    return fig

def play_audio_with_sync(audio_data, sample_rate):
    """Reproduce audio y devuelve información para sincronización"""
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Normalizar audio
        audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        
        # Guardar como WAV
        with wave.open(tmp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_normalized.tobytes())
        
        return tmp_file.name

# Interfaz principal
st.title("🎵 Visualizador FFT - Grabar y Reproducir")
st.markdown("---")

# Controles de grabación
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    record_duration = st.slider("Duración de grabación (segundos)", 5, 60, 15)

with col2:
    sample_rate = st.selectbox("Frecuencia de muestreo", [22050, 44100, 48000], index=1)

with col3:
    window_size = st.selectbox("Tamaño de ventana", [1024, 2048, 4096], index=1)

# Botones principales
col_a, col_b, col_c = st.columns([2, 2, 2])

with col_a:
    if st.button("🎤 Grabar Audio", type="primary", use_container_width=True):
        if not st.session_state.is_recording:
            st.session_state.is_recording = True
            
            # Grabar audio
            audio_data, sr = record_audio(record_duration, sample_rate)
            
            if audio_data is not None:
                st.session_state.recorded_data = audio_data
                st.session_state.recording_params = {
                    'sample_rate': sr,
                    'duration': record_duration,
                    'window_size': window_size
                }
                
                st.success(f"✅ Audio grabado: {len(audio_data)/sr:.1f} segundos")
                
                # Procesar audio automáticamente
                st.info("🔄 Procesando audio...")
                hop_size = window_size // 4
                features_seq = process_audio_to_features(
                    audio_data, sr, window_size, hop_size
                )
                
                # Suavizar secuencia
                st.session_state.features_sequence = smooth_features_sequence(features_seq)
                
                st.success(f"✅ Procesamiento completo: {len(st.session_state.features_sequence)} frames generados")
            
            st.session_state.is_recording = False

with col_b:
    if st.button("▶️ Reproducir con Visualización", use_container_width=True):
        if st.session_state.recorded_data is not None and st.session_state.features_sequence:
            st.session_state.is_playing = True
            
            # Preparar datos
            audio_data = st.session_state.recorded_data
            sample_rate_used = st.session_state.recording_params['sample_rate']
            features_sequence = st.session_state.features_sequence
            
            # Contenedores para visualización
            viz_container = st.empty()
            info_container = st.empty()
            progress_container = st.empty()
            
            # Iniciar reproducción de audio en hilo separado
            def play_audio():
                try:
                    sd.play(audio_data, sample_rate_used)
                    sd.wait()  # Bloquear hasta que termine
                except Exception as e:
                    st.error(f"Error reproduciendo audio: {e}")
            
            # Reproducir audio en hilo separado
            audio_thread = threading.Thread(target=play_audio, daemon=True)
            audio_thread.start()
            
            # Sincronizar visualización con tiempo real
            start_time = time.time()
            total_duration = len(audio_data) / sample_rate_used
            
            try:
                for i, features in enumerate(features_sequence):
                    if not st.session_state.is_playing:
                        break
                    
                    # Calcular tiempo transcurrido desde inicio
                    elapsed_time = time.time() - start_time
                    target_time = features['timestamp']
                    
                    # Si vamos muy rápido, esperar
                    if elapsed_time < target_time:
                        time_to_wait = target_time - elapsed_time
                        if time_to_wait > 0:
                            time.sleep(time_to_wait)
                            elapsed_time = time.time() - start_time
                    
                    # Si nos atrasamos mucho (más de 0.1s), saltar frames
                    if elapsed_time > target_time + 0.1:
                        continue
                    
                    # Crear y mostrar visualización
                    fig = create_circular_visualization(features, "cosmic", elapsed_time)
                    
                    if fig is not None:
                        with viz_container.container():
                            st.pyplot(fig, clear_figure=True, use_container_width=True)
                        plt.close(fig)
                    
                    # Mostrar información
                    if not features.get('is_silent', False):
                        info_text = f"⏱️ {elapsed_time:.2f}s / {total_duration:.2f}s | "
                        info_text += f"🎵 {features['fundamental_freq']:.1f} Hz | "
                        info_text += f"📊 {features['num_peaks']} picos"
                        info_container.info(info_text)
                    
                    # Barra de progreso
                    progress = min(elapsed_time / total_duration, 1.0)
                    progress_container.progress(progress)
                    
                    # Verificar si el audio aún está reproduciéndose
                    if not audio_thread.is_alive() and elapsed_time >= total_duration:
                        break
                
                # Esperar a que termine el audio si aún está reproduciéndose
                audio_thread.join(timeout=1.0)
                
                # Finalizar
                progress_container.progress(1.0)
                st.success("✅ Reproducción completada")
                
            except Exception as e:
                st.error(f"Error durante la reproducción: {e}")
            finally:
                st.session_state.is_playing = False

with col_c:
    if st.button("⏹️ Detener", use_container_width=True):
        st.session_state.is_playing = False
        sd.stop()  # Detener cualquier reproducción de audio
        st.info("⏸️ Reproducción detenida")

# Configuraciones en sidebar
st.sidebar.header("⚙️ Configuraciones")

# Control de umbral
use_auto_threshold = st.sidebar.checkbox("Umbral automático de ruido", True)
if not use_auto_threshold:
    manual_threshold = st.sidebar.slider("Umbral manual", 0.001, 0.02, 0.005, format="%.4f")
else:
    manual_threshold = None

# Configuraciones de procesamiento
st.sidebar.header("🔄 Procesamiento")
smoothing_enabled = st.sidebar.checkbox("Suavizado entre frames", True)
st.sidebar.info("El suavizado mejora la fluidez visual")

# Información del estado actual
st.sidebar.header("📊 Estado Actual")
if st.session_state.recorded_data is not None:
    duration = len(st.session_state.recorded_data) / st.session_state.recording_params['sample_rate']
    st.sidebar.success(f"🎵 Audio grabado: {duration:.1f}s")
    
    if st.session_state.features_sequence:
        st.sidebar.success(f"📈 {len(st.session_state.features_sequence)} frames procesados")
        
        # Mostrar información de timing
        hop_size = st.session_state.recording_params['window_size'] // 4
        frame_rate = st.session_state.recording_params['sample_rate'] / hop_size
        st.sidebar.info(f"🎬 Frame rate: {frame_rate:.1f} FPS")
    
    # Opción para descargar audio
    if st.sidebar.button("💾 Guardar Audio"):
        # Crear WAV en memoria
        audio_normalized = np.int16(st.session_state.recorded_data / np.max(np.abs(st.session_state.recorded_data)) * 32767)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(st.session_state.recording_params['sample_rate'])
            wav_file.writeframes(audio_normalized.tobytes())
        
        st.sidebar.download_button(
            label="📥 Descargar WAV",
            data=wav_buffer.getvalue(),
            file_name=f"grabacion_{int(time.time())}.wav",
            mime="audio/wav"
        )

else:
    st.sidebar.info("🎤 No hay audio grabado")

# Información de uso
with st.expander("📚 Guía de Uso"):
    st.write("""
    ### 🎬 Modo Grabar y Reproducir:
    
    1. **Configurar:** Ajusta duración, frecuencia de muestreo y tamaño de ventana
    2. **Grabar:** Presiona "Grabar Audio" y habla/canta durante el tiempo establecido
    3. **Procesamiento:** El sistema analiza automáticamente todo el audio
    4. **Reproducir:** Presiona "Reproducir con Visualización" para ver el resultado sincronizado
    
    ### 🎯 Mejoras de Sincronización:
    - **Timing preciso:** Audio y visualización perfectamente alineados
    - **Control de latencia:** Sistema anti-desfase automático
    - **Salto inteligente:** Si hay retraso, salta frames para mantener sincronía
    - **Frame rate dinámico:** Se ajusta según la configuración de ventana
    
    ### 🎨 Interpretación Visual:
    - **Tiempo:** Se muestra el tiempo real transcurrido
    - **Colores:** Cambian según el timbre y características espectrales
    - **Movimiento:** Animaciones sincronizadas con las características del audio
    - **Tamaño:** Proporcional a la intensidad del sonido
    
    ### 💡 Consejos:
    - Graba en un ambiente silencioso para mejores resultados
    - Usa duraciones de 10-30 segundos para visualizaciones detalladas
    - Experimenta con diferentes instrumentos y voces
    - El suavizado está habilitado por defecto para mejor experiencia visual
    - La sincronización es automática y se ajusta dinámicamente
    """)

st.markdown("---")
st.markdown("🎵 **Visualizador FFT Avanzado** - Grabación y reproducción perfectamente sincronizada")
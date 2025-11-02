import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# ---- ¡Ajusta aquí si quieres un GIF más corto o más largo! ----
duration = 5.0  # 5 segundos es bueno para un GIF
fps = 30        # 30 fps es ideal para GIFs suaves y de tamaño razonable
total_frames = int(duration * fps)

# Paramètres des ondes
freq1 = 2.0     # Fréquence de l'onde de grande amplitude (Hz)
amp1 = 3.0      # Amplitude majeure
freq2 = 5.0     # Fréquence de l'onde de petite amplitude (Hz)
amp2 = 1.0      # Amplitude mineure

# Paramètres de rotation
phi_speed = 4.0  # Vitesse de rotation en rad/s

# Configurer la figure et les axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Simulation de Mouvement en Coordonnées Polaires', 
             fontsize=14, fontweight='bold')

# Temps pour toute la simulation
t_total = np.linspace(0, duration, 1000)

# Calculer les ondes complètes pour afficher dans le graphique supérieur
wave1_total = amp1 * np.sin(2 * np.pi * freq1 * t_total)
wave2_total = amp2 * np.sin(2 * np.pi * freq2 * t_total)
rayon_total = wave1_total + wave2_total + 4.0

# Configurer le graphique supérieur
ax1.plot(t_total, wave1_total, 'b-', alpha=0.6, linewidth=2, label=f'Onde 1: A={amp1}, f={freq1}Hz')
ax1.plot(t_total, wave2_total, 'r-', alpha=0.6, linewidth=2, label=f'Onde 2: A={amp2}, f={freq2}Hz')
ax1.plot(t_total, rayon_total, 'g-', linewidth=3, label='Rayon Combiné')
ax1.set_xlim(0, duration)
ax1.set_ylim(-1, 8)
ax1.set_xlabel('Temps (s)')
ax1.set_ylabel('Amplitude / Rayon')
ax1.set_title('Ondes Sinusoïdales et Rayon Résultant')
ax1.grid(True, alpha=0.3)
ax1.legend()
time_line = ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)

# Configurer le graphique inférieur
ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)
ax2.set_xlabel('Position X')
ax2.set_ylabel('Position Y')
ax2.set_title('Mouvement du Point en Coordonnées Polaires')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
for radius in [1, 2, 3, 4, 5, 6, 7]:
    circle_ref = plt.Circle((0, 0), radius, fill=False, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.add_patch(circle_ref)
circle = plt.Circle((0, 0), 0.15, color='red', zorder=5)
ax2.add_patch(circle)
trajectory_x, trajectory_y = [], []
trajectory_line, = ax2.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Trajectoire')
radius_line, = ax2.plot([0, 0], [0, 0], 'k-', linewidth=2, alpha=0.7, label='Rayon')
info_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

def animate(frame):
    current_time = frame * duration / total_frames
    rayon_onde1 = amp1 * np.sin(2 * np.pi * freq1 * current_time)
    rayon_onde2 = amp2 * np.sin(2 * np.pi * freq2 * current_time)
    rayon_total = rayon_onde1 + rayon_onde2 + 4.0
    phi = phi_speed * current_time
    x_pos, y_pos = rayon_total * np.cos(phi), rayon_total * np.sin(phi)
    circle.center = (x_pos, y_pos)
    time_line.set_xdata([current_time])
    radius_line.set_data([0, x_pos], [0, y_pos])
    trajectory_x.append(x_pos)
    trajectory_y.append(y_pos)
    if len(trajectory_x) > 150:
        trajectory_x.pop(0)
        trajectory_y.pop(0)
    trajectory_line.set_data(trajectory_x, trajectory_y)
    phi_degrees = np.degrees(phi) % 360
    info_text.set_text(f'Temps: {current_time:.2f}s\nRayon: {rayon_total:.2f}\nAngle φ: {phi_degrees:.1f}°\n'
                      f'X: {x_pos:.2f}\nY: {y_pos:.2f}')
    return circle, time_line, trajectory_line, radius_line, info_text

# Créer l'animation
anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=False)

# Ajuster la mise en page
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajuste para el supertítulo

# --- MODIFICACIÓN CLAVE PARA GIF ---

# Nombre del archivo de salida
output_filename_gif = 'mouvement_polaire.gif'

print(f"Generando el GIF: {output_filename_gif}")
print("Esto puede tardar unos segundos o minutos dependiendo de la duración...")

# Guardar la animación usando el writer 'pillow'
# dpi (dots per inch) controla la resolución. 100-150 es bueno para presentaciones.
start_time = time.time()
anim.save(output_filename_gif, writer='pillow', fps=fps, dpi=100)
end_time = time.time()

print(f"✅ ¡GIF guardado con éxito! -> {output_filename_gif}")
print(f"Tiempo de renderizado: {end_time - start_time:.2f} segundos")

# Cerrar la figura para liberar memoria
plt.close(fig)

# Ya no se necesita plt.show()
# plt.show()
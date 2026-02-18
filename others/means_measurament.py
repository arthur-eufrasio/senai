import numpy as np
import matplotlib.pyplot as plt
from creation_rs_profile import curve_equation_final_fit

surface_measurement_distance = 4.0
resolution_points = 40
distance = np.linspace(0, surface_measurement_distance, resolution_points)
dx = distance[1] - distance[0]

real_stress = curve_equation_final_fit(distance)

beam_diameter = 0.5
overlap_percentage = 0.50
lsp_radius = 3.5 / 2.0

step_size = beam_diameter * (1 - overlap_percentage)
measurement_centers = np.arange(0, surface_measurement_distance, step_size)

points_per_beam = int(beam_diameter / dx)
kernel_radius = points_per_beam // 2

A = np.zeros((resolution_points, resolution_points))
for i in range(resolution_points):
    start_idx = max(0, i - kernel_radius)
    end_idx = min(resolution_points, i + kernel_radius)
    window_length = end_idx - start_idx
    if window_length > 0:
        A[i, start_idx:end_idx] = 1.0 / window_length

discrete_measurements = []
valid_centers = []

for center in measurement_centers:
    idx = (np.abs(distance - center)).argmin()
    
    start_idx = max(0, idx - kernel_radius)
    end_idx = min(resolution_points, idx + kernel_radius)
    
    if start_idx < end_idx:
        measured_val = np.mean(real_stress[start_idx:end_idx])
        discrete_measurements.append(measured_val)
        valid_centers.append(center)

noise_level = 5.0
discrete_measurements = np.array(discrete_measurements)
discrete_measurements += np.random.normal(0, noise_level, len(discrete_measurements))

interpolated_measurement = np.interp(distance, valid_centers, discrete_measurements)

reconstructed_stress, _, _, _ = np.linalg.lstsq(A, interpolated_measurement, rcond=0.05)

plt.figure(figsize=(12, 7))

plt.plot(distance, real_stress, 'k--', linewidth=2, label='Real Stress Profile (High Res)')

plt.errorbar(valid_centers, discrete_measurements, xerr=beam_diameter/2, fmt='ro', 
             capsize=3, alpha=0.7, label=f'Discrete Measurements (Overlap {overlap_percentage*100:.0f}%)')

plt.plot(distance, interpolated_measurement, 'r-', alpha=0.3, label='Interpolated Measurement')

plt.plot(distance, reconstructed_stress, 'g-', linewidth=2, label='Reconstructed (Deconvolved)')

plt.axvline(x=lsp_radius, color='b', linestyle=':', alpha=0.5, label='LSP Radius Edge')

plt.xlabel('Distance from Center (mm)')
plt.ylabel('Residual Stress (MPa)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'LSP Profile Reconstruction: Beam {beam_diameter}mm, Overlap {overlap_percentage*100}%')
plt.show()
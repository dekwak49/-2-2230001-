import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
def gaussian(x, height, mean, sigma):
    return height * np.exp(-(x - mean)**2 / (2 * sigma**2))

def double_gaussian(x, h1, mu1, s1, h2, mu2, s2):
    return gaussian(x, h1, mu1, s1) + gaussian(x, h2, mu2, s2)

from tkinter import Tk, filedialog

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="hist2.csv 파일을 선택하세요",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
root.destroy()

if not file_path:
    raise SystemExit("파일 선택이 취소되었습니다.")

file_path = Path(file_path)
data = np.genfromtxt(file_path, delimiter=",", encoding="utf-8-sig")
data = data[np.isfinite(data[:, :2]).all(axis=1)]

energy = data[:, 0]
count = data[:, 1]
initial_guess = [1000, 1.0, 0.2, 550, 2.0, 0.6]
lower_bounds = [0, energy.min(), 1e-6, 0, energy.min(), 1e-6]
upper_bounds = [
    np.inf, energy.max(), np.inf,
    np.inf, energy.max(), np.inf
]
popt, pcov = curve_fit(
    double_gaussian,
    energy,
    count,
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    maxfev=20000
)
particle_A, particle_B = sorted(
    [popt[:3], popt[3:]],
    key=lambda p: p[1]
)

hA, muA, sigmaA = particle_A
hB, muB, sigmaB = particle_B
area_A = hA * sigmaA * np.sqrt(2 * np.pi)
area_B = hB * sigmaB * np.sqrt(2 * np.pi)

ratio = area_A / area_B
fraction_A = area_A / (area_A + area_B)
fraction_B = area_B / (area_A + area_B)
print("===== Gaussian fitting results =====")
print("A: lower-energy particle")
print(f"Height = {hA:.4f}")
print(f"Mean   = {muA:.4f}")
print(f"Sigma  = {sigmaA:.4f}")

print("\nB: higher-energy particle")
print(f"Height = {hB:.4f}")
print(f"Mean   = {muB:.4f}")
print(f"Sigma  = {sigmaB:.4f}")

print("\n===== Production ratio =====")
print(f"A : B = {ratio:.4f} : 1")
print(f"A fraction = {fraction_A:.2%}")
print(f"B fraction = {fraction_B:.2%}")
x_fit = np.linspace(energy.min(), energy.max(), 1000)

y_A = gaussian(x_fit, hA, muA, sigmaA)
y_B = gaussian(x_fit, hB, muB, sigmaB)
y_total = y_A + y_B

plt.figure(figsize=(10, 6))

plt.scatter(
    energy, count,
    color="gray", s=18, label="Measured data"
)

plt.plot(
    x_fit, y_total,
    color="red", linewidth=2, label="Total Gaussian fit"
)

plt.plot(
    x_fit, y_A,
    "b--", linewidth=2, label="Particle A"
)

plt.plot(
    x_fit, y_B,
    "g--", linewidth=2, label="Particle B"
)

plt.xlabel("Energy")
plt.ylabel("Count")
plt.title(f"Energy spectrum fitting (A : B = {ratio:.3f} : 1)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    file_path.parent / "gaussian_fit.png",
    dpi=300
)

plt.show()

"""Pulse compression with a linear chirp and a matched filter.

Prints the numbers quoted in the post and writes the figures to ./figures.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming, taylor

C = 3e8
B = 2e6
T = 50e-6
FS = 40e6
N = int(T * FS)
K = B / T
TB = B * T
T_AXIS = (np.arange(N) - N / 2) / FS

INK, MUTE, RED, BLUE = "#1f2933", "#6b7280", "#c0392b", "#2b6cb0"
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
chirp = np.exp(1j * np.pi * K * T_AXIS**2)
plain = np.ones(N, complex)


def complex_noise(n, variance):
    return np.sqrt(variance / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def noise_variance(snr_db):
    return FS / B * 10 ** (-snr_db / 10)


def matched_filter(rx, ref, window=None):
    taps = np.conj(ref[::-1])
    if window is not None:
        taps = taps * window
    return np.convolve(rx, taps)


def to_db(y):
    a = np.abs(y)
    return 20 * np.log10(np.maximum(a / a.max(), 1e-6))


def width_and_sidelobe(y):
    a = np.abs(y)
    a = a / a.max()
    peak = a.argmax()
    half = 1 / np.sqrt(2)
    left = peak
    while a[left] > half:
        left -= 1
    right = peak
    while a[right] > half:
        right += 1
    x_left = left + (half - a[left]) / (a[left + 1] - a[left])
    x_right = right - 1 + (a[right - 1] - half) / (a[right - 1] - a[right])
    width = (x_right - x_left) / FS * C / 2
    j = peak
    while a[j + 1] < a[j]:
        j += 1
    return width, 20 * np.log10(a[j:].max())


def weighting_loss_db(w):
    return 10 * np.log10(np.sum(w) ** 2 / np.sum(w**2) / N)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


taylor_win = taylor(N, nbar=4, sll=30)
y_plain_mf = matched_filter(chirp, chirp)
y_hamming = matched_filter(chirp, chirp, hamming(N))
y_taylor = matched_filter(chirp, chirp, taylor_win)
range_axis = (np.arange(len(y_plain_mf)) - (N - 1)) / FS * C / 2

print(f"TB = {TB:.0f}, range res c/2B = {C / 2 / B:.1f} m, plain-pulse res cT/2 = {C * T / 2 / 1e3:.2f} km")
for name, y in (("unweighted", y_plain_mf), ("hamming", y_hamming), ("taylor30", y_taylor)):
    w, sl = width_and_sidelobe(y)
    print(f"{name:11s} 3dB width {w:6.1f} m   peak sidelobe {sl:6.1f} dB")
for name, w in (("hamming", hamming(N)), ("taylor30", taylor_win)):
    print(f"{name} SNR loss vs matched: {weighting_loss_db(w):.2f} dB")

trials = 3000
noise_out = [
    np.abs(matched_filter(complex_noise(N, noise_variance(0)), chirp)[N - 1]) ** 2
    for _ in range(trials)
]
snr_out = N**2 / np.mean(noise_out)
print(f"Input SNR (in B) = 0 dB -> output peak SNR = {10 * np.log10(snr_out):.2f} dB ; 10log10(TB) = {10 * np.log10(TB):.2f} dB")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
us = T_AXIS * 1e6
inst_freq = K * T_AXIS / 1e6
a1.plot(us, chirp.real, color=BLUE, lw=0.7)
a1.set_ylabel("Amplitude (real part)")
a1.set_title("A linear chirp: constant amplitude, frequency ramps through the pulse", loc="left", fontsize=11)
a2.plot(us, inst_freq, color=RED, lw=2)
a2.set_ylabel("Instantaneous frequency (MHz)")
a2.set_xlabel("Time (µs)")
a2.annotate(f"B = {B / 1e6:.0f} MHz\nT = {T * 1e6:.0f} µs", (us[N // 4], inst_freq[N // 4]),
            xytext=(-10, 30), textcoords="offset points", fontsize=9)
for a in (a1, a2):
    style(a)
save(fig, "fig_chirp.png")

fig, ax = plt.subplots(figsize=(8, 5))
near = np.abs(range_axis) < 1500
ax.plot(range_axis[near], to_db(y_plain_mf)[near], color=RED, lw=1.6, label="Matched filter (unweighted)")
ax.plot(range_axis[near], to_db(y_taylor)[near], color=BLUE, lw=1.6, label="Taylor-weighted (−30 dB design)")
ax.set_ylim(-70, 3)
ax.set_xlabel("Range offset from target (m)")
ax.set_ylabel("Output (dB re peak)")
ax.set_title(f"50 µs chirp, B = 2 MHz: a {C * T / 2 / 1e3:.1f} km-long pulse compressed to ~{C / 2 / B:.0f} m", loc="left", fontsize=11)
ax.legend(frameon=False)
style(ax)
save(fig, "fig_compressed.png")


def two_targets(ref, separation_m, snr_db):
    length, start = 6 * N, 2 * N
    delay = int(round(2 * separation_m / C * FS))
    rx = np.zeros(length, complex)
    rx[start:start + N] += ref
    rx[start + delay:start + delay + N] += 0.7 * ref
    rx += complex_noise(length, noise_variance(snr_db))
    y = np.abs(matched_filter(rx, ref))
    return (np.arange(len(y)) - (N - 1) - start) / FS * C / 2, y


fig, axs = plt.subplots(2, 2, figsize=(9, 5.6), sharex="col")
for col, sep in enumerate((300, 60)):
    for row, (ref, label, color) in enumerate(((plain, "Plain 50 µs pulse", MUTE), (chirp, "50 µs chirp, B = 2 MHz", RED))):
        r, a = two_targets(ref, sep, -5)
        ax = axs[row, col]
        window = (r > -500) & (r < sep + 500)
        ax.plot(r[window], a[window] / np.abs(matched_filter(ref, ref)).max(), color=color, lw=1.4)
        ax.set_ylim(0, 1.9)
        for pos in (0, sep):
            ax.axvline(pos, color=INK, lw=0.8, ls=":")
        if row == 0:
            ax.set_title(f"Two targets {sep} m apart", loc="left", fontsize=10)
        ax.set_ylabel(label, fontsize=8.5)
        style(ax)
axs[1, 0].set_xlabel("Range (m)")
axs[1, 1].set_xlabel("Range (m)")
fig.suptitle("Same pulse length, same energy: only the modulation differs", x=0.01, ha="left", fontsize=11)
save(fig, "fig_two_targets.png")

h = np.conj(chirp[::-1])
snr_in = -5.0
variance = noise_variance(snr_in)
length, start = 8 * N, 3 * N + 777
rx = np.zeros(length, complex)
rx[start:start + N] += chirp
rx += complex_noise(length, variance)
y = np.convolve(rx, h)
range_km = (np.arange(len(y)) - (N - 1)) / FS * C / 2 / 1e3
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6))
a1.plot(np.arange(length) / FS * 1e6, np.abs(rx), color=MUTE, lw=0.6)
a1.axvspan(start / FS * 1e6, (start + N) / FS * 1e6, color=RED, alpha=0.12)
a1.set_xlabel("Time (µs)")
a1.set_ylabel("|received|")
a1.set_title(f"Before: the echo (shaded) is buried in noise, input SNR {snr_in:.0f} dB", loc="left", fontsize=10.5)
a2.plot(range_km, np.abs(y) / N, color=RED, lw=1)
a2.set_xlim(0, 50)
a2.set_xlabel("Range (km)")
a2.set_ylabel("|compressed| / N")
a2.set_title("After the matched filter: a clear spike", loc="left", fontsize=10.5)
for a in (a1, a2):
    style(a)
save(fig, "fig_noise_before_after.png")

noise_only = np.convolve(complex_noise(length, variance), h)
noise_power = np.mean(np.abs(noise_only[N:-N]) ** 2)
print(f"in-band input SNR {snr_in} dB -> compressed peak/noise-power {10 * np.log10(N**2 / noise_power):.1f} dB (expect {snr_in + 10 * np.log10(TB):.1f})")
print("peak at range km", range_km[np.abs(y).argmax()], " true", start / FS * C / 2 / 1e3)


def doppler_peak(fd):
    shifted = chirp * np.exp(2j * np.pi * fd * T_AXIS)
    out = np.abs(np.convolve(shifted, h))
    return (out.argmax() - (N - 1)) / FS, out.max() / N


doppler = np.linspace(-200e3, 200e3, 81)
delays, levels = np.array([doppler_peak(fd) for fd in doppler]).T
slope = np.polyfit(doppler, delays, 1)[0]
print(f"peak delay shift per Hz: {slope:.3e} s/Hz ; -1/k = {-1 / K:.3e} ; +1/k = {1 / K:.3e}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
a1.plot(doppler / 1e3, delays * C / 2, color=RED, lw=2)
a1.set_xlabel("Doppler shift (kHz)")
a1.set_ylabel("Shift of compressed peak (m)")
a1.set_title("Peak moves with Doppler", loc="left", fontsize=10.5)
a2.plot(doppler / 1e3, 20 * np.log10(levels), color=BLUE, lw=2)
a2.set_xlabel("Doppler shift (kHz)")
a2.set_ylabel("Peak level (dB re no Doppler)")
a2.set_title("…and the peak level droops slowly", loc="left", fontsize=10.5)
for a in (a1, a2):
    style(a)
save(fig, "fig_doppler.png")

for fd in (0, 10e3, 50e3, 100e3):
    delay, level = doppler_peak(fd)
    print(f"fd={fd / 1e3:6.0f} kHz  shift {delay * C / 2:8.1f} m   level {20 * np.log10(level):6.2f} dB")
fd_xband = 2 * 100 / (C / 10e9)
print("X-band 100 m/s fd =", fd_xband, "Hz ; shift m =", fd_xband / K * C / 2)

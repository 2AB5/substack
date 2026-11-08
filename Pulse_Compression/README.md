# Pulse compression: linear chirp and matched filter

Code and figures for the post "Why Radars Chirp".

A 50 µs chirp with B = 2 MHz is compressed by a matched filter to a peak about 66 m wide. The time-bandwidth product is T×B = 100 (20 dB).

## Files

| File | Purpose |
|---|---|
| `pulse_compression.py` | Simulation. Prints the numbers quoted in the post and writes the figures to `figures/`. |
| `pulse_compression_base.m` | Same simulation in base MATLAB, no toolboxes. |
| `pulse_compression.grc` | GNU Radio Companion flowgraph with an input SNR slider. |
| `gr_pulse_compression_headless.py` | The same flowgraph without a GUI, with checks printed to the terminal. |

## Python

```
pip install -r requirements.txt
python pulse_compression.py
```

## GNU Radio (3.10 or newer)

Open `pulse_compression.grc` in GNU Radio Companion and run it. The top plot is the received signal and the bottom plot is the output of the matched filter. The spike at 50 µs is the filter delay.

For the headless check, run `gr_pulse_compression_headless.py` with the Python interpreter that has the GNU Radio bindings installed. It should print:

- noise-free output matching NumPy to about 7e-7 of the peak, with the peak equal to N = 2000
- a noise source variance of 1.000 for `amp = 1`
- a spike about 15 dB above the noise at an input SNR of −5 dB

The Noise Source amplitude is the standard deviation of the complex noise: `noise_amp = sqrt(samp_rate / B * 10**(-snr_dB / 10))`.

To see a plain pulse, set the `chirp` variable to `np.ones(N)`.

## Parameters

| Name | Value |
|---|---|
| Sample rate | 40 MHz |
| Bandwidth B | 2 MHz |
| Pulse length T | 50 µs |
| Resolution c/2B | 75 m |
| Plain pulse resolution cT/2 | 7.5 km |

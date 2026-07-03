# VectorSpin

Interactive teaching tool for the **DFT/IDFT** and **FIR filtering**. A signal is drawn as a
chain of rotating phasors (vectors) whose tip-to-tail sum reconstructs the waveform
sample-by-sample, animated in real time, with linked time-domain and frequency-domain
magnitude & phase plots around a central IQ (complex-plane) view.

By [The DSP Coach](https://dsp-coach.com).

## Two versions

- **Browser app — [`vector_spin.html`](vector_spin.html)** — fully self-contained, no install.
  Open it in any modern browser. Type a signal (time or frequency samples), watch the phasor
  reconstruction, and export the animation as a looping animated GIF.
  <!-- Live demo (GitHub Pages): https://cdboschen.github.io/vector-spin/vector_spin.html -->
- **Desktop app — [`vector_spin.py`](vector_spin.py)** — the original Matplotlib/Qt version.
  Run `python vector_spin.py` (requires `numpy`, `scipy`, `matplotlib`, `PyQt5`, and optional
  `pyperclip`; see [`vectorspin.yml`](vectorspin.yml) for a conda environment).

## License

MIT — see [LICENSE](LICENSE). © 2026 The DSP Coach.

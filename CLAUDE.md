# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VectorSpin is an interactive teaching tool for the DFT/IDFT and FIR filtering. It renders a
signal as a chain of rotating phasors (vectors) whose tip-to-tail sum reconstructs the
waveform sample-by-sample, animated in real time, with linked time-domain and
frequency-domain magnitude/phase strip charts around a central IQ (complex-plane) plot.

There are two independent implementations of the same idea:

- **`vector_spin.html`** — a self-contained browser app (HTML/CSS/vanilla JS on a `<canvas>`),
  **no dependencies and no build step**. This is the version embedded on the website
  (dsp-coach.com) via an `<iframe>`. It can export the animation as an animated GIF entirely
  client-side.
- **`vector_spin.py`** — the original Matplotlib desktop app (opens a Qt window). Same model,
  rendered with Matplotlib widgets.

This is a personal/educational project: no test suite, no packaging. Licensed MIT,
© The DSP Coach (see `LICENSE`).

## Running

**Web** — just open `vector_spin.html` in any modern browser. Nothing to install. "Download
GIF" writes a looping animated GIF via a built-in median-cut quantizer + LZW encoder (no
external tools).

**Desktop:**

```powershell
python vector_spin.py
```

Requires an interactive Qt window — the module hard-codes `mlib.use("Qt5Agg")`, so **PyQt5
must be installed** and this cannot run headless. Dependencies: `numpy`, `scipy`,
`matplotlib`, `PyQt5`, and optional `pyperclip` for clipboard copy/paste of the input array
(degrades gracefully if missing). `vectorspin.yml` is a conda environment for these.

At the REPL, `run()` returns `(vs, ui)` — the `VectorSpin` and `SpinUI` objects — so you can
drive them programmatically (e.g. `ui.input_values("[1,2+3j,3]")`, `ui.save_ani("out.gif")`).
`SpinUI.save_ani()` writes a GIF via the `ffmpeg` writer, so saving GIFs from the desktop app
needs ffmpeg on PATH.

## Shared model (both versions)

- `tvalues` / `fvalues` — the complex time-domain and frequency-domain arrays. Whichever one
  the user typed is the *input*; the other is derived (normalized DFT/IDFT with `1/N` scaling
  and phase ramps for shifts).
- `input_mode` (`'time'`|`'freq'`) — which domain the user's array represents.
- `iq_mode` (`'time'`|`'freq'`) — which domain the central animated IQ reconstruction shows
  (IFFT of freq bins, or FFT of time samples). Controls spin direction.
- `tshift` / `fshift` — integer index offsets applied as linear phase ramps, used to show
  non-causal / circularly-shifted reconstructions.

## Architecture — `vector_spin.html`

One file, organized into commented sections:

1. **Complex + DSP math** — small complex helpers plus naive DFT/IDFT (replaces numpy/scipy).
2. **Input parser** — a hand-written recursive-descent parser (`parseInput`). Accepts a
   comma-separated list of real/complex numbers with **optional brackets** (`1, 2+3j, 3` or
   `[1,1,1,1,1]`), supports `+ - * /`, parentheses, and the `j` imaginary suffix. **No `eval`**
   (unlike the desktop app).
3. **Plot-data compute** — fills the conjugate domain and precomputes the "ideal" smooth curves.
4. **Animation state** (`anim`) + `step()` — advances the phase and accumulates the phasor
   tip-to-tail sum; bounded history arrays drive the fading blue trace and the strip-chart dots.
5. **Canvas renderer** — `makeMapper` (data→pixel mapping, equal-aspect for the square IQ
   plot), `drawAxes`, `drawStripPlot`, `drawIqPlot`, `drawArrow` (the red DFT-direction arrow).
   Layout: time plots (left column), IQ plot (center, square), freq plots (right column).
6. **GIF export** — `quantizeFrames` (median-cut to a 256-colour palette + cached nearest-colour
   mapper), `lzwEncode` (GIF-flavour variable-width LZW, code-width growth timed to match
   giflib/browser decoders), `assembleGif` (GIF89a with a NETSCAPE looping block). `saveGif`
   captures one full revolution frame-by-frame off the main loop and downloads a Blob.
7. **UI wiring** — the control bar (Time shift · IQ Mode · Freq shift under the plots), input
   row, sliders, Start/Stop, Clear, Download GIF.

## Architecture — `vector_spin.py`

Three classes, one entry point (`run()`):

- **`VectorSpin`** — the model + all plotting. Owns the five subplots plus the central IQ plot.
  The animation (`create_ani`) uses `FuncAnimation` with `blit=True`: `VectorBuilder()` is an
  infinite generator yielding phase; `animate()` sums the phasors tip-to-tail at that phase,
  appending the running tip position to bounded `deque` history buffers (the fading blue trace)
  that mirror onto the magnitude/phase strip charts. `refresh_plots()` is the central redraw:
  it rebuilds `plot_time`, `plot_freq`, `plot_iq`, stops any running animation, clears history,
  and redraws the canvas. Most setters call it.

- **`SpinUI`** — all Matplotlib `widgets` (TextBox, Sliders, Buttons, radio buttons) and their
  callbacks. Holds a reference to the `VectorSpin` as `self.vs` and translates UI events into
  `vs.set_*` calls. **Input is parsed with `eval()`** on the text box contents (so entries like
  `sig.firls(31,[0,.4,.6,1],[1,1,0,0])` work) — intentional for this trusted local tool.

- **`MyRadioButtons`** — a `widgets.RadioButtons` subclass rendering selectable circle markers
  in a legend. This is the main source of Matplotlib-version fragility: `RadioButtons`
  internals changed post-3.7 (`legendHandles` → `legend_handles`, the `CallbackRegistry`
  signature, and a reimplemented `set_active`), which this subclass works around.

## Notes

- The bottom of `vector_spin.py` has commented "interesting cases" — specific input arrays and
  mode/shift settings that produce instructive animations. Useful for manual verification.
- When changing DSP/animation behavior, keep the two implementations in sync where it matters
  (the visible model semantics), even though their rendering stacks are entirely different.

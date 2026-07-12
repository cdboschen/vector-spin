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

## Permalinks

The browser app mirrors its full configuration into the URL hash, so any configuration can
be shared as a link — the address bar updates live as you change things, and the
**Copy Link** button copies the current permalink to the clipboard. `time=` / `freq=` carry
the input array and select its domain; the remaining keys are the same ones the CSV settings
use. Keys omitted from a permalink reset to their defaults when it is opened.

Example — the 5-sample moving-average filter with the IQ plot showing the frequency domain:

```
vector_spin.html#time=[1,1,1,1,1]&iq_mode=freq&update_time=25
```

Supported keys: `time` / `freq` (the input array), `iq_mode` (`time`|`freq`), `time_shift`,
`freq_shift`, `update_time` (ms), `frames`. Values may be hand-typed as above or
percent-encoded — both parse.

## Embedding (auto-resizing iframe)

`vector_spin.html` reports its content height to the parent page via `postMessage`, so a
cross-origin embed can shrink the iframe to fit (no scrollbars, no fixed height). On the host
page (e.g. a Squarespace Code Block) use:

```html
<iframe id="vectorspin"
        src="https://cdboschen.github.io/vector-spin/vector_spin.html"
        style="width:100%; border:0; display:block;"
        allow="clipboard-write"
        scrolling="no"></iframe>
<script>
  window.addEventListener('message', function(e){
    var d = e.data;
    if(d && d.type === 'vectorspin:height'){
      document.getElementById('vectorspin').style.height = d.height + 'px';
    }
  });
</script>
```

The iframe re-posts its height on load, on resize, and whenever the content reflows, so the
host resizes automatically.

## License

MIT — see [LICENSE](LICENSE). VectorSpin © 2026 The DSP Coach.

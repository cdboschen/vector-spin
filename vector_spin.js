"use strict";
(function(){

// ---------------------------------------------------------------------------
// 1. Complex + DSP math (replaces numpy / scipy.fft)
// ---------------------------------------------------------------------------
const TAU = 2 * Math.PI;
const C   = (re, im=0) => ({re, im});
const cadd  = (a,b) => C(a.re+b.re, a.im+b.im);
const cmul  = (a,b) => C(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
const cscale= (a,s) => C(a.re*s, a.im*s);
const cexpj = (t)   => C(Math.cos(t), Math.sin(t));   // e^{j t}
const cabs  = (a)   => Math.hypot(a.re, a.im);
const cang  = (a)   => Math.atan2(a.im, a.re);

// Generic O(N^2) DFT / IDFT (N is small for teaching inputs).
function dft(x){
  const N = x.length, out = new Array(N);
  for(let k=0;k<N;k++){
    let re=0, im=0;
    for(let n=0;n<N;n++){
      const t = -TAU*k*n/N, c=Math.cos(t), s=Math.sin(t);
      re += x[n].re*c - x[n].im*s;
      im += x[n].re*s + x[n].im*c;
    }
    out[k] = C(re, im);
  }
  return out;
}
function idft(X){
  const N = X.length, out = new Array(N);
  for(let n=0;n<N;n++){
    let re=0, im=0;
    for(let k=0;k<N;k++){
      const t = TAU*k*n/N, c=Math.cos(t), s=Math.sin(t);
      re += X[k].re*c - X[k].im*s;
      im += X[k].re*s + X[k].im*c;
    }
    out[n] = C(re/N, im/N);
  }
  return out;
}
// Zero-pad-or-truncate x to length n (matches numpy fft.fft(a, n)).
function pad(x, n){
  const out = new Array(n);
  for(let i=0;i<n;i++) out[i] = (i < x.length) ? C(x[i].re, x[i].im) : C(0,0);
  return out;
}
const fftPad  = (x, n) => dft(pad(x, n));
const ifftPad = (x, n) => idft(pad(x, n));
const scaleArr= (x, s) => x.map(v => cscale(v, s));
// np.roll(arr, shift): out[i] = arr[(i - shift) mod N]
function roll(arr, shift){
  const N = arr.length, out = new Array(N);
  for(let i=0;i<N;i++) out[i] = arr[((i-shift)%N + N)%N];
  return out;
}
const absArr = (x) => x.map(cabs);
const angArr = (x) => x.map(cang);
const maxAbs = (x) => x.reduce((m,v)=>Math.max(m, cabs(v)), 0);
function linspace(start, stop, n){ // endpoint=False
  const step = (stop-start)/n, out = new Array(n);
  for(let i=0;i<n;i++) out[i] = start + i*step;
  return out;
}

// ---------------------------------------------------------------------------
// 2. Input parser (safe — no eval). Supports: [ ... ], real/float literals,
//    imaginary suffix j (3j, 2+3j, 0.5+0.5j), unary +/-, + - * /, parentheses.
// ---------------------------------------------------------------------------
function parseInput(str){
  let i = 0;
  const s = str;
  function ws(){ while(i<s.length && /\s/.test(s[i])) i++; }
  function peek(){ ws(); return s[i]; }
  function eat(ch){ ws(); if(s[i]!==ch) throw new Error("expected "+ch); i++; }

  function primary(){
    ws();
    if(s[i]==='('){ i++; const v=expr(); eat(')'); return v; }
    // number, optional trailing j
    const m = /^[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?/.exec(s.slice(i));
    if(!m) throw new Error("number expected at "+i);
    i += m[0].length;
    let val = parseFloat(m[0]);
    ws();
    if(s[i]==='j' || s[i]==='J'){ i++; return C(0, val); }
    return C(val, 0);
  }
  function factor(){
    ws();
    if(s[i]==='-'){ i++; return cscale(factor(), -1); }
    if(s[i]==='+'){ i++; return factor(); }
    return primary();
  }
  function term(){
    let v = factor();
    for(;;){
      const c = peek();
      if(c==='*'){ i++; v = cmul(v, factor()); }
      else if(c==='/'){ i++; const d=factor(); const dn=d.re*d.re+d.im*d.im;
        v = C((v.re*d.re+v.im*d.im)/dn, (v.im*d.re-v.re*d.im)/dn); }
      else break;
    }
    return v;
  }
  function expr(){
    let v = term();
    for(;;){
      const c = peek();
      if(c==='+'){ i++; v = cadd(v, term()); }
      else if(c==='-'){ i++; v = cadd(v, cscale(term(), -1)); }
      else break;
    }
    return v;
  }

  ws();
  const bracketed = (peek() === '[');
  if(bracketed) eat('[');
  const atEnd = () => bracketed ? (peek() === ']') : (i >= s.length);
  const arr = [];
  if(!atEnd()){
    arr.push(expr());
    while(peek()===','){ i++; if(atEnd()) break; arr.push(expr()); }
  }
  if(bracketed) eat(']');
  ws();
  if(i !== s.length) throw new Error("trailing characters");
  if(arr.length === 0) throw new Error("empty array");
  return arr;
}

// ---------------------------------------------------------------------------
// 3. VectorSpin model
// ---------------------------------------------------------------------------
const vs = {
  tvalues: [], fvalues: [],
  input_mode: 'time', iq_mode: 'time',
  tshift: 0, fshift: 0,
  nsamps: 300, refresh: 100,
  // marker styles: [color, size]
  input_marker: ['blue', 6.3], result_marker: ['red', 6.3],
  phasor_color: '#2ca02c', history_color: '#1f77ff',
  get tmarker(){ return this.input_mode==='time' ? this.input_marker : this.result_marker; },
  get fmarker(){ return this.input_mode==='time' ? this.result_marker : this.input_marker; },
};

function computeFvalues(){
  const N = vs.tvalues.length; if(!N) return;
  let F = dft(vs.tvalues);
  F = F.map((v,k)=> cmul(cscale(v, 1/N), cexpj(-vs.tshift*TAU*k/N)));
  vs.fvalues = roll(F, -vs.fshift);
}
function computeTvalues(){
  const N = vs.fvalues.length; if(!N) return;
  let T = idft(vs.fvalues);
  T = T.map((v,k)=> cmul(v, cexpj(vs.fshift*TAU*k/N)));
  vs.tvalues = roll(T, -vs.tshift);
}

// ---- Plot data cache (recomputed on state change, not per frame) ----
let PD = null;      // plot data
let mappers = {};   // pixel mappers, rebuilt each render

// Largest radius reached by any vertex of the tip-to-tail phasor chain over a
// full revolution. Mirrors step()'s geometry exactly (same shifts/sign/phase
// grid) so the IQ limit can be sized to contain the whole chain.
function chainExtent(phasors){
  const N = phasors.length; if(!N) return 0;
  const iqTime = (vs.iq_mode==='time');
  const sign = iqTime ? 1 : -1;
  const poff = iqTime ? (TAU*vs.tshift/N) : (-TAU*vs.fshift/N);
  const ns = Math.max(1, vs.nsamps);
  let maxR2 = 0;
  for(let s=0;s<ns;s++){
    const signed = sign * (s*TAU/ns);
    let x=0, y=0;
    for(let k=0;k<N;k++){
      const m = iqTime ? (k+vs.fshift) : (k+vs.tshift);
      const p = cmul(phasors[k], cexpj(m*(signed+poff)));
      x += p.re; y += p.im;
      const r2 = x*x + y*y; if(r2 > maxR2) maxR2 = r2;
    }
  }
  return Math.sqrt(maxR2);
}

function computePlotData(){
  const N = vs.tvalues.length;
  const nsamps = vs.nsamps;
  if(!N){ PD = null; return; }

  // --- time domain markers ---
  const tindex = []; for(let k=0;k<N;k++) tindex.push(k + vs.tshift);
  const tmag = absArr(vs.tvalues), tph = angArr(vs.tvalues);

  // --- freq domain markers ---
  const findex = []; for(let k=0;k<N;k++) findex.push(k + vs.fshift);
  const fmag = absArr(vs.fvalues), fph = angArr(vs.fvalues);

  const data = {
    N, nsamps,
    time: { index:tindex, mag:tmag, ph:tph, marker:vs.tmarker,
            stems:(vs.iq_mode==='freq'), ideal:null,
            xlim:[vs.tshift-0.2, vs.tshift+N+0.2], magYlim:[0, 1.2*Math.max(...tmag,1e-9)] },
    freq: { index:findex, mag:fmag, ph:fph, marker:vs.fmarker,
            stems:(vs.iq_mode==='time'), ideal:null,
            xlim:[vs.fshift-0.2, vs.fshift+N+0.2], magYlim:[0, 1.2*Math.max(...fmag,1e-9)] },
    iq: { marker:(vs.iq_mode==='time'?vs.tmarker:vs.fmarker),
          points:(vs.iq_mode==='time'?vs.tvalues:vs.fvalues), ideal:null, lim:1 },
    historyTarget: vs.iq_mode  // 'time' -> time plots, 'freq' -> freq plots
  };

  // --- smooth "ideal" curve for the time plots (only when iq_mode == time) ---
  if(vs.iq_mode==='time'){
    const tsf = vs.fvalues.map((v,k)=> cmul(v, cexpj(vs.tshift*TAU*((k+vs.fshift)%N)/N)));
    let res = ifftPad(scaleArr(tsf, nsamps), nsamps)
                .map((v,n)=> cmul(v, cexpj(vs.fshift*TAU*n/nsamps)));
    if(vs.input_mode==='freq') res = scaleArr(res, 1/N);
    const xax = linspace(vs.tshift, vs.tshift+N, nsamps);
    data.time.ideal = { x:xax, mag:absArr(res), ph:angArr(res) };
    data.time.magYlim = [0, 1.2*Math.max(maxAbs(res),1e-9)];
  }

  // --- smooth "ideal" curve for the freq plots (only when iq_mode == freq) ---
  if(vs.iq_mode==='freq'){
    const fst = vs.tvalues.map((v,k)=> cmul(v, cexpj(-vs.fshift*TAU*((k+vs.tshift)%N)/N)));
    let res = fftPad(scaleArr(fst, nsamps), nsamps).map(v=> cscale(v, 1/nsamps))
                .map((v,n)=> cmul(v, cexpj(-vs.tshift*TAU*n/nsamps)));
    if(vs.input_mode==='time') res = scaleArr(res, 1/N);
    const xax = linspace(vs.fshift, vs.fshift+N, nsamps);
    data.freq.ideal = { x:xax, mag:absArr(res), ph:angArr(res) };
    data.freq.magYlim = [0, 1.2*Math.max(maxAbs(res),1e-9)];
  }

  // --- IQ ideal curve + limits ---
  {
    let res;
    if(vs.iq_mode==='time'){
      const base = (vs.input_mode==='time') ? vs.fvalues : scaleArr(vs.fvalues, 1/N);
      res = ifftPad(scaleArr(base, nsamps), nsamps)
              .map((v,n)=> cmul(v, cexpj(vs.fshift*TAU*n/nsamps)));
    } else {
      const base = (vs.input_mode==='time') ? scaleArr(vs.tvalues, 1/N) : vs.tvalues;
      res = fftPad(scaleArr(base, nsamps), nsamps).map(v=> cscale(v, 1/nsamps))
              .map((v,n)=> cmul(v, cexpj(-vs.tshift*TAU*n/nsamps)));
    }
    data.iq.ideal = res;
    data.iq.lim = Math.max(maxAbs(res), maxAbs(data.iq.points), 1e-9) * 1.2;
    // Also fit the tip-to-tail phasor chain: its intermediate partial sums can
    // bulge past 1.2*tip (esp. the high-freq phasors at the chain's end). If the
    // limit doesn't contain them the green chain gets clipped at the plot edge,
    // which — because only the last of each frame's 1-8 steps is drawn — flashes
    // in and out. Expanding the limit keeps the whole chain on-screen and steady.
    let phasors;
    if(vs.iq_mode==='time') phasors = (vs.input_mode==='time') ? vs.fvalues : scaleArr(vs.fvalues, 1/N);
    else                    phasors = (vs.input_mode==='freq') ? vs.tvalues : scaleArr(vs.tvalues, 1/N);
    data.iq.lim = Math.max(data.iq.lim, chainExtent(phasors) * 1.03);
  }

  PD = data;
}

// ---------------------------------------------------------------------------
// Animation state
// ---------------------------------------------------------------------------
const anim = {
  running:false, phaseMag:0, sign:1, currentPhasors:[],
  chainX:[0], chainY:[0],
  histX:[], histY:[], histTx:[], histMag:[], histPh:[], maxHist:15,
  lastTs:0, acc:0
};

// Spotlight-highlight state: array index k of the impulse (and its paired phasor)
// currently pointed at, or null when nothing is selected. When set, non-selected
// phasors/impulses render pale grey (DIM) while index k stays full colour.
let highlightIndex = null;
const DIM = '#cfcfcf';

function clearHistory(){
  anim.histX.length=0; anim.histY.length=0; anim.histTx.length=0;
  anim.histMag.length=0; anim.histPh.length=0;
}

function armAnimation(){
  const N = vs.tvalues.length;
  anim.maxHist = Math.max(1, Math.round(vs.nsamps*0.05));
  if(!N){ anim.currentPhasors=[]; return; }
  if(vs.iq_mode==='time'){
    anim.currentPhasors = (vs.input_mode==='time') ? vs.fvalues.slice() : scaleArr(vs.fvalues, 1/N);
  } else {
    anim.currentPhasors = (vs.input_mode==='freq') ? vs.tvalues.slice() : scaleArr(vs.tvalues, 1/N);
  }
  anim.sign = (vs.iq_mode==='freq') ? -1 : 1;
  anim.phaseMag = 0;
  anim.chainX=[0]; anim.chainY=[0];
  clearHistory();
}

function step(){
  const N = anim.currentPhasors.length;
  if(!N) return;
  const dphase = TAU/vs.nsamps;
  const signed = anim.sign * anim.phaseMag;
  const iqTime = (vs.iq_mode==='time');
  const poff = iqTime ? (TAU*vs.tshift/N) : (-TAU*vs.fshift/N);

  let x=0, y=0;
  const xs=[0], ys=[0];
  for(let k=0;k<N;k++){
    const m = iqTime ? (k+vs.fshift) : (k+vs.tshift);
    const w = cexpj(m*(signed+poff));
    const p = cmul(anim.currentPhasors[k], w);
    x += p.re; y += p.im;
    xs.push(x); ys.push(y);
  }
  anim.chainX = xs; anim.chainY = ys;

  const sampleIndex = anim.phaseMag * N / TAU + (iqTime ? vs.tshift : vs.fshift);
  // ring-buffer history (newest at end; order irrelevant for scatter)
  anim.histX.push(x);  anim.histY.push(y);
  anim.histTx.push(sampleIndex);
  anim.histMag.push(Math.hypot(x,y));
  anim.histPh.push(Math.atan2(y,x));
  while(anim.histX.length > anim.maxHist){
    anim.histX.shift(); anim.histY.shift(); anim.histTx.shift();
    anim.histMag.shift(); anim.histPh.shift();
  }

  anim.phaseMag = (anim.phaseMag + dphase) % TAU;
}

function frame(ts){
  if(!anim.running){ return; }
  if(!anim.lastTs) anim.lastTs = ts;
  let dt = ts - anim.lastTs; anim.lastTs = ts;
  anim.acc += dt;
  // Cap the catch-up. A long (laggy) frame must not fast-forward many sample
  // steps into a single render: the high-frequency phasors at the chain's end
  // rotate the most per step, so a big multi-step jump makes them alias and
  // whip outward ("jump out from the end"). Clamp the backlog and the steps per
  // frame so the phase advances at most ~2 samples per rendered frame; normal
  // playback (0-2 steps/frame) is unchanged, only lag spikes are tamed.
  if(anim.acc > vs.refresh * 2) anim.acc = vs.refresh * 2;
  let steps = 0;
  while(anim.acc >= vs.refresh && steps < 2){ step(); anim.acc -= vs.refresh; steps++; }
  if(steps===0 && anim.phaseMag===0 && anim.chainX.length===1){ step(); } // draw first frame promptly
  render();
  requestAnimationFrame(frame);
}
function startAnim(){
  if(anim.running || vs.tvalues.length===0) return;
  anim.running = true; anim.lastTs = 0; anim.acc = 0;
  requestAnimationFrame(frame);
}
function stopAnim(){ anim.running = false; }

// ---------------------------------------------------------------------------
// 4. Renderer (replaces matplotlib axes)
// ---------------------------------------------------------------------------
const canvas = document.getElementById('vs-canvas');
// Second canvas holds the strip charts in the stacked (mobile) layout so the IQ
// Mode selector can sit in the DOM gap between the IQ plot and the charts. On
// desktop it is display:none and unused (everything renders on the main canvas).
const canvas2 = document.getElementById('vs-canvas2');
const ctx1 = canvas.getContext('2d');
const ctx2 = canvas2.getContext('2d');
// `ctx` is the *current* draw target; render() points it at ctx1 (IQ / everything
// on desktop) or ctx2 (strip charts, stacked only). All draw helpers use it.
let ctx = ctx1;

// shared plot margins (used for layout + mapping) — mode-independent
const MARGIN = {left:44, right:12, top:22, bottom:32};
const PAD = 8;
const sideGap = 12;

// Layout is recomputed per screen size (see computeLayout). The canvas logical
// size (CW/CH) and all plot rectangles are reassignable: on wide screens they
// reproduce today's exact 1080x558 desktop layout; on narrow screens they reflow
// to a stacked/portrait arrangement. Variable names are unchanged so the render
// functions below keep referencing them directly. CW2/CH2 size the second
// (strip-chart) canvas in stacked mode.
let CW, CH, CW2, CH2, areaW, areaH, colW, iqRect, iqInnerW, iqInnerH,
    iqSquare, iqSqTop, sideH, iqDrawRect, rects;
let layoutMode = 'wide';

// Which layout to use, keyed to the same 700px breakpoint as the CSS @media rule
// so the canvas reflow and the control-bar reflow flip together.
function currentMode(){
  return window.matchMedia('(max-width:700px)').matches ? 'stacked' : 'wide';
}
const clampw = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function computeLayout(mode){
  layoutMode = mode;
  if(mode !== 'stacked'){
    // ---- wide (desktop): identical to the original fixed layout ----
    // IQ spans the middle 2 columns at full height; the left (time) and right
    // (freq) mag/phase plots are sized so each stacked pair spans exactly the IQ
    // plot's square drawing region, vertically aligned with it.
    CW = 1080; CH = 558;
    areaW = CW - 2*PAD; areaH = CH - 2*PAD;
    colW = areaW/4;
    iqRect = {x:PAD+colW, y:PAD, w:colW*2, h:areaH};
    iqInnerW = iqRect.w - MARGIN.left - MARGIN.right;
    iqInnerH = iqRect.h - MARGIN.top - MARGIN.bottom;
    iqSquare = Math.min(iqInnerW, iqInnerH);
    iqSqTop  = iqRect.y + MARGIN.top + (iqInnerH - iqSquare)/2;
    sideH    = (iqSquare - sideGap)/2;
    iqDrawRect = {x:PAD+colW, y:iqSqTop, w:colW*2, h:iqSquare};
    rects = {
      time_mag:   {x:PAD,        y:iqSqTop,               w:colW, h:sideH},
      time_phase: {x:PAD,        y:iqSqTop+sideH+sideGap, w:colW, h:sideH},
      iq:         iqDrawRect,
      freq_mag:   {x:PAD+colW*3, y:iqSqTop,               w:colW, h:sideH},
      freq_phase: {x:PAD+colW*3, y:iqSqTop+sideH+sideGap, w:colW, h:sideH},
    };
    return;
  }
  // ---- stacked (phone/portrait): IQ square on canvas 1, 2x2 strip grid on
  // canvas 2, with the IQ Mode selector living in the DOM gap between them. ----
  const wrapW = (canvas.parentElement && canvas.parentElement.clientWidth) || 360;
  CW = Math.round(clampw(wrapW, 340, 560));   // logical ~= display px -> legible fonts
  CW2 = CW;
  const iqRectW = CW - 2*PAD;
  iqInnerW = iqRectW - MARGIN.left - MARGIN.right;
  iqSquare = iqInnerW;                          // equal-aspect square fills the width
  iqInnerH = iqSquare;
  iqSqTop  = PAD + MARGIN.top;                  // no vertical centering offset
  iqDrawRect = {x:PAD, y:PAD, w:iqRectW, h:iqSquare + MARGIN.top + MARGIN.bottom};
  CH = iqDrawRect.y + iqDrawRect.h + PAD;       // canvas 1: just the IQ plot
  // 2x2 grid of strip charts on canvas 2 (its own coordinate space, y from PAD)
  const colGap = 10, rowGap = 14;
  const gridColW = (CW - 2*PAD - colGap)/2;
  const rowH = Math.round(clampw(gridColW*0.62, 130, 180));
  const leftX = PAD, rightX = PAD + gridColW + colGap;
  const row1Y = PAD, row2Y = PAD + rowH + rowGap;
  rects = {
    time_mag:   {x:leftX,  y:row1Y, w:gridColW, h:rowH},
    freq_mag:   {x:rightX, y:row1Y, w:gridColW, h:rowH},
    iq:         iqDrawRect,
    time_phase: {x:leftX,  y:row2Y, w:gridColW, h:rowH},
    freq_phase: {x:rightX, y:row2Y, w:gridColW, h:rowH},
  };
  CH2 = row2Y + rowH + PAD;                      // canvas 2: the strip grid
  // areaW/areaH/colW/sideH are read only by the wide path; left stale here.
}

// Move the single IQ Mode selector between its desktop home (centre of the shift
// bar, under the IQ plot) and the mobile slot between the two canvases. One live
// element, so its radios + listeners move with it.
let _iqHome = null, _iqAnchor = null;
function placeIqMode(mode){
  const col = document.querySelector('.vs-iqmode-col');
  const mid = document.getElementById('vs-midbar');
  if(!col || !mid) return;
  if(!_iqHome){ _iqHome = col.parentElement; _iqAnchor = col.nextElementSibling; }
  if(mode === 'stacked'){
    if(col.parentElement !== mid) mid.appendChild(col);
  } else if(col.parentElement !== _iqHome){
    _iqHome.insertBefore(col, _iqAnchor);
  }
}

// Move the input help text + input row to the very top of the page on mobile
// (before the IQ plot), and back to their normal spot in the controls on desktop.
let _inpHome = null, _inpAnchor = null;
function placeInputBlock(mode){
  const desc = document.getElementById('vs-input-desc');
  const inp  = document.getElementById('vs-input');
  const row  = inp && inp.closest('.vs-row');
  const up   = document.getElementById('vs-upload-block');
  const root = document.querySelector('.vs-root');
  if(!desc || !row || !root) return;
  const items = [desc, row, up].filter(Boolean);   // move help text, input row, upload block together
  if(!_inpHome){ _inpHome = desc.parentElement; _inpAnchor = items[items.length-1].nextElementSibling; }
  if(mode === 'stacked'){
    // Insert desc, row, up in order just after the title. Use a moving cursor
    // (not a fixed anchor) so repeated calls are idempotent — a fixed anchor that
    // is itself one of the moved items would reshuffle them on re-invocation.
    const title = document.getElementById('vs-title');
    let after = (title && title.parentElement===root) ? title : null;
    for(const it of items){
      const ref = after ? after.nextSibling : root.firstChild;
      if(it !== ref) root.insertBefore(it, ref);
      after = it;
    }
  } else if(desc.parentElement !== _inpHome){
    for(const it of items) _inpHome.insertBefore(it, _inpAnchor);   // restore to their old spot
  }
}

// (Re)size the backing store for crisp HiDPI rendering. Idempotent across
// relayouts: reset the transform before scaling so repeated calls don't compound.
function applyCanvasSize(){
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(CW*dpr);
  canvas.height = Math.round(CH*dpr);
  canvas.style.width = '100%';
  ctx1.setTransform(dpr, 0, 0, dpr, 0, 0);
  if(layoutMode === 'stacked'){
    canvas2.width  = Math.round(CW2*dpr);
    canvas2.height = Math.round(CH2*dpr);
    canvas2.style.width = '100%';
    ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

computeLayout(currentMode());
applyCanvasSize();
placeIqMode(layoutMode);
placeInputBlock(layoutMode);

function niceTicks(min, max, count){
  const span = max - min;
  if(span <= 0 || !isFinite(span)) return [min];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  let stepv;
  if(norm < 1.5) stepv = 1; else if(norm < 3) stepv = 2; else if(norm < 7) stepv = 5; else stepv = 10;
  stepv *= mag;
  const start = Math.ceil(min/stepv)*stepv;
  const out = [];
  for(let v=start; v<=max+stepv*1e-6; v+=stepv) out.push(Math.abs(v)<stepv*1e-6?0:v);
  return out;
}
function fmtTick(v){
  if(v===0) return '0';
  if(Math.abs(v)>=1000 || Math.abs(v)<0.01) return v.toExponential(0);
  return (Math.round(v*100)/100).toString();
}

function makeMapper(rect, xlim, ylim, opts){
  opts = opts||{};
  const m = MARGIN;
  let ix = rect.x + m.left, iy = rect.y + m.top;
  let iw = rect.w - m.left - m.right, ih = rect.h - m.top - m.bottom;
  let sx = iw/(xlim[1]-xlim[0]);
  let sy = ih/(ylim[1]-ylim[0]);
  if(opts.equal){
    const s = Math.min(sx, sy);
    const usedW = s*(xlim[1]-xlim[0]), usedH = s*(ylim[1]-ylim[0]);
    ix += (iw-usedW)/2; iy += (ih-usedH)/2; iw = usedW; ih = usedH; sx = sy = s;
  }
  const X = v => ix + (v-xlim[0])*sx;
  const Y = v => iy + ih - (v-ylim[0])*sy;
  return {rect, ix, iy, iw, ih, X, Y, xlim, ylim, title:opts.title, xlabel:opts.xlabel, ylabel:opts.ylabel};
}

function drawAxes(mp){
  ctx.save();
  // grid
  ctx.strokeStyle = '#e2e2e2'; ctx.lineWidth = 1;
  ctx.fillStyle = '#666'; ctx.font = '12px sans-serif';
  const xt = niceTicks(mp.xlim[0], mp.xlim[1], 5);
  const yt = niceTicks(mp.ylim[0], mp.ylim[1], 5);
  ctx.textAlign='center'; ctx.textBaseline='top';
  for(const t of xt){
    const px = mp.X(t);
    if(px < mp.ix-0.5 || px > mp.ix+mp.iw+0.5) continue;
    ctx.beginPath(); ctx.moveTo(px, mp.iy); ctx.lineTo(px, mp.iy+mp.ih); ctx.stroke();
    ctx.fillText(fmtTick(t), px, mp.iy+mp.ih+3);
  }
  ctx.textAlign='right'; ctx.textBaseline='middle';
  for(const t of yt){
    const py = mp.Y(t);
    if(py < mp.iy-0.5 || py > mp.iy+mp.ih+0.5) continue;
    ctx.beginPath(); ctx.moveTo(mp.ix, py); ctx.lineTo(mp.ix+mp.iw, py); ctx.stroke();
    ctx.fillText(fmtTick(t), mp.ix-4, py);
  }
  // border
  ctx.strokeStyle = '#888'; ctx.lineWidth = 1;
  ctx.strokeRect(mp.ix, mp.iy, mp.iw, mp.ih);
  // title
  if(mp.title){
    ctx.fillStyle = '#222'; ctx.font = 'bold 14.4px sans-serif';
    ctx.textAlign='center'; ctx.textBaseline='bottom';
    ctx.fillText(mp.title, mp.ix+mp.iw/2, mp.iy-4);
  }
  // x label
  if(mp.xlabel){
    ctx.fillStyle='#444'; ctx.font='12px sans-serif';
    ctx.textAlign='center'; ctx.textBaseline='bottom';
    ctx.fillText(mp.xlabel, mp.ix+mp.iw/2, mp.iy+mp.ih+28);
  }
  // y label
  if(mp.ylabel){
    ctx.save();
    ctx.translate(mp.ix-34, mp.iy+mp.ih/2);   // anchored to the y-axis, not the outer rect
    ctx.rotate(-Math.PI/2);
    ctx.fillStyle='#444'; ctx.font='12px sans-serif';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(mp.ylabel, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}
function clip(mp){ ctx.save(); ctx.beginPath(); ctx.rect(mp.ix, mp.iy, mp.iw, mp.ih); ctx.clip(); }
function unclip(){ ctx.restore(); }

function drawMarkers(mp, xs, ys, color, size){
  ctx.fillStyle = color;
  for(let i=0;i<xs.length;i++){
    const px=mp.X(xs[i]), py=mp.Y(ys[i]);
    ctx.beginPath(); ctx.arc(px, py, size/2, 0, TAU); ctx.fill();
  }
}
function drawStems(mp, xs, ys, color, mColor, mSize, hi){
  const y0 = mp.Y(0);
  // stems: when spotlighting (hi != null) dim all but index hi, which thickens
  for(let i=0;i<xs.length;i++){
    ctx.strokeStyle = (hi==null || i===hi) ? color : DIM;
    ctx.lineWidth = (i===hi) ? 3.8 : 2.8;          // 2x wider stems (selected wider still)
    const px=mp.X(xs[i]);
    ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, mp.Y(ys[i])); ctx.stroke();
  }
  // baseline
  ctx.strokeStyle = '#000'; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(mp.ix, y0); ctx.lineTo(mp.ix+mp.iw, y0); ctx.stroke();
  // stem tip dots (same size as plain markers); selected drawn last, enlarged
  if(hi==null){
    drawMarkers(mp, xs, ys, mColor, mSize);
  } else {
    for(let i=0;i<xs.length;i++){ if(i!==hi) drawMarkers(mp, [xs[i]], [ys[i]], DIM, mSize); }
    if(hi>=0 && hi<xs.length) drawMarkers(mp, [xs[hi]], [ys[hi]], mColor, mSize*1.5);
  }
}
function drawLine(mp, xs, ys, color, w){
  ctx.strokeStyle = color; ctx.lineWidth = w;
  ctx.beginPath();
  for(let i=0;i<xs.length;i++){
    const px=mp.X(xs[i]), py=mp.Y(ys[i]);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }
  ctx.stroke();
}
function drawDots(mp, xs, ys, color, r){
  ctx.fillStyle = color;
  for(let i=0;i<xs.length;i++){
    ctx.beginPath(); ctx.arc(mp.X(xs[i]), mp.Y(ys[i]), r, 0, TAU); ctx.fill();
  }
}

function drawStripPlot(name, side, isMag){
  const rect = rects[name];
  const d = PD[side];
  const ylim = isMag ? d.magYlim : [-Math.PI, Math.PI];
  // On the narrow stacked (phone) layout, shorten "Magnitude" so the title fits
  // the small strip charts; desktop keeps the full word.
  const magWord = layoutMode==='stacked' ? 'Mag' : 'Magnitude';
  const title = (side==='time'?'Time':'Freq') + '-Domain: ' + (isMag?magWord:'Phase');
  const xlabel = (side==='time' ? 'Time Index n' : 'Frequency Index k');
  const ylabel = isMag ? 'Magnitude' : 'Radians';
  const mp = makeMapper(rect, d.xlim, ylim, {title, xlabel, ylabel});
  mappers[name] = mp;
  drawAxes(mp);
  clip(mp);
  const yvals = isMag ? d.mag : d.ph;
  // ideal smooth curve (red) under markers
  if(d.ideal){ drawLine(mp, d.ideal.x, isMag?d.ideal.mag:d.ideal.ph, '#d62728', 0.8); }
  // stems (conjugate/source domain — the interactive one). Spotlight index passed
  // only here; the stem domain always matches the phasor source, so highlightIndex
  // (a source-domain array index) selects the right impulse.
  if(d.stems){ drawStems(mp, d.index, yvals, vs.phasor_color, d.marker[0], d.marker[1], highlightIndex); }
  else { drawMarkers(mp, d.index, yvals, d.marker[0], d.marker[1]); }
  unclip();
}

function drawIqPlot(){
  const rect = rects.iq;
  const lim = PD ? PD.iq.lim : 1;
  const title = (vs.iq_mode==='time' ? 'Time' : 'Freq') + ' Domain - IQ Plot';
  const mp = makeMapper(rect, [-lim, lim], [-lim, lim], {title, xlabel:'I', ylabel:'Q', equal:true});
  mappers.iq = mp;
  drawAxes(mp);
  if(!PD) return;
  clip(mp);
  // ideal reconstruction (thin red)
  const ideal = PD.iq.ideal;
  drawLine(mp, ideal.map(v=>v.re), ideal.map(v=>v.im), '#d62728', 0.8);
  // input/result markers
  const pts = PD.iq.points;
  drawMarkers(mp, pts.map(v=>v.re), pts.map(v=>v.im), PD.iq.marker[0], PD.iq.marker[1]);
  unclip();
  // copyright, small and unobtrusive, along the bottom of the IQ plot. In the GIF
  // export it is drawn solid (full opacity) so the text stays crisp and legible
  // after colour quantization rather than a faint, dithered grey.
  ctx.save();
  ctx.fillStyle = gifExporting ? 'rgb(60,60,60)' : 'rgba(90,90,90,0.55)';
  ctx.font = (gifExporting ? 'bold 11px' : '11px') + ' sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('© The DSP Coach 2026  VectorSpin', mp.ix + mp.iw/2, mp.iy + mp.ih - 5);
  ctx.restore();
}

function roundRectPath(x, y, w, h, r){
  r = Math.min(r, w/2, h/2);
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y,   x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x,   y+h, r);
  ctx.arcTo(x,   y+h, x,   y,   r);
  ctx.arcTo(x,   y,   x+w, y,   r);
  ctx.closePath();
}

// Subtle group box around the three plots currently being transformed: the IQ
// plot plus its source-domain mag/phase pair (left column in Time mode, right
// column in Freq mode). Its edge reaches the red DFT arrow, so the box reads as
// "these three -> (arrow) -> the other domain". Drawn behind the plots as a very
// light shade + faint border; excludes the shift controls below the plots.
function drawGroupBox(){
  if(layoutMode !== 'wide') return;   // horizontal source->target aid; N/A in stacked grid
  const m = MARGIN;
  // IQ inner square horizontal extent (mirrors makeMapper's equal-aspect math)
  const iqInW = iqDrawRect.w - m.left - m.right;
  const iqInH = iqDrawRect.h - m.top - m.bottom;
  const iqSide = Math.min(iqInW, iqInH);
  const iqIx    = iqDrawRect.x + m.left + (iqInW - iqSide)/2;
  const iqRight = iqIx + iqSide;
  // vertical span: top of the mag/IQ plots to bottom of the phase plots, padded
  // to clear titles and x-axis labels without crowding them
  const top = iqSqTop - 6;
  const bot = iqSqTop + iqSquare + 6;
  let bx1, bx2;
  if(vs.iq_mode === 'freq'){
    // source = right (freq) column; arrow is left of IQ (points toward time)
    const tmRight = rects.time_mag.x + rects.time_mag.w - m.right;
    const c = (tmRight + (iqIx - 42)) / 2;   // arrow centre (see drawArrow)
    bx1 = c + 18;                            // arrow's IQ-side end (LEN/2)
    bx2 = CW - PAD + 2;                      // enclose the freq column
  } else {
    // source = left (time) column; arrow is right of IQ (points toward freq)
    bx1 = PAD - 2;                           // enclose the time column
    bx2 = iqRight + 14;                      // GAP: meet the arrow's tail
  }
  ctx.save();
  ctx.beginPath();
  roundRectPath(bx1, top, bx2 - bx1, bot - top, 8);
  ctx.fillStyle   = 'rgba(31,79,216,0.03)';   // even lighter blue wash
  ctx.fill();
  ctx.lineWidth   = 1;
  ctx.strokeStyle = 'rgba(96,110,140,0.45)';  // blue-gray border
  ctx.stroke();
  ctx.restore();
}

function drawArrow(){
  // red DFT-direction arrow between IQ and the target column; '/N' label when dividing.
  // ~2x larger than before, kept clear of the IQ plot axis with a fixed gap.
  if(layoutMode !== 'wide') return;   // horizontal DFT arrow; N/A in stacked grid
  const iq = mappers.iq; if(!iq) return;
  ctx.save();
  ctx.strokeStyle = '#d62728'; ctx.fillStyle = '#d62728'; ctx.lineWidth = 8;
  const cy = iq.iy + iq.ih*0.5;
  const GAP = 14;   // clear space between arrow and the IQ axis border
  const LEN = 36;   // shaft length (bounded so it can't reach the side plots)
  const hs  = 15;   // arrowhead size
  let x1,x2,dir,labelX;
  if(vs.iq_mode==='freq'){        // arrow points left (toward time plots)
    // centre it in the gap between the time plots and the IQ y-axis, clear of the Q label
    const tm = mappers.time_mag;
    const leftBound  = tm ? tm.ix + tm.iw : iq.ix - 90;
    const rightBound = iq.ix - 42;   // room for the Q label + y-tick numbers
    const c = (leftBound + rightBound) / 2;
    x1 = c + LEN/2; x2 = c - LEN/2; dir = -1;
  } else {                        // arrow points right (toward freq plots)
    x1 = iq.ix+iq.iw + GAP; x2 = x1 + LEN; dir = 1;
  }
  labelX = (x1 + x2) / 2;
  ctx.lineCap = 'butt';
  // shaft stops at the arrowhead base so the tip stays a sharp point
  const base = x2 - dir*hs;
  ctx.beginPath(); ctx.moveTo(x1, cy); ctx.lineTo(base, cy); ctx.stroke();
  // arrowhead: sharp filled triangle with mitered joins
  ctx.lineJoin = 'miter';
  ctx.beginPath();
  ctx.moveTo(x2, cy);
  ctx.lineTo(base, cy - hs*0.7);
  ctx.lineTo(base, cy + hs*0.7);
  ctx.closePath(); ctx.fill();
  // /N label
  const showN = (vs.iq_mode==='freq' && vs.input_mode==='time') ||
                (vs.iq_mode==='time' && vs.input_mode==='freq');
  if(showN){
    ctx.font = 'bold 16.8px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='top';
    ctx.fillText('/N', labelX, cy + hs*0.7 + 6);
  }
  ctx.restore();
}

function render(){
  const twoCanvas = (layoutMode === 'stacked');
  // strip charts go on canvas 2 in stacked mode, else share the main canvas
  const stripCtx = twoCanvas ? ctx2 : ctx1;

  // clear main canvas (and the strip canvas when split)
  ctx = ctx1;
  ctx.clearRect(0,0,CW,CH); ctx.fillStyle = '#fff'; ctx.fillRect(0,0,CW,CH);
  if(twoCanvas){
    ctx = ctx2;
    ctx.clearRect(0,0,CW2,CH2); ctx.fillStyle = '#fff'; ctx.fillRect(0,0,CW2,CH2);
  }

  ctx = ctx1;
  drawGroupBox();   // subtle shade behind the plots being transformed (wide only)

  if(!PD){
    // empty axes
    const magWord = layoutMode==='stacked' ? 'Mag' : 'Magnitude';
    ctx = stripCtx;
    mappers.time_mag  = makeMapper(rects.time_mag,  [0,1],[0,1], {title:'Time-Domain: '+magWord});
    mappers.time_phase= makeMapper(rects.time_phase,[0,1],[-Math.PI,Math.PI], {title:'Time-Domain: Phase', xlabel:'Time Index n', ylabel:'Radians'});
    mappers.freq_mag  = makeMapper(rects.freq_mag,  [0,1],[0,1], {title:'Freq-Domain: '+magWord});
    mappers.freq_phase= makeMapper(rects.freq_phase,[0,1],[-Math.PI,Math.PI], {title:'Freq-Domain: Phase', xlabel:'Frequency Index k', ylabel:'Radians'});
    drawAxes(mappers.time_mag); drawAxes(mappers.time_phase);
    drawAxes(mappers.freq_mag); drawAxes(mappers.freq_phase);
    ctx = ctx1;
    drawIqPlot();
    drawArrow();
    return;
  }

  // ---- strip charts (+ their history dots) on stripCtx ----
  ctx = stripCtx;
  drawStripPlot('time_mag',  'time', true);
  drawStripPlot('time_phase','time', false);
  drawStripPlot('freq_mag',  'freq', true);
  drawStripPlot('freq_phase','freq', false);
  const magPlot = mappers[PD.historyTarget==='time' ? 'time_mag'  : 'freq_mag'];
  const phPlot  = mappers[PD.historyTarget==='time' ? 'time_phase': 'freq_phase'];
  if(magPlot){ clip(magPlot); drawDots(magPlot, anim.histTx, anim.histMag, vs.history_color, 1.4); unclip(); }
  if(phPlot){  clip(phPlot);  drawDots(phPlot,  anim.histTx, anim.histPh,  vs.history_color, 1.4); unclip(); }

  // ---- IQ plot (+ dynamic overlays) on the main canvas ----
  ctx = ctx1;
  drawIqPlot();
  // IQ phasor chain (green) + tip trace (blue)
  const iq = mappers.iq;
  clip(iq);
  // Colour the phasor-tip dots by the domain they came from: the impulses driving
  // the phasors live in the conjugate domain (freq when the IQ plot shows Time, and
  // vice-versa), so tips are red in Time mode / blue in Freq mode. Origin stays green.
  const tipColor = (vs.iq_mode==='time') ? vs.fmarker[0] : vs.tmarker[0];
  const cX = anim.chainX, cY = anim.chainY;
  if(highlightIndex==null){
    drawLine(iq, cX, cY, vs.phasor_color, 2.8);                    // phasor shafts match stem width
    drawDots(iq, cX.slice(1), cY.slice(1), tipColor, 3.15);        // tips: source-domain colour
  } else {
    // Spotlight: dim every phasor segment/tip except the selected index k, drawn last (on top).
    const k = highlightIndex;
    for(let s=0;s<cX.length-1;s++){
      if(s===k) continue;
      drawLine(iq, [cX[s],cX[s+1]], [cY[s],cY[s+1]], DIM, 2.8);
      drawDots(iq, [cX[s+1]], [cY[s+1]], DIM, 3.15);
    }
    if(k>=0 && k<cX.length-1){
      drawLine(iq, [cX[k],cX[k+1]], [cY[k],cY[k+1]], vs.phasor_color, 3.8);
      drawDots(iq, [cX[k+1]], [cY[k+1]], tipColor, 4.2);
    }
  }
  if(anim.chainX.length) drawDots(iq, [cX[0]], [cY[0]], vs.phasor_color, 3.15);  // origin stays green
  drawDots(iq, anim.histX, anim.histY, vs.history_color, 1.4);
  unclip();

  drawArrow();
}

// ---------------------------------------------------------------------------
// 4b. Animated GIF export (self-contained: median-cut palette + LZW encoder)
// ---------------------------------------------------------------------------
function gifShort(a, v){ a.push(v & 0xff, (v>>8) & 0xff); }
function gifStr(a, s){ for(let i=0;i<s.length;i++) a.push(s.charCodeAt(i)); }

// Add one frame's pixels (sampled every sampleStride) to a colour histogram.
// Called incrementally so we never hold every frame's pixels in memory at once.
function buildHistogram(hist, data, sampleStride){
  for(let i=0;i<data.length;i+=4*sampleStride){
    const key = ((data[i]>>3)<<10)|((data[i+1]>>3)<<5)|(data[i+2]>>3);
    let e = hist.get(key);
    if(!e){ e={c:0,r:0,g:0,b:0}; hist.set(key,e); }
    e.c++; e.r+=data[i]; e.g+=data[i+1]; e.b+=data[i+2];
  }
}

// Median-cut a colour histogram to a 256-entry palette, plus a cached
// nearest-colour mapper. Colours are bucketed in 15-bit (5-5-5) space for speed.
function paletteFromHistogram(hist){
  const chanAvg = (e,ch)=> ch===0? e.r/e.c : ch===1? e.g/e.c : e.b/e.c;
  const ranges = box=>{
    let rmn=255,rmx=0,gmn=255,gmx=0,bmn=255,bmx=0;
    for(const e of box){ const r=e.r/e.c,g=e.g/e.c,b=e.b/e.c;
      if(r<rmn)rmn=r; if(r>rmx)rmx=r; if(g<gmn)gmn=g; if(g>gmx)gmx=g; if(b<bmn)bmn=b; if(b>bmx)bmx=b; }
    return [rmx-rmn, gmx-gmn, bmx-bmn];
  };
  let boxes = [Array.from(hist.values())];
  while(boxes.length < 256){
    let bi=-1, best=-1, ch=0;
    for(let k=0;k<boxes.length;k++){
      if(boxes[k].length < 2) continue;
      const rg = ranges(boxes[k]); const mx = Math.max(rg[0],rg[1],rg[2]);
      if(mx > best){ best=mx; bi=k; ch = (rg[0]>=rg[1]&&rg[0]>=rg[2])?0:(rg[1]>=rg[2]?1:2); }
    }
    if(bi < 0) break;
    const box = boxes[bi]; box.sort((a,b)=> chanAvg(a,ch)-chanAvg(b,ch));
    const total = box.reduce((s,e)=>s+e.c,0);
    let acc=0, sp=0; for(; sp<box.length-1; sp++){ acc+=box[sp].c; if(acc*2>=total) break; }
    sp = Math.min(sp, box.length - 2);   // guarantee both halves are non-empty
    boxes.splice(bi,1, box.slice(0,sp+1), box.slice(sp+1));
  }
  const palette = boxes.map(box=>{
    let c=0,r=0,g=0,b=0; for(const e of box){ c+=e.c; r+=e.r; g+=e.g; b+=e.b; }
    return [Math.round(r/c), Math.round(g/c), Math.round(b/c)];
  });
  while(palette.length < 256) palette.push([0,0,0]);
  const cache = new Map();
  function nearest(r,g,b){
    const key = ((r>>3)<<10)|((g>>3)<<5)|(b>>3);
    let idx = cache.get(key);
    if(idx===undefined){
      let bd=1e12, bp=0;
      for(let p=0;p<256;p++){
        const pr=palette[p][0]-r, pg=palette[p][1]-g, pb=palette[p][2]-b;
        const d=pr*pr+pg*pg+pb*pb; if(d<bd){ bd=d; bp=p; }
      }
      idx=bp; cache.set(key,idx);
    }
    return idx;
  }
  return {palette, nearest};
}

// Variable-width LZW (GIF flavour) → array of bytes. The code-width increase is
// applied at emit time (after shifting out the code, testing the running code
// against the current max) to match giflib/standard GIF decoders exactly.
function lzwEncode(indices, minCodeSize){
  const clearCode = 1<<minCodeSize, eoiCode = clearCode+1;
  let codeSize = minCodeSize+1, maxCode = 1<<codeSize;
  let next = eoiCode+1, dict = new Map();
  const bytes=[]; let cur=0, nbits=0;
  function emit(code){
    cur |= code<<nbits; nbits += codeSize;
    while(nbits>=8){ bytes.push(cur&0xff); cur>>=8; nbits-=8; }
    if(next >= maxCode && codeSize < 12){ codeSize++; maxCode = 1<<codeSize; }
  }
  emit(clearCode);
  let prefix = indices[0];
  for(let i=1;i<indices.length;i++){
    const c = indices[i], key = prefix*256 + c;
    const found = dict.get(key);
    if(found!==undefined){ prefix = found; }
    else {
      emit(prefix);
      if(next >= 4096){
        emit(clearCode);
        dict = new Map(); next = eoiCode+1; codeSize = minCodeSize+1; maxCode = 1<<codeSize;
      } else {
        dict.set(key, next++);
      }
      prefix = c;
    }
  }
  emit(prefix); emit(eoiCode);
  if(nbits>0) bytes.push(cur&0xff);
  return bytes;
}

function assembleGif(width, height, palette, frames, delayCs){
  const out=[];
  gifStr(out, "GIF89a");
  gifShort(out, width); gifShort(out, height);
  out.push(0xF7, 0x00, 0x00);            // GCT: 256 entries, colour res 8
  for(let i=0;i<256;i++){ const c=palette[i]||[0,0,0]; out.push(c[0],c[1],c[2]); }
  out.push(0x21,0xFF,0x0B); gifStr(out,"NETSCAPE2.0");
  out.push(0x03,0x01); gifShort(out,0); out.push(0x00);   // loop forever
  for(const idx of frames){
    out.push(0x21,0xF9,0x04,0x00); gifShort(out,delayCs); out.push(0x00,0x00);
    out.push(0x2C); gifShort(out,0); gifShort(out,0);
    gifShort(out,width); gifShort(out,height); out.push(0x00);
    const lzw = lzwEncode(idx, 8);
    out.push(8);
    for(let i=0;i<lzw.length;i+=255){
      const n = Math.min(255, lzw.length-i);
      out.push(n);
      for(let j=0;j<n;j++) out.push(lzw[i+j]);
    }
    out.push(0x00);
  }
  out.push(0x3B);
  return out;
}

let gifBusy = false, gifExporting = false;
async function saveGif(){
  if(gifBusy || vs.tvalues.length===0) return;
  gifBusy = true; gifExporting = true;
  const btn = el('vs-savegif'), status = el('vs-gif-status');
  const wasRunning = anim.running; stopAnim(); btn.disabled = true;
  const yield_ = ()=> new Promise(r=>setTimeout(r,0));
  // Always export the wide 1080x558 layout so phone users still get the full
  // shareable GIF (restored in finally). On desktop this is a no-op.
  const prevMode = layoutMode;
  computeLayout('wide'); applyCanvasSize();
  try{
    // Render the vectors at 1.5x resolution for crisp lines/text: point the main
    // canvas at the export size and scale the drawing transform to match, so the
    // whole scene is re-rasterised at 1620x837 (true supersampling, not an upscale).
    const SS = 1.5;
    const SIZE_CEIL = 5*1024*1024, TARGET = Math.round(SIZE_CEIL*0.94);
    // A footer band (attribution URL + DSP Coach logo) is baked below the plots
    // so the shared GIF carries branding. FH is extra logical height for it.
    const logo = document.querySelector('img.vs-logo');
    const footText = (document.querySelector('.vs-foot-text') || {}).textContent || 'VectorSpin  https://dsp-coach.com';
    const FH = 46;
    const GW = Math.round(CW*SS), GH = Math.round((CH+FH)*SS);
    canvas.width = GW; canvas.height = GH;
    ctx1.setTransform(SS,0,0,SS,0,0);

    const drawFooter = ()=>{
      const c = ctx1;
      c.fillStyle = '#fff'; c.fillRect(0, CH, CW, FH);        // footer background
      c.strokeStyle = '#ccc'; c.lineWidth = 1;                // divider line
      c.beginPath(); c.moveTo(PAD, CH+0.5); c.lineTo(CW-PAD, CH+0.5); c.stroke();
      const midY = CH + FH/2;
      if(logo && logo.complete && logo.naturalWidth){         // logo, right-aligned
        const LH = 34, LW = LH * logo.naturalWidth/logo.naturalHeight;
        c.drawImage(logo, CW-PAD-LW, midY-LH/2, LW, LH);
      }
      c.fillStyle = '#444'; c.font = '16px sans-serif';       // attribution, left
      c.textAlign = 'left'; c.textBaseline = 'middle';
      c.fillText(footText, PAD, midY+1);
    };

    const totalSteps = vs.nsamps;
    // Render is deterministic, so we run the animation more than once rather than
    // holding every full-res frame in memory. warmup() replays one revolution so
    // the fading trace loops seamlessly; frameData() renders + reads one frame.
    const warmup = ()=>{ armAnimation(); for(let i=0;i<totalSteps;i++) step(); };
    const frameData = ()=>{ render(); drawFooter(); return ctx1.getImageData(0,0,GW,GH); };
    const indexFrame = (data)=>{
      const idx = new Uint8Array(GW*GH);
      for(let p=0,q=0; q<idx.length; p+=4,q++) idx[q] = nearest(data[p],data[p+1],data[p+2]);
      return idx;
    };

    // ---- pass 1: build the palette from ~80 sampled frames; keep a few to
    //      measure the average encoded frame size for the file-size budget ----
    status.textContent = 'Analyzing…'; await yield_();
    const pStride = Math.max(1, Math.ceil(totalSteps/80));
    const hist = new Map();
    const samples = [];
    warmup();
    let sc = 0;
    for(let f=0; f<totalSteps; f++){
      step();
      if(f % pStride === 0){
        const img = frameData();
        buildHistogram(hist, img.data, 2);
        if(samples.length < 5 && sc % 16 === 0) samples.push(img);
        if(++sc % 10 === 0){ status.textContent = `Analyzing… ${sc}`; await yield_(); }
      }
    }
    var {palette, nearest} = paletteFromHistogram(hist);

    // average compressed bytes per frame (LZW body + ~20 bytes GIF structure)
    let measBytes = 0;
    for(const img of samples) measBytes += lzwEncode(indexFrame(img.data), 8).length + 20;
    const avgFrame = samples.length ? measBytes/samples.length : 70000;
    // pick the frame count: as many as the slider asks, capped to fit the 5 MB
    // ceiling (and a memory backstop). Fewer frames -> larger per-frame delay.
    const budgetFrames = Math.max(2, Math.floor((TARGET - 800) / avgFrame));
    const finalFrames = Math.min(totalSteps, budgetFrames, 300);
    const stride = Math.max(1, Math.round(totalSteps/finalFrames));

    // ---- pass 2: render + index the chosen frames ----
    status.textContent = 'Rendering frames…'; await yield_();
    const indexed = [];
    warmup();
    for(let f=0; f<totalSteps; f++){
      step();
      if(f % stride === 0){
        indexed.push(indexFrame(frameData().data));
        if(indexed.length % 10 === 0){ status.textContent = `Encoding… ${indexed.length}`; await yield_(); }
      }
    }
    let delayCs = Math.max(2, Math.round(vs.refresh*stride/10));

    // ---- assemble; if the estimate overshoots, trim frames to fit the ceiling ----
    status.textContent = 'Building GIF…'; await yield_();
    let frames = indexed;
    let bytes = assembleGif(GW, GH, palette, frames, delayCs);
    while(bytes.length > SIZE_CEIL && frames.length > 2){
      const keep = Math.max(2, Math.floor(frames.length * TARGET / bytes.length));
      const stepF = frames.length / keep, kept = [];
      for(let i=0;i<keep;i++) kept.push(frames[Math.min(frames.length-1, Math.floor(i*stepF))]);
      delayCs = Math.round(delayCs * frames.length / kept.length);
      frames = kept;
      await yield_();
      bytes = assembleGif(GW, GH, palette, frames, delayCs);
    }

    const blob = new Blob([new Uint8Array(bytes)], {type:'image/gif'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download='vectorspin.gif';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(url), 2000);
    status.textContent = `Saved GIF — ${frames.length} frames, ${GW}×${GH}, ${Math.round(bytes.length/1024)} KB`;
  } catch(err){
    status.textContent = 'GIF export failed: ' + err.message;
  } finally {
    btn.disabled = false; gifBusy = false; gifExporting = false;
    // restore the on-screen layout (stacked on phones) before repainting
    computeLayout(prevMode); applyCanvasSize(); placeIqMode(prevMode); _lastKey = '';
    armAnimation();
    if(wasRunning) startAnim(); else render();
  }
}

// ---------------------------------------------------------------------------
// 5. UI wiring
// ---------------------------------------------------------------------------
const el = id => document.getElementById(id);
const inputEl   = el('vs-input');
const inputDesc = el('vs-input-desc');

function refreshPlots(){
  highlightIndex = null;   // data/domain/shift changed -> any selection is stale
  computePlotData();
  armAnimation();
  render();
}

function applyInput(){
  const txt = inputEl.value.trim();
  if(!txt) return;
  let arr;
  try{ arr = parseInput(txt); }
  catch(e){ inputEl.classList.add('vs-error'); return; }
  inputEl.classList.remove('vs-error');
  if(vs.input_mode==='time'){ vs.tvalues = arr; computeFvalues(); }
  else                      { vs.fvalues = arr; computeTvalues(); }
  refreshPlots();
  startAnim();
}

// Format a complex value back into parseInput-compatible text (e.g. 2+3j, 0.5, -1j-free).
function fmtComplex(c){
  const n = x => String(x);
  if(c.im === 0) return n(c.re);
  return n(c.re) + (c.im < 0 ? '-' : '+') + n(Math.abs(c.im)) + 'j';
}

// Parse a two-column (real, imag) CSV into an array of complex samples. Throws with
// a human-readable reason on any format problem (surfaced as "Incorrect format: …").
function parseCsvIQ(text){
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if(!lines.length) throw new Error('the file is empty.');
  // optional header row: skip line 1 if it contains any non-numeric token
  let start = 0;
  const first = lines[0].split(',').map(s => s.trim());
  if(first.some(s => s !== '' && isNaN(Number(s)))) start = 1;
  const arr = [];
  for(let i = start; i < lines.length; i++){
    const cols = lines[i].split(',').map(s => s.trim());
    if(cols.length !== 2)
      throw new Error(`line ${i+1} has ${cols.length} column${cols.length===1?'':'s'}, expected 2 (real, imag).`);
    const re = Number(cols[0]), im = Number(cols[1]);
    if(cols[0] === '' || cols[1] === '' || isNaN(re) || isNaN(im))
      throw new Error(`line ${i+1} has a non-numeric value.`);
    arr.push(C(re, im));
  }
  if(!arr.length) throw new Error('no data rows found.');
  return arr;
}

function setInputMode(mode){
  vs.input_mode = mode;
  inputDesc.textContent = mode==='time'
    ? 'Input array as time samples — brackets optional;\nexamples: 1, 2+3j, 3   or   [1,1,1,1,1]'
    : 'Input array as frequency samples — brackets optional;\nexamples: 0.3, 0, 0, 0.5+0.5j';
  applyInput();
}

function setIqMode(mode){
  vs.iq_mode = mode;
  refreshPlots();
  startAnim();
}

function updateShiftLabels(){
  el('vs-tshift-lbl').textContent = vs.tshift;
  el('vs-fshift-lbl').textContent = vs.fshift;
}

function setTshift(v){
  vs.tshift = v;
  if(vs.input_mode==='time') computeFvalues(); else computeTvalues();
  updateShiftLabels(); refreshPlots(); startAnim();
}
function setFshift(v){
  vs.fshift = v;
  if(vs.input_mode==='time') computeFvalues(); else computeTvalues();
  updateShiftLabels(); refreshPlots(); startAnim();
}

// events
inputEl.addEventListener('change', applyInput);
inputEl.addEventListener('keydown', e=>{ if(e.key==='Enter'){ applyInput(); }});

// CSV upload: the button reveals the format help + file picker; picking a file
// validates the format and loads it (or shows an "Incorrect format" message).
function closeUploadPanel(){
  const p = el('vs-upload-panel'), m = el('vs-upload-msg');
  p.style.display = 'none';
  m.className = 'vs-upload-msg'; m.textContent = '';
}
function closeSamplesPanel(){
  const p = el('vs-samples-panel'), m = el('vs-samples-msg');
  p.style.display = 'none';
  m.className = 'vs-upload-msg'; m.textContent = '';
}
// Toggle a panel open/closed; opening one closes the other and clears its message.
function togglePanel(panel, closeOther, msg){
  const open = (panel.style.display === 'none' || !panel.style.display);
  closeOther();
  if(open){ msg.className = 'vs-upload-msg'; msg.textContent = ''; }
  panel.style.display = open ? 'block' : 'none';
}
el('vs-upload-btn').addEventListener('click', ()=>
  togglePanel(el('vs-upload-panel'), closeSamplesPanel, el('vs-upload-msg')));
el('vs-samples-btn').addEventListener('click', ()=>
  togglePanel(el('vs-samples-panel'), closeUploadPanel, el('vs-samples-msg')));

el('vs-upload-file').addEventListener('change', e=>{
  const file = e.target.files && e.target.files[0];
  const msg = el('vs-upload-msg');
  if(!file) return;
  const reader = new FileReader();
  reader.onload = ()=>{
    try{
      const arr = parseCsvIQ(String(reader.result));
      inputEl.value = arr.map(fmtComplex).join(', ');   // reuse the normal text-input path
      applyInput();
      e.target.value = '';        // allow re-selecting the same file next time
      closeUploadPanel();         // success -> collapse back to just the Upload button
      return;
    }catch(err){
      msg.className = 'vs-upload-msg err';
      msg.textContent = 'Incorrect format: ' + err.message;   // keep panel open on error
    }
    e.target.value = '';
  };
  reader.onerror = ()=>{ msg.className = 'vs-upload-msg err'; msg.textContent = 'Incorrect format: could not read the file.'; };
  reader.readAsText(file);
});

// Downloadable sample CSVs. To offer another, drop a file in samples/ and add an
// entry here — the list below is built from this array.
const SAMPLES = [
  { file:'samples/halfband_17_kaiser12.csv', label:'Half-band low-pass filter', note:'17 taps · Kaiser β=12' },
  { file:'samples/hilbert_31_kaiser8.csv',   label:'Hilbert transformer',        note:'31 taps · 2/n · Kaiser β=8' },
];
(function buildSamplesList(){
  const list = el('vs-samples-list');
  if(!list) return;
  for(const s of SAMPLES){
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'vs-sample-item';
    const strong = document.createElement('b'); strong.textContent = s.label;
    const note = document.createElement('span'); note.className = 'note'; note.textContent = s.note;
    b.append(strong, note);
    b.addEventListener('click', ()=> loadSample(s));
    list.appendChild(b);
  }
})();
// Clicking a sample: fetch it, load its array into the Input Array, and — only if
// the "Save CSV to disk" checkbox is ticked — download the CSV; then close the panel.
function loadSample(s){
  const msg = el('vs-samples-msg');
  msg.className = 'vs-upload-msg'; msg.textContent = 'Loading…';
  fetch(s.file)
    .then(r=>{ if(!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
    .then(text=>{
      const arr = parseCsvIQ(text);
      inputEl.value = arr.map(fmtComplex).join(', ');   // auto-load into the input
      applyInput();
      const save = el('vs-samples-save');
      if(save && save.checked){                         // optionally save the CSV to disk
        const a = document.createElement('a');
        a.href = s.file; a.download = s.file.split('/').pop();
        document.body.appendChild(a); a.click(); a.remove();
      }
      closeSamplesPanel();
    })
    .catch(err=>{ msg.className = 'vs-upload-msg err'; msg.textContent = 'Could not load sample: ' + err.message; });
}

for(const r of document.querySelectorAll('input[name="inputmode"]'))
  r.addEventListener('change', e=> setInputMode(e.target.value));
for(const r of document.querySelectorAll('input[name="iqmode"]'))
  r.addEventListener('change', e=> setIqMode(e.target.value));

el('vs-tplus').addEventListener('click', ()=> setTshift(vs.tshift+1));
el('vs-tminus').addEventListener('click',()=> setTshift(vs.tshift-1));
el('vs-fplus').addEventListener('click', ()=> setFshift(vs.fshift+1));
el('vs-fminus').addEventListener('click',()=> setFshift(vs.fshift-1));

el('vs-time').addEventListener('input', e=>{
  vs.refresh = +e.target.value; el('vs-time-val').textContent = e.target.value;
});
el('vs-frames').addEventListener('input', e=>{
  vs.nsamps = +e.target.value; el('vs-frames-val').textContent = e.target.value;
  refreshPlots();
});

el('vs-startstop').addEventListener('click', ()=>{
  if(anim.running) stopAnim(); else startAnim();
});
el('vs-savegif').addEventListener('click', saveGif);
el('vs-clear').addEventListener('click', ()=>{
  stopAnim();
  vs.tvalues=[]; vs.fvalues=[]; vs.tshift=0; vs.fshift=0;
  inputEl.value=''; inputEl.classList.remove('vs-error');
  updateShiftLabels();
  PD=null; anim.chainX=[0]; anim.chainY=[0]; clearHistory(); highlightIndex=null;
  render();
});

// Responsive relayout: switch between the wide desktop layout and the stacked
// phone layout when the viewport crosses the breakpoint (or the container width
// changes). Rebuild only on an actual mode/size change to avoid thrashing during
// animation or scroll. The running animation repaints itself; when paused we
// render() explicitly.
let _relayoutRAF = 0, _lastKey = '';
function scheduleRelayout(){
  if(gifBusy) return;   // never relayout mid-GIF capture (it forces wide temporarily)
  cancelAnimationFrame(_relayoutRAF);
  _relayoutRAF = requestAnimationFrame(()=>{
    const mode = currentMode();
    const wrapW = (canvas.parentElement && canvas.parentElement.clientWidth) || 360;
    const key = mode + ':' + (mode==='wide' ? '1080' : Math.round(clampw(wrapW,340,560)));
    if(key === _lastKey) return;
    _lastKey = key;
    computeLayout(mode); applyCanvasSize(); placeIqMode(mode); placeInputBlock(mode);
    if(!anim.running) render();
  });
}
if(window.ResizeObserver){
  // observe the wrapper, not the canvas, to avoid feedback from our own resize
  new ResizeObserver(scheduleRelayout).observe(canvas.parentElement);
}
window.matchMedia('(max-width:700px)').addEventListener('change', scheduleRelayout);

// ---- Impulse spotlight: hover (desktop) / tap (phone) a source-domain impulse ----
// Which canvas holds the interactive (source-domain) strip charts in this layout.
function interactiveCanvas(){
  return (layoutMode==='stacked') ? {cnv:canvas2, w:CW2, h:CH2} : {cnv:canvas, w:CW, h:CH};
}
// Map a pointer event to the source-domain impulse index under it, or null.
function hitTestImpulse(e, cnv, w, h){
  if(!PD) return null;
  const r = cnv.getBoundingClientRect();
  if(!r.width || !r.height) return null;
  const lx = (e.clientX - r.left) * w / r.width;   // -> logical coords (dpr absorbed by setTransform)
  const ly = (e.clientY - r.top)  * h / r.height;
  const srcSide = (vs.iq_mode==='time') ? 'freq' : 'time';   // stem/source domain
  const d = PD[srcSide]; const N = d.index.length;
  const shift = (srcSide==='freq') ? vs.fshift : vs.tshift;
  for(const suffix of ['_mag','_phase']){
    const mp = mappers[srcSide+suffix];
    if(!mp) continue;
    if(lx < mp.ix-4 || lx > mp.ix+mp.iw+4 || ly < mp.iy || ly > mp.iy+mp.ih) continue;
    const sx = mp.iw/(mp.xlim[1]-mp.xlim[0]);
    const dataX = mp.xlim[0] + (lx-mp.ix)/sx;
    const k = Math.round(dataX - shift);
    if(k>=0 && k<N && Math.abs((k+shift)-dataX) <= 0.5) return k;   // ±0.5 band per impulse
  }
  return null;
}
function setHighlight(k){
  if(k===highlightIndex) return;
  highlightIndex = k;
  if(!anim.running) render();      // running loop repaints itself
}
function onPointerMove(e){
  if(e.pointerType==='touch') return;            // touch has no hover; handled by click
  const ic = interactiveCanvas();
  if(e.currentTarget !== ic.cnv) return;
  const k = hitTestImpulse(e, ic.cnv, ic.w, ic.h);
  ic.cnv.style.cursor = (k!=null) ? 'pointer' : 'default';
  setHighlight(k);                                // clears (null) the moment it leaves an impulse
}
function onPointerLeave(e){
  if(e.pointerType==='touch') return;
  setHighlight(null);
}
for(const c of [canvas, canvas2]){
  c.addEventListener('pointermove', onPointerMove);
  c.addEventListener('pointerleave', onPointerLeave);
}
// Tap path (phone) + desktop click, handled at the document level so that ANY
// click not on an impulse clears the selection (taps on the IQ plot, the controls,
// or empty space) — only a tap on an impulse in the interactive canvas selects.
document.addEventListener('click', e=>{
  const ic = interactiveCanvas();
  setHighlight(e.target === ic.cnv ? hitTestImpulse(e, ic.cnv, ic.w, ic.h) : null);
});

// initial state
updateShiftLabels();
applyInput();   // loads the prefilled [1, 2+3j, 3]

})();

// ---------------------------------------------------------------------------
// Iframe height reporting (for embedding, e.g. Squarespace)
// ---------------------------------------------------------------------------
(function(){
  if(window.parent===window) return;   // not embedded
  function postHeight(){
    var h=Math.ceil(document.documentElement.getBoundingClientRect().height);
    window.parent.postMessage({type:'vectorspin:height', height:h}, '*');
  }
  window.addEventListener('load', postHeight);
  window.addEventListener('resize', postHeight);
  if(window.ResizeObserver){ new ResizeObserver(postHeight).observe(document.body); }
  postHeight();
})();

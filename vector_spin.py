# -*- coding: utf-8 -*-
"""
Vector Spin Graphing Utility - Updated for Modern Matplotlib

Updated December 2024 for compatibility with current Matplotlib versions

@author: C Daniel Boschen

MIT License
SPDX-License-Identifier: MIT

VectorSpin © 2026 The DSP Coach (https://dsp-coach.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of the software (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlib
from matplotlib import animation
import scipy.fft as fft
from collections import deque
import matplotlib.widgets as widg
import scipy.signal as sig

try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    print("Warning: pyperclip not installed. Clipboard features disabled.")
    print("Install with: conda install -c conda-forge pyperclip")

mlib.use("Qt5Agg")

# Phase wraps at +179° / -181° instead of ±180°: a real signal's 180° phase,
# which numerical noise flips between +pi and -pi, then always lands at -180°.
PHASE_WRAP_HI = np.pi - np.pi / 180
# |sample| at or below this is numerically zero, so its phase is undefined:
# the phase plots show it as a gray dot at 0.
PHASE_ZERO_EPS = 1e-15
# Same idea for the dense "ideal" curves, but relative to the curve's peak so
# it absorbs FFT round-off (~N·eps·peak, well above 1e-15 for 300-point sums).
PHASE_ZERO_REL = 1e-12
# Phase plots span slightly past ±pi so content at -180° (the default for real
# signals after the wrap above) sits visibly inside the plot, not on its border.
PHASE_YLIM = (-np.pi - 0.25, np.pi + 0.25)


def wrap_phase(ph):
    """Wrap angle(s) in radians into (-181°, +179°] instead of (-180°, +180°]."""
    return np.where(ph > PHASE_WRAP_HI, ph - 2 * np.pi, ph)


def curve_phase(values):
    """Phase of a dense ideal curve. Zero-crossing points whose magnitude is
    pure round-off have undefined phase — blank them to NaN so the plotted
    line breaks there instead of spiking through a noise angle."""
    mag = np.abs(values)
    ph = np.asarray(wrap_phase(np.angle(values)), dtype=float)
    ph[mag <= mag.max() * PHASE_ZERO_REL] = np.nan
    return ph


class MyRadioButtons(widg.RadioButtons):

    def __init__(self, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="vertical", **kwargs):
        widg.AxesWidget.__init__(self, ax)

        # AxesWidget.__init__ already exposes `canvas` as a read-only property
        # (ax.figure.canvas) in current Matplotlib; assigning it raises AttributeError.
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([], [], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self._buttons = self.box.legend_handles
        
        self._activecolor = activecolor

        for c in self._buttons:
            c.set_picker(5)

        try:
            self._observers = mlib.cbook.CallbackRegistry(signals=["clicked"])
        except TypeError:
            self._observers = mlib.cbook.CallbackRegistry()

        self.connect_event('pick_event', self._clicked)

    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
                event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self._buttons:
            self.set_active(self._buttons.index(event.artist))
    
    def set_active(self, index):
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid RadioButton index: {index}')
        
        self.value_selected = self.labels[index].get_text()
        
        for i, btn in enumerate(self._buttons):
            if i == index:
                btn.set_facecolor(self._activecolor)
            else:
                btn.set_facecolor(self.ax.get_facecolor())
        
        if self.drawon:
            self.ax.figure.canvas.draw()
        
        if not self.eventson:
            return
        
        self._observers.process('clicked', self.value_selected)
    
    def on_clicked(self, func):
        return self._observers.connect('clicked', lambda val: func(val))


class VectorSpin():

    def __init__(self):
        self.iq_mode = 'time'
        self.input_mode = 'time'
        self.reset()
        self.input_marker = ('bo', 5)
        self.result_marker = ('ro', 3)
        self.phasor_color = 'green'
        self.history_color = 'blue'
        self.animation = False
        self.tmarker = self.input_marker
        self.fmarker = self.result_marker
        self.create_fig()

    def reset(self):
        self.nsamps = 300
        self.refresh = 100
        self.fshift = 0
        self.tshift = 0
        self.tvalues = np.array([])
        self.fvalues = np.array([])
        self.set_history_len(int(self.nsamps * .05))

    def create_fig(self):
        self.fig = plt.figure(figsize=(13, 7), facecolor='white')
        self.win = self.fig.canvas.manager.window
        try:
            self.win.setFixedSize(self.win.size())
        except:
            pass

        iq = plt.subplot2grid((2, 4), (0, 1), rowspan=2, colspan=2)
        iq.text(.02, .02, "Dan Boschen 2023, Vector Spin", transform=iq.transAxes)
        if self.iq_mode == 'time':
            iq.set_title("IQ Plot - Time Domain")
        else:
            iq.set_title("IQ Plot - Freq Domain")
        iq.grid()
        iq.set_xlabel("I")
        iq.set_ylabel("Q")
        iq.set_aspect('equal')

        time_mag = plt.subplot2grid((2, 4), (0, 0))
        time_mag.set_title("Time-Domain: Magnitude")
        time_mag.grid()
        time_phase = plt.subplot2grid((2, 4), (1, 0))
        time_phase.set_title("Time-Domain: Phase")
        time_phase.set_ylabel("Radians")
        time_phase.set_xlabel("Time Index n")
        time_phase.grid()

        freq_mag = plt.subplot2grid((2, 4), (0, 3))
        freq_mag.set_title("Freq-Domain: Magnitude")
        freq_mag.grid()

        freq_phase = plt.subplot2grid((2, 4), (1, 3))
        freq_phase.set_title("Freq-Domain: Phase")
        freq_phase.set_ylabel("Radians")
        freq_phase.set_xlabel("Frequency Index k")
        freq_phase.grid()

        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.3)

        self.iq = iq
        self.time_mag = time_mag
        self.time_phase = time_phase
        self.freq_mag = freq_mag
        self.freq_phase = freq_phase
        self.create_timefreq_arrow()

    def create_timefreq_arrow(self):
        try:
            self.label.remove()
        except:
            pass
        try:
            self.timefreq_arrow.remove()
        except:
            pass
        if self.iq_mode == 'freq':
            self.timefreq_arrow = self.iq.annotate('',
                                                    xy=(-.24, 0.5),
                                                    xycoords='axes fraction',
                                                    xytext=(-.13, 0.5),
                                                    arrowprops=dict(arrowstyle="<|-", color='red', linewidth=5,
                                                                    mutation_scale=25))
            if self.input_mode == 'time':
                self.label = self.iq.text(-.2, 0.44, '/N', transform=self.iq.transAxes, color='red', fontsize=12)
        else:
            self.timefreq_arrow = self.iq.annotate('',
                                                    xy=(1.05, 0.5),
                                                    xycoords='axes fraction',
                                                    xytext=(1.15, 0.5),
                                                    arrowprops=dict(arrowstyle="-|>", color='red', linewidth=5,
                                                                    mutation_scale=25))
            if self.input_mode == 'freq':
                self.label = self.iq.text(1.1, 0.44, '/N', transform=self.iq.transAxes, color='red', fontsize=12)

    def plot_time(self):
        self.time_mag.clear()
        self.time_phase.clear()
        self.time_mag.set_title("Time-Domain: Magnitude")
        self.time_mag.grid()
        self.time_phase.set_ylabel("Radians")
        self.time_phase.set_xlabel("Time Index n")
        self.time_phase.set_title("Time-Domain: Phase")
        self.time_phase.grid()

        N = len(self.tvalues)
        tindex = np.arange(N) + self.tshift

        if N:
            self.time_mag.set_xlim((-.2 + tindex[0], .2 + tindex[-1] + 1))
            self.time_mag.set_ylim((0, 1.2 * np.max(np.abs(self.tvalues))))
            self.time_phase.set_xlim((-.2 + tindex[0], .2 + tindex[-1] + 1))
            self.time_phase.set_ylim(PHASE_YLIM)

            tzero = np.abs(self.tvalues) <= PHASE_ZERO_EPS
            tph = np.where(tzero, 0.0, wrap_phase(np.angle(self.tvalues)))

            self.time_mag.plot(tindex, np.abs(self.tvalues), self.tmarker[0], markersize=self.tmarker[1])
            self.time_phase.plot(tindex, tph, self.tmarker[0], markersize=self.tmarker[1])

            if self.iq_mode == 'freq':
                markerline, stemline, baseline = self.time_mag.stem(tindex, np.abs(self.tvalues),
                                                                     markerfmt=self.tmarker[0],
                                                                     basefmt='')
                plt.setp(markerline, markersize=self.tmarker[1])
                plt.setp(stemline, color=self.phasor_color)
                plt.setp(baseline, color='black')
                markerline, stemline, baseline = self.time_phase.stem(tindex, tph,
                                                                       markerfmt=self.tmarker[0],
                                                                       basefmt='')
                plt.setp(markerline, markersize=self.tmarker[1])
                plt.setp(stemline, color=self.phasor_color)
                plt.setp(baseline, color='black')
            else:
                x_axis = np.linspace(tindex[0], tindex[-1] + 1, self.nsamps, endpoint=False)
                phase = 2 * np.pi * (np.arange(self.nsamps) / self.nsamps)
                time_shiftf = self.fvalues * np.exp(
                    1j * self.tshift * 2 * np.pi * np.roll(np.arange(N), -self.fshift) / N)
                result = fft.ifft(time_shiftf * self.nsamps, self.nsamps) * np.exp(1j * self.fshift * phase)
                if self.input_mode == 'freq':
                    result = result / N
                self.time_mag.set_ylim((0, 1.2 * np.max(np.abs(result))))
                self.time_mag.plot(x_axis, np.abs(result), 'r', linewidth=0.3)
                self.time_phase.plot(x_axis, curve_phase(result), 'r', linewidth=0.3)
                self.mag, = self.time_mag.plot([], [], color=self.history_color, marker='.', linestyle='', ms=1)
                self.ph, = self.time_phase.plot([], [], color=self.history_color, marker='.', linestyle='', ms=1)

            # numerically-zero samples have undefined phase: overdraw a gray dot at 0
            if np.any(tzero):
                self.time_phase.plot(tindex[tzero], np.zeros(np.count_nonzero(tzero)),
                                     'o', color='gray', markersize=self.tmarker[1])

    def plot_freq(self):
        self.freq_mag.clear()
        self.freq_phase.clear()
        self.freq_phase.set_title("Freq-Domain: Phase")
        self.freq_mag.set_title("Freq-Domain: Magnitude")
        self.freq_mag.grid()
        self.freq_phase.set_ylabel("Radians")
        self.freq_phase.set_xlabel("Frequency Index k")
        self.freq_phase.grid()

        N = len(self.fvalues)
        findex = np.arange(N) + self.fshift

        if N:
            self.freq_mag.set_xlim((-.2 + findex[0], +0.2 + findex[-1] + 1))
            self.freq_mag.set_ylim((0, 1.2 * np.max(np.abs(self.fvalues))))
            self.freq_phase.set_xlim((-.2 + findex[0], +0.2 + findex[-1] + 1))
            self.freq_phase.set_ylim(PHASE_YLIM)
            fzero = np.abs(self.fvalues) <= PHASE_ZERO_EPS
            fph = np.where(fzero, 0.0, wrap_phase(np.angle(self.fvalues)))

            self.freq_mag.plot(findex, np.abs(self.fvalues), self.fmarker[0], markersize=self.fmarker[1])
            self.freq_phase.plot(findex, fph, self.fmarker[0], markersize=self.fmarker[1])

            if self.iq_mode == 'time':
                markerline, stemline, baseline = self.freq_mag.stem(findex, np.abs(self.fvalues),
                                                                     markerfmt=self.fmarker[0],
                                                                     basefmt='')
                plt.setp(markerline, markersize=self.fmarker[1])
                plt.setp(stemline, color=self.phasor_color)
                plt.setp(baseline, color='black')
                markerline, stemline, baseline = self.freq_phase.stem(findex, fph,
                                                                       markerfmt=self.fmarker[0],
                                                                       basefmt='')
                plt.setp(markerline, markersize=self.fmarker[1])
                plt.setp(stemline, color=self.phasor_color)
                plt.setp(baseline, color='black')
            else:
                x_axis = np.linspace(findex[0], findex[-1] + 1, self.nsamps, endpoint=False)
                phase = 2 * np.pi * (np.arange(self.nsamps) / self.nsamps)
                freq_shiftt = self.tvalues * np.exp(
                    -1j * self.fshift * 2 * np.pi * np.roll(np.arange(N), -self.tshift) / N)
                result = fft.fft(freq_shiftt * self.nsamps, self.nsamps) / self.nsamps * np.exp(-1j * self.tshift * phase)
                if self.input_mode == 'time':
                    result = result / N
                self.freq_mag.set_ylim((0, 1.2 * np.max(np.abs(result))))
                self.freq_mag.plot(x_axis, np.abs(result), 'r', linewidth=0.3)
                self.freq_phase.plot(x_axis, curve_phase(result), 'r', linewidth=0.3)
                self.mag, = self.freq_mag.plot([], [], color=self.history_color, marker='.', linestyle='', ms=1)
                self.ph, = self.freq_phase.plot([], [], color=self.history_color, marker='.', linestyle='', ms=1)

            # numerically-zero samples have undefined phase: overdraw a gray dot at 0
            if np.any(fzero):
                self.freq_phase.plot(findex[fzero], np.zeros(np.count_nonzero(fzero)),
                                     'o', color='gray', markersize=self.fmarker[1])

    def plot_iq(self):
        for artist in self.iq.lines:
            artist.remove()

        self.line, = self.iq.plot([], [], 'ro-', markersize=3, lw=1.5)
        self.line.set_color(self.phasor_color)
        self.trace, = self.iq.plot([], [], color=self.history_color, marker='.', linestyle='', ms=1)
        if self.iq_mode == 'time':
            array = self.tvalues
            marker = self.tmarker
        else:
            array = self.fvalues
            marker = self.fmarker
        N = len(array)
        if N:
            self.iq.plot(np.real(array), np.imag(array), marker[0], markersize=marker[1], zorder=2)
            phase = 2 * np.pi * np.arange(self.nsamps) / self.nsamps
            if self.iq_mode == 'time':
                if self.input_mode == 'time':
                    result = fft.ifft(self.fvalues * self.nsamps, self.nsamps) * np.exp(1j * self.fshift * phase)
                else:
                    result = fft.ifft(self.fvalues / N * self.nsamps, self.nsamps) * np.exp(1j * self.fshift * phase)
            else:
                if self.input_mode == 'time':
                    result = fft.fft(self.tvalues / N * self.nsamps, self.nsamps) / self.nsamps * np.exp(
                        -1j * self.tshift * phase)
                else:
                    result = fft.fft(self.tvalues * self.nsamps, self.nsamps) / self.nsamps * np.exp(
                        -1j * self.tshift * phase)
            self.iq.plot(np.real(result), np.imag(result), 'r', linewidth=0.3)
            scale = np.max(np.abs(result)) * 1.2
            self.iq.set_xlim((-scale, scale))
            self.iq.set_ylim((-scale, scale))
            self.iq.set_aspect('equal')

    def refresh_plots(self):
        self.plot_time()
        self.plot_freq()
        self.plot_iq()
        try:
            self.my_ani.event_source.stop()
            del self.my_ani
        except AttributeError:
            pass
        self.clear_history()
        self.create_timefreq_arrow()
        self.fig.canvas.draw()
        if len(self.tvalues) > 0:
            self.create_ani()

    def set_tshift(self, shift):
        self.tshift = shift
        if self.input_mode == 'time':
            self.compute_fvalues()
        else:
            self.compute_tvalues()
        self.refresh_plots()

    def set_input_mode(self, mode='time'):
        if mode.lower() == 'time':
            self.input_mode = 'time'
            self.tmarker = self.input_marker
            self.fmarker = self.result_marker
        else:
            self.input_mode = 'freq'
            self.tmarker = self.result_marker
            self.fmarker = self.input_marker

    def set_iq_mode(self, mode='time'):
        if mode.lower() == 'time':
            self.iq_mode = 'time'
            self.iq.set_title("IQ Plot - Time Domain")
        else:
            self.iq_mode = 'freq'
            self.iq.set_title("IQ Plot - Freq Domain")
        self.create_timefreq_arrow()
        self.refresh_plots()

    def set_fshift(self, shift):
        self.fshift = shift
        if self.input_mode == 'time':
            self.compute_fvalues()
        else:
            self.compute_tvalues()
        self.refresh_plots()

    def clear_history(self):
        self.history_x.clear()
        self.history_y.clear()
        self.history_tx.clear()
        self.history_mag.clear()
        self.history_ph.clear()

    def set_history_len(self, history_len):
        self.history_x = deque(maxlen=history_len)
        self.history_y = deque(maxlen=history_len)
        self.history_tx = deque(maxlen=history_len)
        self.history_xaxis = deque(maxlen=history_len)
        self.history_mag = deque(maxlen=history_len)
        self.history_ph = deque(maxlen=history_len)

    def compute_fvalues(self):
        N = len(self.tvalues)
        if N:
            self.fvalues = np.roll(
                fft.fft(self.tvalues) / N * np.exp(-1j * self.tshift * 2 * np.pi * np.arange(N) / N), -self.fshift)

    def compute_tvalues(self):
        N = len(self.fvalues)
        if N:
            self.tvalues = np.roll(fft.ifft(self.fvalues) * np.exp(1j * self.fshift * 2 * np.pi * np.arange(N) / N),
                                   -self.tshift)

    def create_ani(self):
        N = len(self.tvalues)
        if self.iq_mode == 'time':
            self.current_phasors = self.fvalues.copy() if self.input_mode == 'time' else self.fvalues.copy() / N
        else:
            self.current_phasors = self.tvalues.copy() if self.input_mode == 'freq' else self.tvalues.copy() / N

        def VectorBuilder():
            phase = 0
            sign = -1 if self.iq_mode == 'freq' else 1
            while True:
                phase = (phase + 2 * np.pi / self.nsamps) % (2 * np.pi)
                yield sign * phase

        def animate(phase):
            x = 0
            y = 0
            xs = [0]
            ys = [0]

            if phase == 0 or abs(phase) < 0.01:
                self.clear_history()

            for k, vec in enumerate(self.current_phasors):
                if self.iq_mode == 'time':
                    m = k + self.fshift
                    phase_offset = 2 * np.pi * self.tshift / len(self.current_phasors)
                else:
                    m = k + self.tshift
                    phase_offset = -2 * np.pi * self.fshift / len(self.current_phasors)
                x = xs[-1] + np.real(vec * np.exp(1j * m * (phase + phase_offset)))
                y = ys[-1] + np.imag(vec * np.exp(1j * m * (phase + phase_offset)))
                xs.append(x)
                ys.append(y)

            self.history_x.appendleft(x)
            self.history_y.appendleft(y)

            if self.iq_mode == 'time':
                sample_index = np.abs(phase) * (len(self.current_phasors)) / (2 * np.pi) + self.tshift
            else:
                sample_index = np.abs(phase) * (len(self.current_phasors)) / (2 * np.pi) + self.fshift
            self.history_tx.appendleft(sample_index)
            mag = np.abs(x + 1j * y)
            self.history_mag.appendleft(mag)
            self.history_ph.appendleft(0.0 if mag <= PHASE_ZERO_EPS
                                       else float(wrap_phase(np.arctan2(y, x))))
            self.line.set_data(xs, ys)
            self.trace.set_data(self.history_x, self.history_y)
            self.mag.set_data(self.history_tx, self.history_mag)
            self.ph.set_data(self.history_tx, self.history_ph)

            return self.line, self.trace, self.mag, self.ph

        try:
            self.my_ani.event_source.stop()
            del self.my_ani
        except AttributeError:
            pass

        if np.size(self.tvalues) != 0:
            self.animation = True
            self.my_ani = animation.FuncAnimation(fig=self.fig,
                                                  func=animate,
                                                  frames=VectorBuilder(),
                                                  interval=self.refresh,
                                                  save_count=self.nsamps,
                                                  blit=True,
                                                  cache_frame_data=False)


class SpinUI():

    def __init__(self, vspin):
        loc = (.22, .18)
        tloc = (.103, .2)
        floc = (.847, .2)
        sscloc = (.81, .05)

        self.input_desc = vspin.fig.text(*loc, f"(Input Array as {vspin.input_mode} samples, examples: [1,2+3j,3] )")

        user_entry_ax = vspin.fig.add_axes([loc[0], loc[1] - 0.06, 0.38, 0.05])
        self.user_entry = widg.TextBox(user_entry_ax, 'Input Array', hovercolor='0.975')

        input_select_ax = vspin.fig.add_axes([loc[0] + 0.38, loc[1] - 0.06, 0.14, 0.05])
        self.input_select = MyRadioButtons(input_select_ax, ['Time', 'Freq'], active=0,
                                           activecolor='blue', orientation="horizontal")

        self.iqmode_desc = vspin.fig.text(loc[0] - .05, loc[1] - 0.1, "IQ Mode")
        iq_mode_ax = vspin.fig.add_axes([loc[0], loc[1] - 0.13, 0.24, 0.05])
        self.iq_mode = MyRadioButtons(iq_mode_ax, ['Time Domain', 'Freq Domain'], active=0,
                                      activecolor='black', orientation="horizontal")

        update_time_ax = vspin.fig.add_axes([loc[0] + .4, loc[1] - 0.11, 0.1, 0.04])
        self.update_time = widg.Slider(update_time_ax, "Update Time (ms)", 10, 500, valinit=vspin.refresh, valstep=1)

        update_frames_ax = vspin.fig.add_axes([loc[0] + .4, loc[1] - 0.14, 0.1, 0.04])
        self.update_frames = widg.Slider(update_frames_ax, "Frames", 50, 1000, valinit=vspin.nsamps, valstep=10)

        tshift_minus_ax = vspin.fig.add_axes([tloc[0], tloc[1], 0.03, 0.03])
        self.tshift_minus = widg.Button(tshift_minus_ax, "<", hovercolor='0.975')
        tshift_plus_ax = vspin.fig.add_axes([tloc[0] + 0.05, tloc[1], 0.03, 0.03])
        self.tshift_plus = widg.Button(tshift_plus_ax, ">", hovercolor='0.975')

        fshift_minus_ax = vspin.fig.add_axes([floc[0], floc[1], 0.03, 0.03])
        self.fshift_minus = widg.Button(fshift_minus_ax, "<", hovercolor='0.975')
        fshift_plus_ax = vspin.fig.add_axes([floc[0] + 0.05, floc[1], 0.03, 0.03])
        self.fshift_plus = widg.Button(fshift_plus_ax, ">", hovercolor='0.975')

        self.start_ani_ax = vspin.fig.add_axes([sscloc[0], sscloc[1], 0.08, 0.05], visible=False)
        self.start_ani = widg.Button(self.start_ani_ax, "Start/Stop", hovercolor='0.975')

        self.clear_ax = vspin.fig.add_axes([sscloc[0] + .09, sscloc[1], 0.08, 0.05], visible=False)
        self.clear = widg.Button(self.clear_ax, "Clear", hovercolor='0.975')

        if CLIPBOARD_AVAILABLE:
            self.copy_ax = vspin.fig.add_axes([loc[0] + 0.54, loc[1] - 0.06, 0.04, 0.05])
            self.copy_btn = widg.Button(self.copy_ax, "Copy", hovercolor='0.975')
            self.paste_ax = vspin.fig.add_axes([loc[0] + 0.59, loc[1] - 0.06, 0.04, 0.05])
            self.paste_btn = widg.Button(self.paste_ax, "Paste", hovercolor='0.975')

        self.user_entry.on_submit(self.input_values)
        self.input_select.on_clicked(self.update_input_type)
        self.iq_mode.on_clicked(self.update_iq_mode)
        self.tshift_minus.on_clicked(self.minus_time)
        self.tshift_plus.on_clicked(self.plus_time)
        self.fshift_minus.on_clicked(self.minus_freq)
        self.fshift_plus.on_clicked(self.plus_freq)
        self.update_time.on_changed(self.change_time)
        self.update_frames.on_changed(self.change_frames)
        self.start_ani.on_clicked(self.start_stop_ani)
        self.clear.on_clicked(self.reset)
        
        if CLIPBOARD_AVAILABLE:
            self.copy_btn.on_clicked(self.copy_to_clipboard)
            self.paste_btn.on_clicked(self.paste_from_clipboard)

        self.vs = vspin

    def input_values(self, event):
        if event:
            try:
                values = eval(event)
                if self.vs.input_mode == 'time':
                    self.vs.tvalues = np.array(values)
                    self.vs.compute_fvalues()
                else:
                    self.vs.fvalues = np.array(values)
                    self.vs.compute_tvalues()
            except:
                print("Error in input values used.")
                self.user_entry.set_val("Input Error")
            self.start_ani_ax.set_visible(True)
            self.clear_ax.set_visible(True)
            self.vs.refresh_plots()

    def update_input_type(self, event):
        self.vs.clear_history()
        if event == "Time":
            self.vs.set_input_mode('time')
            self.input_values(self.user_entry.text)
        else:
            self.vs.set_input_mode('freq')
            self.input_values(self.user_entry.text)
        self.input_desc.set_text(f"(Input Array as {self.vs.input_mode} samples, examples: [1,2+3j,3] )")

    def update_iq_mode(self, event):
        self.vs.timefreq_arrow.remove()
        self.vs.clear_history()
        if event == "Time Domain":
            self.vs.set_iq_mode('time')
        else:
            self.vs.set_iq_mode('freq')

    def reset(self, event):
        self.user_entry.set_val("")
        self.update_time.set_val(self.vs.refresh)
        self.update_frames.set_val(self.vs.nsamps)
        self.vs.reset()
        self.start_ani_ax.set_visible(False)
        self.clear_ax.set_visible(False)
        self.vs.refresh_plots()

    def plus_time(self, event):
        self.vs.tshift += 1
        self.vs.set_tshift(self.vs.tshift)

    def minus_time(self, event):
        self.vs.tshift -= 1
        self.vs.set_tshift(self.vs.tshift)

    def plus_freq(self, event):
        self.vs.fshift += 1
        self.vs.set_fshift(self.vs.fshift)

    def minus_freq(self, event):
        self.vs.fshift -= 1
        self.vs.set_fshift(self.vs.fshift)

    def change_time(self, event):
        try:
            self.vs.my_ani.event_source.interval = event
            self.vs.refresh = event
        except AttributeError:
            self.vs.refresh = event

    def change_frames(self, event):
        self.vs.set_history_len(int(event * .05))
        self.vs.nsamps = int(event)

    def start_stop_ani(self, event):
        self.vs.animation = not self.vs.animation
        try:
            if self.vs.animation:
                self.vs.my_ani.resume()
            else:
                self.vs.my_ani.pause()
        except (AttributeError, RuntimeError):
            if self.vs.animation and len(self.vs.tvalues) > 0:
                self.vs.create_ani()

    def copy_to_clipboard(self, event):
        try:
            current_text = self.user_entry.text
            if current_text:
                pyperclip.copy(current_text)
                print(f"Copied to clipboard: {current_text}")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")

    def paste_from_clipboard(self, event):
        try:
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                self.user_entry.set_val(clipboard_text)
                print(f"Pasted from clipboard: {clipboard_text}")
                self.input_values(clipboard_text)
        except Exception as e:
            print(f"Error pasting from clipboard: {e}")

    def save_ani(self, filename='./animation.gif'):
        self.start_ani_ax.set_visible(False)
        self.clear_ax.set_visible(False)
        print(f"Saving animation as {filename}, this can take a few minutes...")
        self.vs.my_ani.save(filename, writer='ffmpeg', fps=60)
        print(f"Animation saved as {filename}.")
        self.start_ani_ax.set_visible(True)
        self.clear_ax.set_visible(True)


def run():
    vs = VectorSpin()
    ui = SpinUI(vs)
    plt.show()
    return vs, ui


if __name__ == "__main__":
    vs, ui = run()

    # Interesting cases to show:
    # sig.firls(31, [0,.4,.6,1], [1,1,0,0])
    # (in input_mode = time and iq_mode = Freq,
    #  dial time offset so that zero is at center tap)
    #
    # [0.3, 0,0,0,0,0,0,0,0,0,0.5+.5j, 0, 1+1j]
    # iq mode: time, input: freq, shift freq index to -9
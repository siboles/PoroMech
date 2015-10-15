from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import conj
import string


class Data(object):
    '''
    '''

    def __init__(self, filename):
        self.filename = filename
        # Dictionary to store analysis results
        self.results = {}
        self._parseFile()

    def _parseFile(self):
        if self.filename.lower().endswith('.tdms'):
            tdms = TdmsFile(self.filename)
            self.time = {}
            self.data = {}
            self.groups = tdms.groups()
            self.channels = {}
            for g in self.groups:
                self.time[g] = {}
                self.data[g] = {}
                self.channels[g] = tdms.group_channels(g)
                for c in self.channels[g]:
                    if c.has_data:
                        props = c.properties
                        self.time[g][props["NI_ChannelName"]] = c.time_track()
                        self.data[g][props["NI_ChannelName"]] = c.data

    def makePlot(self, group, keys, screen=True, disk=False, colors=False):
        if type(keys) is list:
            n = len(keys)
            f, axarr = plt.subplots(n, sharex=True)
            for i, k in enumerate(keys):
                axarr[i].plot(self.time[group][k], self.data[group][k])
                axarr[i].set_ylabel(k)
        else:
            ax = plt.plot(self.time[group][keys],
                          self.data[group][keys])
            ax.set_ylabel(keys)
            ax.set_xlabel("Time (s)")
        if screen:
            plt.show()

    def movingAverage(self, group, channel, win=10):
        newkey = string.join([channel, "avg", str(win)], "_")
        w = np.blackman(win)
        y = self.data[group][channel]
        s = np.r_[2 * y[0] - y[win:1:-1], y, 2 * y[-1] - y[-1:-win:-1]]
        self.data[group][newkey] = np.convolve(
            w / w.sum(), s, mode='same')[win - 1: -win + 1]
        self.time[group][newkey] = self.time[group][channel]

    def windowData(self, group, channel, start, end):
        newkey = string.join([channel, "crop", str(start), str(end)], "_")
        self.data[group][newkey] = self.data[group][channel][start:end + 1]
        self.time[group][newkey] = self.time[group][channel][start:end + 1]

    def getViscoModuli(self, group, disp, force):
        rkey = string.join([group, disp, force], "_")
        #Fourier transform each channel
        disp_dft = fft(self.data[group][disp])
        force_dft = fft(self.data[group][force])
        d = ifft(disp_dft * conj(force_dft))
        phase_shift = np.argmax(np.abs(d))
        disp_amp = (np.linalg.norm(self.data[group][disp])
                    - np.mean(self.data[group][disp])) / np.sqrt(2)
        force_amp = (np.linalg.norm(self.data[group][force])
                     - np.mean(self.data[group][force])) / np.sqrt(2)
        try:
            self.results['storage modulus'][rkey] = (force_amp / disp_amp
                                                     * np.cos(phase_shift))
            self.results['loss modulus'][rkey] = (force_amp / disp_amp
                                                  * np.sin(phase_shift))
        except:
            self.results['storage modulus'] = {}
            self.results['loss modulus'] = {}
            self.results['storage modulus'][rkey] = (force_amp / disp_amp
                                                     * np.cos(phase_shift))
            self.results['loss modulus'][rkey] = (force_amp / disp_amp
                                                  * np.sin(phase_shift))

        print self.results['storage modulus'][rkey]

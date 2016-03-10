from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import conj
import string
from itertools import izip, count, islice, ifilter
from collections import OrderedDict


class Data(object):
    '''
    '''

    def __init__(self, filename):
        self.filename = filename
        # Dictionary to store analysis results
        self.results = {}
        # Dictionary to store Mach-1 thicknesses
        self.thicknesses = {}
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
        elif self.filename.lower().endswith('.txt'):
            fid = open(self.filename, "r")
            if "<Mach-1 File>" in fid.readline():
                contents = fid.readlines()
                fid.close()
                self.groups = []
                self.time = OrderedDict()
                self.data = OrderedDict()
                self.channels = OrderedDict()
                info_blocks = [i for i, j in izip(count(), contents) if "<INFO>" in j or "<END INFO>" in j]
                info_blocks = izip(islice(info_blocks, 0, None, 2), islice(info_blocks, 1, None, 2))
                data_blocks = [i for i, j in izip(count(), contents) if "<DATA>" in j or "<END DATA>" in j]
                data_blocks = izip(islice(data_blocks, 0, None, 2), islice(data_blocks, 1, None, 2))
                for ind in info_blocks:
                    a = list(ifilter(lambda x: "Time" in x, contents[ind[0]+1:ind[1]]))[0].rstrip("\r\n")
                    self.groups.append(a.replace("\t", " "))
                for i, ind in enumerate(data_blocks):
                    g = self.groups[i]
                    header = contents[ind[0]+1].rstrip("\r\n").split("\t")
                    self.channels[g] = header
                    data = contents[ind[0]+2:ind[1]]
                    for j, d in enumerate(data):
                        data[j] = d.rstrip("\r\n").split("\t")
                    data = np.array(data, float)
                    self.time[g] = OrderedDict()
                    self.data[g] = OrderedDict()
                    for j, c in enumerate(self.channels[g][1:]):
                        self.time[g][c] = data[:, 0]
                        self.data[g][c] = data[:, j+1]

    def getMaxMinIndex(self, group, channel):
        return np.argmin(self.data[group][channel]), np.argmax(self.data[group][channel])

    def getThicknessMach1(self, group):
        ind = self.getMaxMinIndex(group, "Fz, N")[0]
        self.thicknesses[group] = self.data[group]["Position (z), mm"][ind]

    def movingAverage(self, group, channel, win=10):
        newkey = string.join([channel, "avg", str(win)], "_")
        w = np.blackman(win)
        y = self.data[group][channel]
        s = np.r_[2 * y[0] - y[win:1:-1], y, 2 * y[-1] - y[-1:-win:-1]]
        self.data[group][newkey] = np.convolve(
            w / w.sum(), s, mode='same')[win - 1: -win + 1]
        self.time[group][newkey] = self.time[group][channel]

    def windowData(self, group, channel, start, end):
        newkey = string.join([channel, "crop",
                              "{:6.2f}".format(start),
                              "{:6.2f}".format(end)], "_")
        start_index = np.argmin(np.abs(self.time[group][channel] - start))
        end_index = np.argmin(np.abs(self.time[group][channel] - end))
        self.data[group][newkey] = self.data[group][channel][start_index:end_index + 1]
        self.time[group][newkey] = self.time[group][channel][start_index:end_index + 1]

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

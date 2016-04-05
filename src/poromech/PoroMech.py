from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import conj
from scipy.signal import medfilt, find_peaks_cwt
from scipy.optimize import curve_fit
import string
from itertools import izip, count, islice, ifilter
from collections import OrderedDict
from operator import methodcaller
import pandas as pd


class Data(object):
    '''
    '''

    def __init__(self, filename):
        self.filename = filename
        # Dictionary to store analysis results
        self.results = {}
        # Dictionary to store Mach-1 thicknesses
        self.thicknesses = OrderedDict()
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
                self.time = OrderedDict()
                self.data = OrderedDict()
                self.channels = OrderedDict()
                info_blocks = [i for i, j in izip(count(), contents) if "<INFO>" in j or "<END INFO>" in j]
                info_blocks = izip(islice(info_blocks, 0, None, 2), islice(info_blocks, 1, None, 2))
                data_blocks = [i for i, j in izip(count(), contents) if "<DATA>" in j or "<END DATA>" in j]
                data_blocks = izip(islice(data_blocks, 0, None, 2), islice(data_blocks, 1, None, 2))
                self.groups = range(1, len(list(info_blocks))+1)
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

    def lineFit(self, x, y0, k1):
        return k1*x + y0

    def piecewiseLineFitObj(self, x, x0, y0, k1, k2):
        func = np.piecewise(x, [x < x0], [lambda x: k1*x + y0-k1*x0,
                                          lambda x: k2*x + y0-k2*x0])
        return func

    def getThicknessMach1(self, group):
        dt = self.time[group]["Fz, N"][1] - self.time[group]["Fz, N"][0]
        strain_rate = np.diff(self.data[group]["Position (z), mm"]) / dt
        if not np.any(strain_rate < 0):
            print("Warning: It seems the needle never made contact. Terminating the thickness calculation for group: {:d}".format(group))
            self.thicknesses[group] = (np.nan, (np.nan, np.nan))
            return
        strain_rate = np.mean(strain_rate[strain_rate > 0])
        end = self.getMaxMinIndex(group, "Fz, N")[0]
        tmp = self.data[group]["Fz, N"][0:end+1]
        start = np.where(np.abs(medfilt(tmp, kernel_size=19)) < 1e-2)[0][-1]
        windows = np.arange(int(.05/dt), int(1/dt), 1)
        transition = find_peaks_cwt(-tmp[start:], widths=windows, min_length=int(windows.size/2), min_snr=1.0)
        tmp = tmp[start:]
        transition = np.array(transition)
        if transition.size > 0:
            #shift start index to transition point (peak with max value at least 50 microns after contact)
            transition = transition[transition>.05/strain_rate/dt]
            if transition.any():
                ind = np.argmax(tmp[transition])
                transition = transition[ind]
                #if transition is within 10 microns of the minimum force position try scale-space with weaker tolerance
                if (tmp.size - transition) < .01/strain_rate/dt:
                    transition = find_peaks_cwt(-tmp, widths=windows, min_length=int(windows.size/4), min_snr=1.0)
                    transition = np.array(transition)
                    transition = transition[transition>.05/strain_rate/dt]
                    ind = np.argmax(tmp[transition])
                    transition = transition[ind]
                    #if still too close to minimum - revert to piecewise fit
                    if (tmp.size - transition) < .01/strain_rate/dt:
                        #fit line to first 50 microns
                        n = int(0.05/strain_rate/dt)
                        p1, e1 = curve_fit(self.lineFit, np.arange(n), tmp[0:n])
                        #fit line to last 50 microns
                        p2, e2 = curve_fit(self.lineFit, np.arange(tmp.size-n, tmp.size), tmp[-n:])
                        #intersection of these two lines
                        intersect = (p2[0] - p1[0]) / (p1[1] - p2[1])
                        #reslice tmp to have equal points before and after intersection
                        if 2*int(intersect) < tmp.size:
                            tmp = tmp[0:2*int(intersect)]
                        p, e = curve_fit(self.piecewiseLineFitObj, np.arange(tmp.size)*dt, tmp)
                        transition = p[0]/dt
            else:
                #fit line to first 50 microns
                n = int(0.05/strain_rate/dt)
                p1, e1 = curve_fit(self.lineFit, np.arange(n), tmp[0:n])
                #fit line to last 50 microns
                p2, e2 = curve_fit(self.lineFit, np.arange(tmp.size-n, tmp.size), tmp[-n:])
                #intersection of these two lines
                intersect = (p2[0] - p1[0]) / (p1[1] - p2[1])
                #reslice tmp to have equal points before and after intersection
                tmp = tmp[0:2*int(intersect)]

                p, e = curve_fit(self.piecewiseLineFitObj, np.arange(tmp.size)*dt, tmp)
                transition = p[0]/dt
        self.thicknesses[group] = (np.abs(self.data[group]["Position (z), mm"][start + int(transition)] -
                                          self.data[group]["Position (z), mm"][start]),
                                   (float(start + transition)*dt, start*dt))
    def readMach1PositionMap(self, filename):
        fid = open(filename, "r")
        lines = map(string.strip, fid.readlines())
        fid.close()
        start = lines.index("PixelX	PixelY	PointID	PointType	Sub-SurfaceID")
        lines =  map(methodcaller("split", "\t"), lines[start+1:])
        lines = [item for item in lines if len(item)==5]
        self.MachPositions = pd.DataFrame(lines, columns=["PixelX", "PixelY", "PointID", "PointType", "SubSurfaceID"])
        self.MachPositions[["PixelX", "PixelY", "PointID", "PointType"]] = self.MachPositions[["PixelX", "PixelY", "PointID", "PointType"]].astype(np.uint16)

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

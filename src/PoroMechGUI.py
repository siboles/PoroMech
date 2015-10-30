from Tkinter import *
import tkFileDialog
from PoroMech import Data
import os
import pickle
import string
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure


class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self.intSettings = {'Group': IntVar(value=1)}
        self.buttonLoadFile = Button(self, text="Load Data File",
                                     command=self.loadFile)
        self.buttonLoadFile.grid(row=0, column=0, padx=5, pady=5,
                                 sticky=W + E)
        self.buttonSaveFile = Button(self, text="Save Selected to Pickle",
                                     command=self.saveFile)
        self.buttonSaveFile.grid(row=3, column=0, padx=5, pady=5,
                                 sticky=NW)

        self.buttonSaveCSV = Button(self, text="Save Selected to CSV",
                                    command=self.saveCSV)
        self.buttonSaveCSV.grid(row=4, column=0, padx=5, pady=5, sticky=NW)

        self.buttonMovingAvg = Button(self, text="Apply Moving Average",
                                      command=self.applyMovingAvg)
        self.buttonMovingAvg.grid(row=2, column=1, padx=5, pady=5,
                                  sticky=W + E)
        self.windowSize = IntVar(value=10)
        Label(self, text="Window Size").grid(row=3, column=1, padx=5, pady=5,
                                             sticky=NW)
        Entry(self, textvariable=self.windowSize).grid(row=4, column=1, padx=5,
                                                       pady=5, sticky=NW)

        self.frameGroups = LabelFrame(self, text="Group Selection")
        self.frameGroups.grid(row=1, column=0, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.frameGroups, text="").pack(anchor=W)
        self.frameChannels = LabelFrame(self, text="Channel Selection")
        self.frameChannels.grid(row=1, column=1, padx=5, pady=5,
                                sticky=N + E + W + S)
        Label(self.frameChannels, text="").pack(anchor=W)

        self.buttonPlot = Button(self, text="Plot Selected Channels",
                                 command=self.plotChannels)
        self.buttonPlot.grid(row=2, column=0, padx=5, pady=5,
                             sticky=W + E)
        self.grid()

    def loadFile(self):
        self.filename = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Select a Data File.")
        if self.filename:
            self.FileObject = Data(self.filename)
            for child in self.frameGroups.pack_slaves():
                child.destroy()
            for g in self.FileObject.groups:
                Radiobutton(self.frameGroups,
                            text=g,
                            indicatoron=0,
                            width=20,
                            variable=self.intSettings["Group"],
                            command=self.populateChannelList,
                            value=1).pack(anchor=W)
            g = self.FileObject.groups[self.intSettings["Group"].get() - 1]
            self.channelSelections = {}
            for c in self.FileObject.time[g].keys():
                self.channelSelections[c] = IntVar(value=0)
                Checkbutton(self.frameChannels,
                            text=c,
                            variable=self.channelSelections[c]).pack(anchor=W)

    def makePlot(self, group, keys):
        f = Figure()
        self.axes = []
        if type(keys) is list:
            n = len(keys)
            for i, k in enumerate(keys):
                if i == 0:
                    self.axes.append(f.add_subplot(n, 1, i + 1))
                else:
                    self.axes.append(f.add_subplot(n, 1, i + 1,
                                                   sharex=self.axes[0]))
                self.axes[i].plot(self.FileObject.time[group][k],
                                  self.FileObject.data[group][k])
                self.axes[i].set_ylabel(k)
        else:
            self.axes.append(f.add_subplot(1,1,1))
            self.axes[0].plot(self.FileObject.time[group][keys],
                              self.FileObject.data[group][keys])
            self.axes[0].set_ylabel(keys)
        self.axes[-1].set_xlabel("Time (s)")
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, column=2, columnspan=2,
                                    padx=5, pady=5, sticky=NW)

        toolbar_frame = Frame(root)
        toolbar_frame.grid(row=1, column=2, sticky=NW)
        toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
        toolbar.update()

        Button(root, text="Crop", command=self.cropData).grid(
            row=1, column=3, sticky=NE)

    def cropData(self):
        group = self.FileObject.groups[self.intSettings["Group"].get() - 1]
        (start, end) = self.axes[0].xaxis.get_view_interval()
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                self.FileObject.windowData(group, c, start, end)
        self.populateChannelList()

    def populateChannelList(self):
        g = self.FileObject.groups[self.intSettings["Group"].get() - 1]
        self.channelSelections = {}
        for child in self.frameChannels.pack_slaves():
            child.destroy()
        for c in self.FileObject.time[g].keys():
            self.channelSelections[c] = IntVar(value=0)
            Checkbutton(self.frameChannels,
                        text=c,
                        variable=self.channelSelections[c]).pack(anchor=W)

    def plotChannels(self):
        keys = []
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                keys.append(c)

        self.makePlot(
            self.FileObject.groups[self.intSettings["Group"].get() - 1],
            keys)

    def applyMovingAvg(self):
        group = self.FileObject.groups[
            self.intSettings["Group"].get() - 1]
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                self.FileObject.movingAverage(
                    group, c, win=self.windowSize.get())
        self.populateChannelList()

    def saveFile(self):
        group = self.FileObject.groups[self.intSettings["Group"].get() - 1]
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                fid = open(os.path.abspath(string.replace(
                    self.filename, ".tdms", "_{:s}_{:s}.pkl".format(group, c))), "wb")
                pickle.dump((self.FileObject.time[group][c], self.FileObject.data[group][c]),
                            fid, 2)
                fid.close()
    def saveCSV(self):
        group = self.FileObject.groups[self.intSettings["Group"].get() - 1]
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                fid = open(os.path.abspath(string.replace(
                    self.filename, ".tdms", "_{:s}_{:s}.csv".format(group, c))), "wt")
                fid.write("Time, {:s}\n".format(c))
                for (t, v) in zip(
                        self.FileObject.time[group][c], self.FileObject.data[group][c]):
                    fid.write("{:12.6f}, {:12.6f}\n".format(t, v))
                fid.close()

root = Tk()
root.title("Welcome to the PoroMech GUI.")
app = Application(root)

root.mainloop()

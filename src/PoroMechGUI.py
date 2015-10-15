from Tkinter import *
import tkFileDialog
from PoroMech import Data
import os
import pickle


class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self.intSettings = {'Group': IntVar(value=1)}
        self.buttonLoadFile = Button(self, text="Load Data File",
                                     command=self.loadFile)
        self.buttonLoadFile.grid(row=0, column=0, padx=5, pady=5,
                                 sticky=W + E)
        self.buttonSaveFile = Button(self, text="Save to CSV",
                                     command=self.saveFile)
        self.buttonSaveFile.grid(row=0, column=1, padx=5, pady=5,
                                 sticky=W + E)

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

        self.FileObject.makePlot(
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
        pass

root = Tk()
root.title("Welcome to the PoroMech GUI.")
app = Application(root)

root.mainloop()

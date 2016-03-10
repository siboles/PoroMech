from Tkinter import *
from ttk import Notebook
import tkFileDialog
from PoroMech import Data
import os
import pickle
import string
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from numpy import ma
from scipy.ndimage import imread
from scipy.interpolate import griddata
from PIL import Image, ImageDraw

sns.set()


class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self.notebook = Notebook(self)
        self.tab1 = Frame(self.notebook)
        self.tab2 = Frame(self.notebook)
        self.notebook.add(self.tab1, text="Data Interaction")
        self.notebook.add(self.tab2, text="Mach-1 Image Grid")
        self.notebook.grid(row=0, column=0, sticky=NW)

        #####  BEGIN TAB 1 #####
        self.intSettings = {'Group': IntVar(value=1)}
        self.buttonLoadFile = Button(self.tab1, text="Load Data File",
                                     command=self.loadFile)
        self.buttonLoadFile.grid(row=0, column=0, padx=5, pady=5,
                                 sticky=W + E)
        self.buttonSaveFile = Button(self.tab1, text="Save Selected to Pickle",
                                     command=self.saveFile)
        self.buttonSaveFile.grid(row=3, column=0, padx=5, pady=5,
                                 sticky=NW)

        self.buttonGetThickness = Button(self.tab1, text="Get Mach-1 Thicknesses",
                                         command=self.findThicknesses)
        self.buttonGetThickness.grid(row=5, column=0, padx=5, pady=5, sticky=NW)

        self.buttonSaveCSV = Button(self.tab1, text="Save Selected to CSV",
                                    command=self.saveCSV)
        self.buttonSaveCSV.grid(row=4, column=0, padx=5, pady=5, sticky=NW)

        self.buttonMovingAvg = Button(self.tab1, text="Apply Moving Average",
                                      command=self.applyMovingAvg)
        self.buttonMovingAvg.grid(row=2, column=1, padx=5, pady=5,
                                  sticky=W + E)
        self.windowSize = IntVar(value=10)
        Label(self.tab1, text="Window Size").grid(row=3, column=1, padx=5, pady=5,
                                             sticky=NW)
        Entry(self.tab1, textvariable=self.windowSize).grid(row=4, column=1, padx=5,
                                                       pady=5, sticky=NW)

        self.frameGroups = LabelFrame(self.tab1, text="Group Selection")
        self.frameGroups.grid(row=1, column=0, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.frameGroups, text="").grid(row=0, column=0, sticky=NW)
        self.frameChannels = LabelFrame(self.tab1, text="Channel Selection")
        self.frameChannels.grid(row=1, column=1, padx=5, pady=5,
                                sticky=N + E + W + S)
        Label(self.frameChannels, text="").grid(row=0, column=0, sticky=NW)

        self.buttonPlot = Button(self.tab1, text="Plot Selected Channels",
                                 command=self.plotChannels)
        self.buttonPlot.grid(row=2, column=0, padx=5, pady=5,
                             sticky=W + E)
        #####  END TAB 1 #####

        ##### BEGIN TAB 2 #####
        self.buttonLoadImage = Button(self.tab2, text="Load Image",
                                      command=self.loadImage)
        self.buttonLoadImage.grid(row=0, column=0, padx=5, pady=5, sticky=NW)
        ##### END TAB 2 #####
        self.grid()

    def loadFile(self):
        self.filename = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Select a Data File.")
        if self.filename:
            self.FileObject = Data(self.filename)
            for child in self.frameGroups.grid_slaves():
                child.grid_remove()
                del child
            row = 0
            column = 0
            for i, g in enumerate(self.FileObject.groups):
                if i % 12 == 0:
                    row = 0
                    column += 1
                Radiobutton(self.frameGroups,
                            text=g,
                            indicatoron=0,
                            width=20,
                            variable=self.intSettings["Group"],
                            command=self.populateChannelList,
                            value=i+1).grid(row=row, column=column, sticky=NW)
                row += 1
            g = self.FileObject.groups[self.intSettings["Group"].get() - 1]
            for child in self.frameChannels.grid_slaves():
                child.grid_remove()
                del child
            row = 0
            column = 0
            self.channelSelections = {}
            for c in self.FileObject.time[g].keys():
                if i % 12 == 0:
                    row = 0
                    column += 1
                self.channelSelections[c] = IntVar(value=0)
                Checkbutton(self.frameChannels,
                            text=c,
                            variable=self.channelSelections[c]).grid(row=row, column=column, sticky=NW)
                row += 1

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
                if self.FileObject.thicknesses and k == "Position (z), mm":
                    self.axes[i].axhline(y=self.FileObject.thicknesses[group])
                self.axes[i].set_ylabel(k)
        else:
            self.axes.append(f.add_subplot(1,1,1))
            self.axes[0].plot(self.FileObject.time[group][keys],
                              self.FileObject.data[group][keys])
            if self.FileObject.thicknesses and k == "Position (z), mm":
                self.axes[i].axhline(y=self.FileObject.thicknesses[group])
            self.axes[0].set_ylabel(keys)
        self.axes[-1].set_xlabel("Time (s)")
        canvas = FigureCanvasTkAgg(f, master=self.tab1)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, column=2, columnspan=2,
                                    rowspan=4, padx=5, pady=5, sticky=NW)

        toolbar_frame = Frame(self.tab1)
        toolbar_frame.grid(row=5, column=2, sticky=NW)
        toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
        toolbar.update()

        Button(self.tab1, text="Crop", command=self.cropData).grid(
            row=5, column=3, sticky=NE)

    def findThicknesses(self):
        for i, g in enumerate(self.FileObject.groups):
            self.FileObject.getThicknessMach1(g)

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
        for child in self.frameChannels.grid_slaves():
            child.grid_remove()
            del child
        row = 0
        column = 0
        for i, c in enumerate(self.FileObject.time[g].keys()):
            if i % 12 == 0:
                row = 0
                column += 1
            self.channelSelections[c] = IntVar(value=0)
            Checkbutton(self.frameChannels,
                        text=c,
                        variable=self.channelSelections[c]).grid(row=row, column=column, sticky=NW)
            row += 1

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
                if self.filename.lower().endswith(".tdms"):
                    oname = string.replace(self.filename, ".tdms", "_{:s}_{:s}.csv".format(group, c))
                    oname = string.replace(oname, " ", "_")
                    fid = open(os.path.abspath(oname), "wt")
                elif self.filename.lower().endswith(".txt"):
                    oname = string.replace(self.filename, ".txt", "_{:s}_{:s}.csv".format(group, c))
                    oname = string.replace(oname, " ", "_")
                    fid = open(os.path.abspath(oname), "wt")
                fid.write("Time, {:s}\n".format(c))
                for (t, v) in zip(
                        self.FileObject.time[group][c], self.FileObject.data[group][c]):
                    fid.write("{:12.6f}, {:12.6f}\n".format(t, v))
                fid.close()

    def loadImage(self):
        self.imagefile = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            filetypes=[('image files', '.jpg .jpeg .png .tif .tiff')],
            title="Select a Image.")
        if not self.imagefile:
            print("A file was not selected")
            return
        self.image = imread(self.imagefile)
        self.cropImage()

        #try:
        #    self.image = imread(self.imagefile)
        #    self.cropImage()
        #except:
        #    print("Image loading failed. Ensure that the selected file is a valid image.")

    def cropImage(self):
        f = Figure()
        ax = f.add_subplot(111)
        ax.imshow(self.image)
        self.points = []
        self.polygons = []
        canvas = FigureCanvasTkAgg(f, master=self.tab2)
        canvas.get_tk_widget().bind("<Button-1>", self.XY_handler)
        canvas.get_tk_widget().bind("<Button-2>", self.nextPolygon)
        canvas.get_tk_widget().bind("<Return>", self.UpdateMask)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, column=2, columnspan=2,
                                    rowspan=4, padx=5, pady=5, sticky=NW)

    def XY_handler(self, aHandledEVENT):
        self.points.append((aHandledEVENT.x, aHandledEVENT.y))

    def nextPolygon(self, aHandledEVENT):
        self.polygons.append(np.copy(self.points))
        self.points = []

    def UpdateMask(self, aHandledEVENT):
        img = Image.new('L', self.image.shape[0:2], 0)
        for p in self.polygons:
            ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
        self.maskimage = np.array([np.array(img), np.array(img), np.array(img)])
        matplotlib.pyplot.imshow(ma.masked_where(self.maskimage==0, self.image))
        matplotlib.pyplot.show()


    def plotImage(self, data=None):
        f = Figure()
        ax = f.add_subplot(111)
        if self.cropimage:
            ax.imshow(self.cropimage)
        else:
            ax.imshow(self.image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.grid(False)
        grid_size = (960, 1280)
        datax = np.random.randint(grid_size[0], size=(22, 1))
        datay = np.random.randint(grid_size[1], size=(22, 1))
        z = np.random.rand(22)
        gridx, gridy = np.mgrid[0:grid_size[0], 0:grid_size[1]]
        data = np.hstack((datax, datay))
        data = np.float32(data)
        gridz = griddata(data, z, (gridx, gridy), method='nearest')
        if not(data is None):
            ax.hold(True)
            cmap = sns.cubehelix_palette(light=1, as_cmap=True)
            im = ax.imshow(gridz, cmap=cmap, alpha=0.5)
            f.colorbar(im)

        canvas = FigureCanvasTkAgg(f, master=self.tab2)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, column=2, columnspan=2,
                                    rowspan=4, padx=5, pady=5, sticky=NW)


root = Tk()
root.title("Welcome to the PoroMech GUI.")
app = Application(root)

root.mainloop()

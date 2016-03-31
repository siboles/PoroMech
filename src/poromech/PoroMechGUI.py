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
from matplotlib.path import Path
import matplotlib.patches as patches
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

        self.buttonLoadMapFile = Button(self.tab2, text="Load Mach-1 Site Locations",
                                        command=self.loadMachMap)

        self.buttonLoadMapFile.grid(row=1, column=0, padx=5, pady=5, sticky=NW)
        self.buttonDefineMask = Button(self.tab2, text="Define Mask",
                                       command=self.cropImage)
        self.buttonDefineMask.grid(row=2, column=0, padx=5, pady=5, sticky=NW)
        self.buttonClearMask = Button(self.tab2, text="Clear Mask",
                                      command=self.clearMask)
        self.buttonClearMask.grid(row=3, column=0, padx=5, pady=5, sticky=NW)

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
                if self.FileObject.thicknesses and k == "Fz, N":
                    try:
                        self.axes[i].axvline(x=self.FileObject.thicknesses[group][1][0], color='r')
                        self.axes[i].axvline(x=self.FileObject.thicknesses[group][1][1], color='g')
                    except:
                        pass
                self.axes[i].set_ylabel(k)
        else:
            self.axes.append(f.add_subplot(1,1,1))
            self.axes[0].plot(self.FileObject.time[group][keys],
                              self.FileObject.data[group][keys])
            if self.FileObject.thicknesses and k == "Fz, N":
                self.axes[i].axvline(l=self.FileObject.thicknesses[group][1][0], color='r')
                self.axes[i].axvline(l=self.FileObject.thicknesses[group][1][1], color='g')
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
        self.cropimage=False
        self.plotImage()

        #try:
        #    self.image = imread(self.imagefile)
        #    self.cropImage()
        #except:
        #    print("Image loading failed. Ensure that the selected file is a valid image.")
    def loadMachMap(self):
        self.mapfile = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            filetypes=[('Mach-1 map files', '.map')],
            title="Select the Mach-1 map file")
        if not self.mapfile:
            print("A file was not selected")
            return
        self.FileObject.readMach1PositionMap(self.mapfile)

    def cropImage(self):
        self.image_fig = Figure((6.0, 6.0/self.image_aspect), dpi=self.image_dpi, frameon=False)
        self.image_ax = self.image_fig.add_axes([0, 0, 1.0, 1.0,])
        self.image_ax.imshow(self.image)
        self.image_ax.get_xaxis().set_visible(False)
        self.image_ax.get_yaxis().set_visible(False)
        self.image_ax.grid(False)
        self.points = []
        self.polygons = []
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=self.tab2)
        self.image_canvas.get_tk_widget().bind("<Button-1>", self.XY_handler)
        self.image_canvas.get_tk_widget().bind("<Button-3>", self.nextPolygon)
        self.image_canvas.get_tk_widget().bind("<Return>", self.UpdateMask)
        self.image_canvas.show()
        self.image_canvas.get_tk_widget().grid(row=0, column=2, columnspan=2,
                                    rowspan=4, padx=5, pady=5, sticky=NW)

    def XY_handler(self, aHandledEVENT):
        self.points.append((aHandledEVENT.x, aHandledEVENT.y))
        if len(self.points) > 1:
            self.addLine()

    def addLine(self):
        codes = [Path.MOVETO] + [Path.LINETO] * (len(self.points) - 1)
        path = Path(tuple(self.points), codes)
        patch = patches.PathPatch(path, lw=2)
        self.image_ax.add_patch(patch)
        self.image_canvas.draw()

    def nextPolygon(self, aHandledEVENT):
        tmp = np.copy(self.points)
        tmp = np.vstack((tmp, tmp[0,:]))
        self.polygons.append(tmp)
        self.points = []

    def UpdateMask(self, aHandledEVENT):
        img = Image.new('L', (self.image.shape[1], self.image.shape[0]), 1)
        drw = ImageDraw.Draw(img, 'L')
        for p in self.polygons:
            p = p.ravel()
            drw.polygon(tuple(p), outline=1, fill=0)
        self.maskimage = np.array(img, dtype=bool)
        self.cropimage = True
        self.plotImage()

    def clearMask(self):
        self.cropimage = False
        self.plotImage()

    def plotImage(self, data=None):
        self.image_aspect = float(self.image.shape[1])/float(self.image.shape[0])
        self.image_dpi = self.image.shape[1]/6.0
        self.image_fig = Figure((6.0, 6.0/self.image_aspect), dpi=self.image_dpi, frameon=False)
        self.image_ax = self.image_fig.add_axes([0.0, 0.0, 1.0, 1.0,])
        if self.cropimage:
            cropped = np.copy(self.image)
            cropped[self.maskimage, :] = 0
            self.image_ax.imshow(cropped)
        else:
            self.image_ax.imshow(self.image)
        self.image_ax.get_xaxis().set_visible(False)
        self.image_ax.get_yaxis().set_visible(False)
        self.image_ax.grid(False)
        #grid_size = (960, 1280)
        #datax = np.random.randint(grid_size[0], size=(22, 1))
        #datay = np.random.randint(grid_size[1], size=(22, 1))
        #z = np.random.rand(22)
        #gridx, gridy = np.mgrid[0:grid_size[0], 0:grid_size[1]]
        #data = np.hstack((datax, datay))
        #data = np.float32(data)
        #gridz = griddata(data, z, (gridx, gridy), method='nearest')
        #if not(data is None):
        #    self.image_ax.hold(True)
        #    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        #    im = self.image_ax.imshow(gridz, cmap=cmap, alpha=0.5)
        #    self.image_fig.colorbar(im)

        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=self.tab2)
        self.image_canvas.draw()
        self.image_canvas.get_tk_widget().grid(row=0, column=2, columnspan=2,
                                    rowspan=4, padx=5, pady=5, sticky=NW)
    def overlayData(self):
        self.image_ax.hold(True)

root = Tk()
root.title("Welcome to the PoroMech GUI.")
app = Application(root)

root.mainloop()

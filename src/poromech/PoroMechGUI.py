from Tkinter import *
from ttk import Notebook
import tkFileDialog, tkMessageBox
from PoroMech import Data
import os
import pickle
import string
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from mpl_toolkits import axes_grid1
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
from numpy import ma
from scipy.ndimage import imread
from scipy.interpolate import Rbf, griddata
from scipy.stats import sem, t
from PIL import Image, ImageDraw
from collections import OrderedDict

sns.set()


class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.FileObjects = []
        self.FileObjectNames = []
        self.intSettings = {'Group': IntVar(value=1),
                            'DataObject': IntVar(value=1),
                            'FieldVariables': IntVar(value=1)}

        self.FieldVariables = OrderedDict()

        self.notebook = Notebook(self)
        self.tab1 = Frame(self.notebook)
        self.tab2 = Frame(self.notebook)
        self.notebook.add(self.tab1, text="Data Interaction")
        self.notebook.add(self.tab2, text="Mach-1 Image Grid")
        self.notebook.grid(row=0, column=0, sticky=NW)

        #####  BEGIN TAB 1 #####
        self.frameDataFiles = Frame(self.tab1)
        self.frameDataFiles.grid(row=0, column=0, sticky=N+W+E)
        self.buttonLoadFile = Button(self.frameDataFiles, text="Load Data File",
                                     command=self.loadFile)
        self.buttonLoadFile.grid(row=0, column=0, padx=1, pady=1,
                                 sticky=N+W+E)
        self.buttonRemoveFile = Button(self.frameDataFiles, text="Remove Selected File",
                                       command=self.removeDataObject)
        self.buttonRemoveFile.grid(row=1, column=0, padx=1, pady=1, sticky=N+W+E)
        self.frameDataObjects = LabelFrame(self.tab1, text="Data Files")
        self.frameDataObjects.grid(row=0, column=1, padx=1, pady=1, sticky=N+W+E)

        self.frameGroups = LabelFrame(self.tab1, text="Group Selection")
        self.frameGroups.grid(row=1, column=0, padx=1, pady=1,
                              sticky=N+W+E )
        Label(self.frameGroups, text="").grid(row=0, column=0, sticky=N+W+E)
        self.frameChannels = LabelFrame(self.tab1, text="Channel Selection")
        self.frameChannels.grid(row=1, column=1, padx=1, pady=1,
                                sticky=N+W+E)
        Label(self.frameChannels, text="").grid(row=0, column=0, sticky=N+W+E)

        self.frameTab1BottomLeft = Frame(self.tab1)
        self.frameTab1BottomLeft.grid(row=2, column=0, padx=1, pady=1, sticky=N+W+E)
        self.buttonSaveFile = Button(self.frameTab1BottomLeft, text="Save Selected to Pickle",
                                     command=self.saveFile)
        self.buttonSaveFile.grid(row=0, column=0, padx=1, pady=1,
                                 sticky=N+W+E)

        self.buttonSaveCSV = Button(self.frameTab1BottomLeft, text="Save Selected to CSV",
                                    command=self.saveCSV)
        self.buttonSaveCSV.grid(row=1, column=0, padx=1, pady=1, sticky=N+W+E)

        self.buttonGetThickness = Button(self.frameTab1BottomLeft, text="Get Mach-1 Thicknesses",
                                         command=self.findThicknesses)
        self.buttonGetThickness.grid(row=2, column=0, padx=1, pady=1, sticky=N+W+E)

        self.buttonPlot = Button(self.frameTab1BottomLeft, text="Plot Selected Channels",
                                 command=self.plotChannels)
        self.buttonPlot.grid(row=3, column=0, padx=1, pady=1,
                             sticky=N+W+E)

        self.frameTab1BottomRight = Frame(self.tab1)
        self.frameTab1BottomRight.grid(row=2, column=1, padx=1, pady=1, sticky=N+W+E)
        self.buttonMovingAvg = Button(self.frameTab1BottomRight, text="Apply Moving Average",
                                      command=self.applyMovingAvg)
        self.buttonMovingAvg.grid(row=0, column=0, padx=1, pady=1, columnspan=2,
                                  sticky=N+W+E)
        self.windowSize = IntVar(value=10)
        Label(self.frameTab1BottomRight, text="Window Size").grid(row=1, column=0, padx=1, pady=1,
                                                                  sticky=N+W)
        Entry(self.frameTab1BottomRight, textvariable=self.windowSize, width=4).grid(row=1, column=1, padx=1,
                                                                                     pady=1, sticky=N+W)
        #####  END TAB 1 #####

        ##### BEGIN TAB 2 #####
        self.frameImageButtons = Frame(self.tab2)
        self.frameImageButtons.grid(row=0, column=0, padx=1, pady=1, sticky=N+W+E)
        self.buttonLoadImage = Button(self.frameImageButtons, text="Load Image",
                                      command=self.loadImage)
        self.buttonLoadImage.grid(row=0, column=0, padx=1, pady=1, sticky=N+W+E)

        self.buttonLoadMapFile = Button(self.frameImageButtons, text="Load Mach-1 Site Locations",
                                        command=self.loadMachMap)

        self.buttonLoadMapFile.grid(row=1, column=0, padx=1, pady=1, sticky=N+W+E)
        self.buttonDefineMask = Button(self.frameImageButtons, text="Define Mask",
                                       command=self.cropImage)
        self.buttonDefineMask.grid(row=2, column=0, padx=1, pady=1, sticky=N+W+E)
        self.buttonClearMask = Button(self.frameImageButtons, text="Clear Mask",
                                      command=self.clearMask)
        self.buttonClearMask.grid(row=3, column=0, padx=1, pady=1, sticky=N+W+E)

        self.frameFieldVariables = LabelFrame(self.tab2, text="Field Variables")
        self.frameFieldVariables.grid(row=1, column=0, padx=1, pady=1, sticky=N+W+E)

        ##### END TAB 2 #####
        self.grid()

    def loadFile(self):
        self.filename = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Select a Data File.")
        if self.filename:
            self.FileObjects.append(Data(self.filename))
            self.FileObjectNames.append(os.path.basename(self.filename))
            for child in self.frameGroups.grid_slaves():
                child.grid_remove()
                del child
            self.intSettings["Group"].set(1)
            row = 0
            column = 0
            for i, g in enumerate(self.FileObjects[-1].groups):
                if i % 12 == 0:
                    row = 0
                    column += 1
                Radiobutton(self.frameGroups,
                            text=g,
                            indicatoron=0,
                            width=5,
                            variable=self.intSettings["Group"],
                            command=self.populateChannelList,
                            value=i+1).grid(row=row, column=column, sticky=NW)
                row += 1
            g = self.FileObjects[-1].groups[self.intSettings["Group"].get() - 1]
            for child in self.frameChannels.grid_slaves():
                child.grid_remove()
                del child
            row = 0
            column = 0
            self.channelSelections = {}
            for c in self.FileObjects[-1].time[g].keys():
                if i % 12 == 0:
                    row = 0
                    column += 1
                self.channelSelections[c] = IntVar(value=0)
                Checkbutton(self.frameChannels,
                            text=c,
                            variable=self.channelSelections[c]).grid(row=row, column=column, sticky=NW)
                row += 1
            counter = len(self.frameDataObjects.grid_slaves()) + 1
            Radiobutton(self.frameDataObjects,
                        text=self.FileObjectNames[-1],
                        indicatoron=0,
                        variable=self.intSettings["DataObject"],
                        command=self.selectDataObject,
                        value=counter).grid(row=counter, column=0, sticky=N+E+W)

    def selectDataObject(self):
        for child in self.frameGroups.grid_slaves():
            child.grid_remove()
            del child
        self.intSettings["Group"].set(1)
        row = 0
        column = 0
        for i, g in enumerate(self.FileObjects[self.intSettings["DataObject"].get()-1].groups):
            if i % 12 == 0:
                row = 0
                column += 1
            Radiobutton(self.frameGroups,
                        text=g,
                        indicatoron=0,
                        width=5,
                        variable=self.intSettings["Group"],
                        command=self.populateChannelList,
                        value=i+1).grid(row=row, column=column, sticky=NW)
            row += 1
        g = self.FileObjects[self.intSettings["DataObject"].get()-1].groups[self.intSettings["Group"].get() - 1]
        for child in self.frameChannels.grid_slaves():
            child.grid_remove()
            del child
        row = 0
        column = 0
        self.channelSelections = {}
        for c in self.FileObjects[self.intSettings["DataObject"].get()-1].time[g].keys():
            if i % 12 == 0:
                row = 0
                column += 1
            self.channelSelections[c] = IntVar(value=0)
            Checkbutton(self.frameChannels,
                        text=c,
                        variable=self.channelSelections[c]).grid(row=row, column=column, sticky=NW)
            row += 1

    def removeDataObject(self):
        if tkMessageBox.askyesno(message="Really remove the selected data?"):
            del self.FileObjects[self.intSettings["DataObject"].get()-1]
            del self.FileObjectNames[self.intSettings["DataObject"].get()-1]
            for child in self.frameDataObjects.grid_slaves():
                child.grid_remove()
                del child
            for i, o in enumerate(self.FileObjects):
                Radiobutton(self.frameDataObjects,
                            text=self.FileObjectNames[i],
                            indicatoron=0,
                            variable=self.intSettings["DataObject"],
                            command=self.selectDataObject,
                            value=i+1).grid(row=i, column=0, sticky=N+E+W)
            if len(self.FileObjects) > 0:
                self.intSettings["DataObject"].set(1)
            else:
                for child in self.frameGroups.grid_slaves():
                    child.grid_remove()
                    del child
                for child in self.frameChannels.grid_slaves():
                    child.grid_remove()
                    del child


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
                self.axes[i].plot(self.FileObjects[self.intSettings["DataObject"].get()-1].time[group][k],
                                  self.FileObjects[self.intSettings["DataObject"].get()-1].data[group][k])
                if self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses and k == "Fz, N":
                    try:
                        self.axes[i].axvline(x=self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses[group][1][0], color='r')
                        self.axes[i].axvline(x=self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses[group][1][1], color='g')
                    except:
                        pass
                self.axes[i].set_ylabel(k)
        else:
            self.axes.append(f.add_subplot(1,1,1))
            self.axes[0].plot(self.FileObjects[self.intSettings["DataObject"].get()-1].time[group][keys],
                              self.FileObjects[self.intSettings["DataObject"].get()-1].data[group][keys])
            if self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses and k == "Fz, N":
                self.axes[i].axvline(l=self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses[group][1][0], color='r')
                self.axes[i].axvline(l=self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses[group][1][1], color='g')
            self.axes[0].set_ylabel(keys)
        self.axes[-1].set_xlabel("Time (s)")
        canvas_frame = Frame(self.tab1)
        canvas_frame.grid(row=0, column=2, rowspan=4, sticky=N+W+E+S)
        canvas = FigureCanvasTkAgg(f, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,
                                    padx=1, pady=1, sticky=N+W+E+S)

        toolbar_frame = Frame(self.tab1)
        toolbar_frame.grid(row=4, column=2, sticky=NW)
        toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
        toolbar.update()

        Button(self.tab1, text="Crop", command=self.cropData).grid(
            row=4, column=2, sticky=NE)

    def findThicknesses(self):
        self.FieldVariables["Thicknesses"] = []
        for i, g in enumerate(self.FileObjects[self.intSettings["DataObject"].get()-1].groups):
            self.FileObjects[self.intSettings["DataObject"].get()-1].getThicknessMach1(g)
            self.FieldVariables["Thicknesses"].append(self.FileObjects[self.intSettings["DataObject"].get()-1].thicknesses[g][0])
        self.populateFieldVariableList()

    def cropData(self):
        group = self.FileObjects[self.intSettings["DataObject"].get()-1].groups[self.intSettings["Group"].get() - 1]
        (start, end) = self.axes[0].xaxis.get_view_interval()
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                self.FileObjects[self.intSettings["DataObject"].get()-1].windowData(group, c, start, end)
        self.populateChannelList()

    def populateChannelList(self):
        g = self.FileObjects[self.intSettings["DataObject"].get()-1].groups[self.intSettings["Group"].get() - 1]
        self.channelSelections = {}
        for child in self.frameChannels.grid_slaves():
            child.grid_remove()
            del child
        row = 0
        column = 0
        for i, c in enumerate(self.FileObjects[self.intSettings["DataObject"].get()-1].time[g].keys()):
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
            self.FileObjects[self.intSettings["DataObject"].get()-1].groups[self.intSettings["Group"].get() - 1],
            keys)

    def applyMovingAvg(self):
        group = self.FileObjects[self.intSettings["DataObject"].get()-1].groups[
            self.intSettings["Group"].get() - 1]
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                self.FileObjects[self.intSettings["DataObject"].get()-1].movingAverage(
                    group, c, win=self.windowSize.get())
        self.populateChannelList()

    def saveFile(self):
        group = self.FileObjects[self.intSettings["DataObject"].get()-1].groups[self.intSettings["Group"].get() - 1]
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                fid = open(os.path.abspath(string.replace(
                    self.filename, ".tdms", "_{:s}_{:s}.pkl".format(group, c))), "wb")
                pickle.dump((self.FileObjects[self.intSettings["DataObject"].get()-1].time[group][c], self.FileObjects[self.intSettings["DataObject"].get()-1].data[group][c]),
                            fid, 2)
                fid.close()

    def saveCSV(self):
        group = self.FileObjects[self.intSettings["DataObject"].get()-1].groups[self.intSettings["Group"].get() - 1]
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
                        self.FileObjects[self.intSettings["DataObject"].get()-1].time[group][c], self.FileObjects[self.intSettings["DataObject"].get()-1].data[group][c]):
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

    def loadMachMap(self):
        self.mapfile = tkFileDialog.askopenfilename(
            parent=root,
            initialdir=os.getcwd(),
            filetypes=[('Mach-1 map files', '.map')],
            title="Select the Mach-1 map file")
        if not self.mapfile:
            print("A file was not selected")
            return
        self.FileObjects[self.intSettings["DataObject"].get()-1].readMach1PositionMap(self.mapfile)
        self.maskFromMap()
        self.getTestLocations()

    def maskFromMap(self):
        ind = self.intSettings["DataObject"].get()-1
        self.polygons = []
        for p in self.FileObjects[ind].MachPositions["SubSurfaceID"].unique():
            points = self.FileObjects[ind].MachPositions.query('(SubSurfaceID == "{:s}") & (PointType == 1)'.format(p))[["PixelX", "PixelY"]]
            points = np.array(points)
            points = np.vstack((points, points[0,:]))
            self.polygons.append(points)
            self.UpdateMask()

    def getTestLocations(self):
        ind = self.intSettings["DataObject"].get()-1
        self.TestLocations = np.array(self.FileObjects[ind].MachPositions.query("(PointType == 0)")[["PixelX", "PixelY"]], dtype=float)

    def cropImage(self):
        self.points = []
        self.polygons = []
        self.image_canvas.get_tk_widget().bind("<Button-1>", self.XY_handler)
        self.image_canvas.get_tk_widget().bind("<Button-3>", self.nextPolygon)
        self.image_canvas.get_tk_widget().bind("<Return>", self.UpdateMask)
        #self.image_canvas.get_tk_widget().grid(row=0, column=0, padx=1, pady=1, sticky=N+E+W+S)
        self.image_canvas.draw()

    def XY_handler(self, aHandledEVENT):
        self.points.append((aHandledEVENT.x*self.screen2index, aHandledEVENT.y*self.screen2index))
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

    def UpdateMask(self, aHandledEVENT=None):
        img = Image.new('L', (self.image.shape[1], self.image.shape[0]), 1)
        drw = ImageDraw.Draw(img, 'L')
        for p in self.polygons:
            p = p.ravel()
            drw.polygon(tuple(p), outline=1, fill=0)
        self.maskimage = np.array(img, dtype=bool)
        self.cropimage = True
        if aHandledEVENT is None:
            self.image_canvas.get_tk_widget().unbind("<Button-1>")
            self.image_canvas.get_tk_widget().unbind("<Button-3>")
            self.image_canvas.get_tk_widget().unbind("<Return>")
        self.plotImage()

    def clearMask(self):
        self.cropimage = False
        self.plotImage()

    def populateFieldVariableList(self):
        for child in self.frameFieldVariables.grid_slaves():
            child.grid_remove()
            del child
        for i, (k, v) in enumerate(self.FieldVariables.items()):
            Radiobutton(self.frameFieldVariables,
                        text=k,
                        indicatoron=0,
                        variable=self.intSettings["FieldVariables"],
                        value=i+1).grid(row=i, column=0, sticky=N+E+W)
        try:
            self.buttonPlotOverlay
        except:
            self.buttonPlotOverlay = Button(self.frameImageButtons,
                                            text="Plot Selected",
                                            command=self.overlayData)
            self.buttonPlotOverlay.grid(row=4, column=0, sticky=N+W+E)

    def plotImage(self, data=None):
        try:
            self.image
        except:
            return
        self.imageFrame = Frame(self.tab2)
        self.image_width_inches = 6.0
        self.imageFrame.grid(row=0, column=1, padx=1, pady=1, sticky=N+E+W+S)
        self.image_aspect = float(self.image.shape[1])/float(self.image.shape[0])
        self.image_dpi = 96
        self.screen2index = self.image.shape[1] / (self.image_width_inches * self.image_dpi)
        self.image_fig = Figure((self.image_width_inches, self.image_width_inches/self.image_aspect), dpi=self.image_dpi, frameon=False)
        self.image_ax = self.image_fig.add_axes([0.0, 0.0, 1.0, 1.0,])
        self.image_ax.imshow(self.image)
        self.image_ax.get_xaxis().set_visible(False)
        self.image_ax.get_yaxis().set_visible(False)
        self.image_ax.grid(False)

        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=self.imageFrame)
        self.image_canvas.get_tk_widget().grid(row=0, column=0, padx=1, pady=1, sticky=N+E+W+S)
        self.image_canvas.get_tk_widget().config(width=self.image_width_inches*self.image_dpi,
                                                 height=self.image_width_inches*self.image_dpi/self.image_aspect)
        self.image_toolbar_frame = Frame(self.tab2)
        self.image_toolbar_frame.grid(row=1, column=1, sticky=NW)
        self.image_toolbar = NavigationToolbar2TkAgg(self.image_canvas, self.image_toolbar_frame)
        self.image_toolbar.update()
        self.image_canvas.draw()

    def overlayData(self):
        self.image_ax.hold(True)
        grid_size = self.image.shape[0:2]
        gridx, gridy = np.mgrid[0:grid_size[0], 0:grid_size[1]]
        key = self.FieldVariables.keys()[self.intSettings["FieldVariables"].get() - 1] 
        data = np.array(self.FieldVariables[key][-self.TestLocations.shape[0]:])
        m, se = np.mean(data), sem(data)
        h = se * t.ppf(0.975, data.size - 1)
        #rbf = Rbf(self.TestLocations[:,0], self.TestLocations[:,1], data, epsilon=2)
        #gridz = rbf(gridx, gridy)
        gridz = griddata(self.TestLocations[:, [1, 0]], data, (gridx, gridy), 'nearest')
        if self.cropimage:
            gridz = ma.masked_where(self.maskimage, gridz, copy=False)
            #gridz = ma.masked_where(np.abs((gridz - med))/mdev > 7.0, gridz, copy=False) 
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        im = self.image_ax.imshow(gridz, cmap=cmap, alpha=0.75,
                                  norm=colors.Normalize(vmin=data.min(), vmax=m+h, clip=False))

        self.image_fig.colorbar(im, shrink=0.75) 
        self.image_ax.scatter(self.TestLocations[:,0], self.TestLocations[:,1])
        text = [str(i+1) for i in xrange(self.TestLocations.shape[0])]
        for i, tt in enumerate(text):
            self.image_ax.text(self.TestLocations[i,0] + 10, self.TestLocations[i,1] - 10, tt, color="orange", size=8)
        self.image_canvas.draw()

root = Tk()
root.title("Welcome to the PoroMech GUI.")
app = Application(root)

root.mainloop()

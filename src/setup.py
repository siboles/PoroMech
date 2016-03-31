from setuptools import setup
from setuptools.dist import Distribution
from codecs import open
import os
import platform


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False
if "windows" in platform.system().lower():
    subprocess.call(["pip", "install", "numpy", "--index-url", "https://webdisk.ucalgary.ca/~ssibole/public_html/"])
    subprocess.call(["pip", "install", "-r", os.path.join(here,"requirements_win.txt")])
    if os.getenv("VIRTUAL_ENV") is not None:
        if os.getenv("TCL_LIBRARY") is None:
            path = os.getenv("PATH").lower().split(";")
            pythonpath = min(fnmatch.filter(path, "*python2*"), key=len)
            tcldir = os.path.join(pythonpath, "tcl")
            dirs = [name for name in os.listdir(tcldir) if os.path.isdir(os.path.join(tcldir, name))]
            tcldir = os.path.join(tcldir, fnmatch.filter(dirs, "tcl8.*")[0])
            subprocess.call("echo set TCL_LIBRARY={:s} >> {:s}".format(tcldir,
                                                                       os.path.join(os.getenv("VIRTUAL_ENV"),
                                                                                    "Scripts",
                                                                                    "activate.bat")), shell=True)
else:
    subprocess.call(["pip", "install", "-r", os.path.join(here,"requirements.txt")])
setup(
    name = 'poromech',
    version = '0.0',
    description = 'A python module for the analysis of mechanical testing data; particularly poroelastic materials.',
    packages = ['poromech'],
    url = "https://github.com/siboles/PoroMech",
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'GPL',
    py_modules = ['poromech.__init__','poromech.PoroMech','poromech.PoroMechGUI'],
    distclass=BinaryDistribution
)

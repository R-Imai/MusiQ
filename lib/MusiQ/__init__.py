# -*- coding: utf-8 -*-
less_package = False
try:
    import numpy
except ImportError:
    print("[error] Please install \"numpy\"")
    less_package = True

try:
    import scipy
except ImportError:
    print("[error] Please install \"scipy\"")
    less_package = True

try:
    import matplotlib
except ImportError:
    print("[error] Please install \"matplotlib\"")
    less_package = True

try:
    import pyaudio
except ImportError:
    print("[warning] The play function can not be used unless you install \"pyaudio\"")

try:
    import sklearn
except ImportError:
    print("[warning] The delta function can not be used unless you install \"sklearn\"")

if less_package:
    exit()

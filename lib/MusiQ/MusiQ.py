# -*- coding: utf-8 -*-
"""
音関係に関するあれこれ
"""
#----------------------------------
__author__ = "R.Imai"
__version__ = "2.1.0"
__created__ = "2016/01/14"
__date__ = "2017/05/29"
#----------------------------------
import __init__

import sys
import csv
import wave
from math import*

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal as sg



class data:
    def __init__(self, path, fs = 16000, graph = False):
        self.path = path
        self.fs = fs
        self.import_file(self.path, graph = graph)

        self.window_data = []
        self.fftsig = []
        self.fftfreq = []


    #-------------IO--------------
    def import_file(self, path, graph = False):
        """import sound file
        # arguments
            path: path of sound file
            graph: this arg is True, plot sound graph
        """
        if ".wav" in path:
            self.importWave(path, graph = graph)
        else:
            f = open(path, "rb")
            rawData = f.read()
            self.data = np.frombuffer(rawData, dtype="int16")
            if graph:
                plt.plot(self.data)
                plt.show()

    def importWave(self, path, graph = False):
        """import wave sound file
        #arguments
            path: path of wave sound file (*.wav)
            graph: this arg is True, plot sound graph
        """
        wave_file = wave.open(path,"r")
        x = wave_file.readframes(wave_file.getnframes())
        self.fs = wave_file.getframerate()
        self.data = np.frombuffer(x, dtype= "int16")
        if graph:
            plt.plot(self.data)
            plt.show()

    def outputCSV(self, data, path):
        """output csv file
        #arguments
            data: output data
            path: path of csv file
        """
        try:
            fp = open(path, 'w')
            csvWriter = csv.writer(fp, lineterminator='\n')
        except IOError:
            print (path + " cannot be opened.")
            exit()
        except Exception as e:
            print('type' + str(type(e)))
            exit()
        for elem in data:
            csvWriter.writerow(elem)
        print("save " + path)

    def outputWave(self, data, path):
        """output wave file
        #arguments
            data: output data
            path: path of wave file
        """
        write_wave = wave.Wave_write(path)
        data = data.astype(np.int16)
        write_wave.setparams((1,2,self.fs,len(data),"NONE", "not compressed"))
        write_wave.writeframes(data)
        write_wave.close()

    def transrate(self, path):
        """sound data output text data
        #arguments
            path: path output file
        """
        try:
            fp = open(path, 'w')
        except IOError:
            print (path + " cannot be opened.")
            exit()
        except Exception as e:
            print('type' + str(type(e)))
            exit()
        for i in range(len(self.data) - 1):
            fp.write(str(self.data[i]) + "\n")
        print("save " + path)

    def windowing(self, length, shift = 0, silent_cut = False, last_cut = True, cut_val = 1000, data = None):
        """cut window
        #argument
            length: window length
            shift: window shift [default: length]
            silent_cut: if this arg is True silent window cut
            last_cut: if this arg is False last odd window append
            cut_val: silent_cut threshold [default: 1000]
        """
        if data is None:
            data = self.data
        if shift == 0:
            shift = length
        if silent_cut:
            self.window_data = [data[i*shift : i*shift+length] for i in range((len(data) - length)//shift + 1) if max(abs(data[i*shift : i*shift+length])) > cut_val]
        else:
            self.window_data = [data[i*shift : i*shift+length] for i in range((len(data) - length)//shift + 1)]
        if not(last_cut):
            elem = list(data[((len(data) - length)//shift)*shift + length + 1:]) + [0 for i in range(length)]
            self.window_data.append(np.array(elem[:length]))
        return self.window_data


    def play(self, data = None):
        """sound play
        this method need pyaudio
        """
        import pyaudio

        if data is None:
            data = self.data
        p = pyaudio.PyAudio()
        stream = p.open(format = p.get_format_from_width(2), channels = 1, rate = self.fs, output = True)
        chunk = 1024
        cnt = 1
        Pdata = data[(cnt - 1)*chunk : cnt*chunk + chunk]
        while stream.is_active():
            stream.write(Pdata)
            cnt += 1
            Pdata = data[(cnt - 1)*chunk : cnt*chunk + chunk]
            #print(str((cnt - 1)*chunk) + " : " + str(cnt*chunk - chunk - 1))
            if len(Pdata) == 0:
                stream.stop_stream()
        stream.close()
        p.terminate()

    #-----------/IO-----------

#------------filter-----------
def _sinc(x):
    re = 1.0
    if not(x == 0.0):
        re = np.sin(x) / x
    return re

def fir(data, fil_coe):
    y = [0.0] * len(data)  # フィルタの出力信号
    N = len(fil_coe) - 1      # フィルタ係数の数
    for n in range(len(data)):
        for i in range(N+1):
            if n - i >= 0:
                y[n] += fil_coe[i] * data[n - i]
    return y

def bpf2(fs, cf, bw, delta = 100.0):
    """band pass filter ver.2
    #argument
        fs: frequency
        cf: center frequency
        bw: band width
    """
    delta = delta/(fs/2)
    fe1 = cf/fs - (bw/fs)/2
    fe2 = cf/fs + (bw/fs)/2
    N = round(3.1 / delta) - 1
    if (N + 1) % 2 == 0:
        N += 1
    N = int(N)
    b = []
    for i in range(int(-N/2), int(N/2 + 1)):
        b.append(2 * fe2 * self._sinc(2 * pi * fe2 * i) - 2 * fe1 * self._sinc(2 * pi * fe1 * i))
    hanningWindow = np.hanning(N + 1)
    for i in range(len(b)):
        b[i] *= hanningWindow[i]
    return b

def bpf(fs, data, cf, bw):
    """band pass filter
    #argument
        fs: frequency
        data: sound data
        cf: center frequency
        bw: band width
    """
    nyq = self.fs / 2.0
    fe1 = (cf - bw/2)/nyq
    fe2 = (cf + bw/2)/nyq

    if len(data)%2 == 0:
        length = len(data) - 1
    else:
        length = len(data)
    sybpf = sg.firwin(length, [fe1, fe2], pass_zero=False)
    sig = sg.lfilter(sybpf, 1, data)

    return sig

def gabor(length, center, width, K = 1, phi = 0, fs = 16000):
    """band pass filter ver.2
    #argument
        fs: frequency
        cf: center frequency
        bw: band width
    """
    print("length = " + str(length) + " center = " + str(center) + " width = " + str(width) )
    width = width/5500
    sigma = sqrt(2*pi)/width
    ga = sg.gaussian(length, sigma)
    sinWave = np.array([cos(2*pi*i*center/fs) for i in range(0, length)])
    filt = ga * sinWave
    return filt
#-----------/filter-----------

#------------delta-----------
def delta(data):
    """calc delta param
    # argument
        data: data for which you want to calculate the delta
    # return
        numpy array two dimensions
    """
    from sklearn import linear_model
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.array([data])
    data = data.T
    re = []
    for flame in data:
        re_elem = []
        clf = linear_model.LinearRegression()
        flame = [flame[0],flame[0]] + list(flame) + [flame[-1],flame[-1]]
        for c in range(2,len(flame)-2):
            elem = flame[c-2:c+3]
            times = [[i+1]for i in range(len(elem))]
            clf.fit(times, elem)
            re_elem.append(clf.coef_[0])
        re.append(np.array(re_elem))

    return np.array(re).T
#------------/delta-----------

#------------fft-----------
def fft(mq_data, data = None, fs = None, graph = False, save_name = None):
    """fast fourier transform
    #arguments
        mq_data: [MusiQ.data]
        data: if you want to apply FFT to something other than data.window_data, write it here
        fs: sampling rate
        graph: this arg is True, plot sound graph
        save_name: save graph name
    """
    if fs is None:
        fs = mq_data.fs
    if data is None:
        if len(mq_data.window_data) != 0:
            data = mq_data.window_data
        else:
            print("[error] please make window at MusiQ.data.divide() or please pass the data to the argument 'data'")
            exit()
    else:
        if len(np.array(data).shape) == 1:
            data = [data]

    fftsig = []
    fftfreq = []
    cnt = 0
    for elem in data:
        win_func = sg.hamming(len(elem))
        win_sig = elem*win_func
        sig = np.abs(fftpack.fft(win_sig))
        fftsig.append(sig[:len(sig)/2])
        freq = fftpack.fftfreq(len(sig), d = 1.0 / fs)
        fftfreq.append(freq[:len(sig)/2])

        if graph or not(save_name is None):
            plt.plot(freq[:len(sig)/2], sig[:len(sig)/2])
            #plt.ylim(0, 5000000)
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude spectrum")
            if save_name is None:
                plt.show()
            else:
                plt.savefig(save_name + str(cnt) +".png")
                plt.close()
                cnt += 1

    return fftfreq, fftsig
#-----------/fft-----------

#------------cepstrum-----------
def cepstrum(mq_data, dim = 10, graph = False, save_name = None):
    """cepstrum
    #arguments
        mq_data: [MusiQ.data]
        dim: dimension
        graph: this arg is True, plot sound graph
        save_name: save graph name
    """
    ceps = []
    highCeps = []
    fftfreq, fftsig = fft(mq_data)
    cnt = 0
    for spec, freq in zip(fftsig, fftfreq):
        AdftLog = 20 * np.log10(spec)
        cps = np.real(fftpack.ifft(AdftLog))
        cpsLif = np.array(cps)
        cpsLifLif = np.array(cps)
        cpsLif[dim:len(cpsLif) - dim + 1] = 0
        cpsLifLif[0:len(cpsLifLif)/2 - dim/2] = 0
        cpsLifLif[len(cpsLifLif)/2 + dim/2 + 1:len(cpsLif) - 1] = 0
        dftSpc = np.real(fftpack.fft(cpsLif))
        ceps.append(dftSpc[:len(freq)/2])
        dftSpcSpc = np.real(fftpack.fft(cpsLifLif))
        highCeps.append(dftSpcSpc[:len(freq)/2])
        if graph or not(save_name is None):
            plt.plot(freq[:len(freq)/2], AdftLog[:len(freq)/2])
            plt.plot(freq[:len(freq)/2], dftSpc[:len(freq)/2], color="red")
            #plt.plot(freq[:len(freq)/2], dftSpcSpc[:len(freq)/2], color="green")
            plt.xlabel("frequency [Hz]")
            plt.ylabel("log amplitude spectrum")
            if save_name is None:
                plt.show()
            else:
                plt.savefig(save_name + str(cnt) +".png")
                plt.close()
                cnt += 1

    return ceps, highCeps
#-----------/cepstrum-----------

#------------mfcc-----------
def preEmphasis(data, p = 0.97):
    """pre enphasis filter
    #arguments
        data: sound data
        p: coefficient
    """
    return sg.lfilter([1.0, -p], 1, data)

def hz2mel(f):
    """transe Hz to mel
    #arguments
        f: frequency
    """
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """transe mel to Hz
    #arguments
        mel: mel
    """
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(mq_data, nfft, numChannels):
    """mel filter bank
    #arguments
        mq_data: [MusiQ.data]
        nfft: num fft
        numChannels: cannnel number
    """
    fmax = mq_data.fs / 2
    melmax = hz2mel(fmax)
    nmax = floor(nfft / 2)
    df = mq_data.fs / nfft
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters


def mfcc(mq_data, dim = 20 ,mel = 20 ,graph = False, save_name = None, preEm = True):
    """mel frequency cepstrum coefficient
    #arguments
        mq_data: [MusiQ.data]
        dim: dimension
        mel: channel number of mel filter bank
        graph: this arg is True, plot sound graph
        save_name: save graph name
    """
    data = mq_data.window_data
    mfcc = []
    cnt = 0
    for windata in data:
        preEmData = windata
        if preEm:
            preEmData = preEmphasis(windata, 0.97)
        nfft = len(preEmData)
        winFunc = sg.hamming(len(preEmData))
        wavdata = preEmData * winFunc
        spec = np.abs(fftpack.fft(wavdata))[:nfft/2]
        fscale = fftpack.fftfreq(nfft, d = 1.0 / mq_data.fs)[:nfft/2]

        # メルフィルタバンクを作成
        df = mq_data.fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
        filterbank, fcenters = melFilterBank(mq_data, nfft, mel)

        # 振幅スペクトルにメルフィルタバンクを適用
        mspec = np.log10(np.dot(spec, filterbank.T))

        # 離散コサイン変換
        ceps = fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)[:dim]
        mfcc.append(ceps)

        if graph or not(save_name is None):
            plt.subplot(311)
            plt.plot(fscale, np.log10(spec))
            plt.xlabel("frequency")

            plt.subplot(312)
            plt.plot(fcenters, mspec, "o-")
            plt.xlabel("frequency")

            plt.subplot(313)
            plt.plot(ceps)
            plt.xlabel("frequency")
            if save_name is None:
                plt.show()
            else:
                plt.savefig(save_name + str(cnt) +".png")
                plt.close()
                cnt += 1
    return mfcc
#------------/mfcc-----------

#------------lpc-----------
def _autocor(x, nlags=None):
    """calc autocorrelation
    """
    N = len(x)
    if nlags is None:
        nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r

def _levDur(R, order):
    """Levinson Durbin Recursion
    """
    a = np.zeros(order + 1)
    E = np.zeros(order + 1)
    a[0] = 1
    a[1] = -R[1] / R[0]
    E[1] = R[0] + a[1] * R[1]

    for n in range(1,order):
        lam = 0
        for j in range(n + 1):
            lam -= a[j] * R[n - j + 1]
        lam /= E[n]

        U = np.array(a[:n + 2])
        V = np.array((a[:n + 2])[::-1])

        for (i,elem) in enumerate(U + lam*V):
            a[i] = elem
        E[n + 1] = (1 - lam**2)*E[n]

    return a,E


def lpc(mq_data, order = 32, graph = False, save_name = None, debug = False, nfft = 2048):
    """Linear Predictive Coding
    #arguments
        mq_data: [MusiQ.data]
        order: lpc order
        graph: this arg is True, plot sound graph
        save_name: save graph name
    """
    data = mq_data.window_data
    lpcData = []
    lpc_fscale = []
    cnt = 0
    for sig in data:
        preEm = preEmphasis(sig, p = 0.97) * sg.hamming(len(sig))
        r = _autocor(preEm, order + 1)
        a, e = _levDur(r, order)

        #nfft = 2048#len(preEm)
        fscale = np.fft.fftfreq(nfft, d = 1.0 / mq_data.fs)[:nfft/2]
        lpc_fscale.append(fscale)
        freqsig = np.abs(fftpack.fft(preEm,nfft))
        logspec = 20 * np.log10(freqsig)

        w, h = sg.freqz(np.sqrt(e), a, nfft, "whole")
        lpcspec = np.abs(h)
        loglpcspec = 20 * np.log10(lpcspec)
        lpcData.append(loglpcspec[:nfft/2])
        if graph or not(save_name is None):
            if debug:
                plt.subplot(211)
                plt.plot(sig)
                plt.subplot(212)
            plt.plot(fscale, logspec[:nfft/2])
            plt.plot(fscale, loglpcspec[:nfft/2], "r", linewidth=2)

            plt.xlim((0, 10000))
            if save_name is None:
                plt.show()
            else:
                plt.savefig(save_name + str(cnt) +".png")
                plt.close()
                cnt += 1

    return lpcData, lpc_fscale
#-----------/lpc-----------

#-----------formant--------
def get_formant(lpc, lpc_f):
    fol = []
    for i in range(1,len(lpc) - 1):
        if max(lpc[i-1:i+2]) == lpc[i]:
            fol.append(lpc_f[i])
    return fol

def formant(mq_data):
    lpcData, lpc_fscale = lpc(mq_data)
    formant_data = []

    for l, lf in zip(lpcData, lpc_fscale):
        f_ = get_formant(l, lf)
        formant_data.append(f_[:5])
    return formant_data
#-----------/formant-------







if __name__ == '__main__':
    pass

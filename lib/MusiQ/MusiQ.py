# -*- coding: utf-8 -*-
"""
音関係に関するあれこれ
"""
#----------------------------------
__author__ = "R.Imai"
__version__ = "1.1.0"
__created__ = "2016/04/09"
__date__ = "2017/01/18"
#----------------------------------

import sys
import csv
import wave
from math import*

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal as sg



class MusiQ:
    def __init__(self, filename, fs = 16000, graph = False):
        self.filename = filename
        self.fs = fs
        self.importFile(self.filename, graph = graph)
        self.windowData = []

    #-------------IO--------------
    def importFile(self, rawfile, graph = False):
        if ".wav" in rawfile:
            self.importWave(rawfile, graph = graph)
        else:
            f = open(rawfile, "rb")
            rawData = f.read()
            self.data = np.frombuffer(rawData, dtype="int16")
            if graph:
                plt.plot(self.data)
                plt.show()

    def importWave(self, wavfile, graph = False):
        wave_file = wave.open(wavfile,"r") #Open
        x = wave_file.readframes(wave_file.getnframes()) #frameの読み込み
        self.fs = wave_file.getframerate()
        self.data = np.frombuffer(x, dtype= "int16")#numpy.arrayに変換
        if graph:
            plt.plot(self.data)
            plt.show()

    def outputCSV(self, data, filename):
        try:
            fp = open(filename + ".csv", 'w')
            csvWriter = csv.writer(fp, lineterminator='\n')
        except IOError:
            print (filename + " cannot be opened.")
            exit()
        except Exception as e:
            print('type' + str(type(e)))
            exit()
        for elem in data:
            csvWriter.writerow(elem)
        print("save " + filename +".csv")

    def outputWave(self, data, filename):
        write_wave = wave.Wave_write(filename)
        data = data.astype(np.int16)
        write_wave.setparams((1,2,self.fs,len(data),"NONE", "not compressed"))
        write_wave.writeframes(data)
        write_wave.close()

    def transrate(self, filename):
        try:
            fp = open(filename, 'w')
        except IOError:
            print (filename + " cannot be opened.")
            exit()
        except Exception as e:
            print('type' + str(type(e)))
            exit()
        for i in range(len(self.data) - 1):
            fp.write(str(self.data[i]) + "\n")
        print("save " + filename)

    def divide(self, length, shift = 0, silentCut = False, lastCut = False, cut_val = 1000):
        self.windowData = []
        if shift == 0:
            shift = length
        for i in range(len(self.data)//shift):
            if silentCut:
                if max(self.data[i*shift : i*shift + length]) > cut_val:
                    self.windowData.append(self.data[i*shift : i*shift + length].astype(np.int64))
            else:
                self.windowData.append(self.data[i*shift : i*shift + length].astype(np.int64))
        if not(lastCut):
            if not(len(self.data[len(self.data)//shift*shift : i*shift + length]) == 0):
                if silentCut:
                    if max(self.data[len(self.data)//shift*shift : i*shift + length]) > cut_val:
                        self.windowData.append(self.data[len(self.data)//shift*shift : i*shift + length].astype(np.int64))
                else:
                    self.windowData.append(self.data[len(self.data)//shift*shift : i*shift + length].astype(np.int64))

    def play(self, data = None):
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
    def sinc(self, x):
        re = 1.0
        if not(x == 0.0):
            re = np.sin(x) / x
        return re

    def fir(self, x, b):
        y = [0.0] * len(x)  # フィルタの出力信号
        N = len(b) - 1      # フィルタ係数の数
        for n in range(len(x)):
            for i in range(N+1):
                if n - i >= 0:
                    y[n] += b[i] * x[n - i]
        return y

    def bpf2(self, CF, BW, delta = 100.0):
        delta = delta/(self.fs/2)
        fe1 = CF/self.fs - (BW/self.fs)/2
        fe2 = CF/self.fs + (BW/self.fs)/2
        N = round(3.1 / delta) - 1
        if (N + 1) % 2 == 0:
            N += 1
        N = int(N)
        b = []
        for i in range(int(-N/2), int(N/2 + 1)):
            b.append(2 * fe2 * self.sinc(2 * pi * fe2 * i) - 2 * fe1 * self.sinc(2 * pi * fe1 * i))
        hanningWindow = np.hanning(N + 1)
        for i in range(len(b)):
            b[i] *= hanningWindow[i]
        return b

    def bpf(self, data, cf, bw):

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

    def gabor(self, length, center, width, K = 1, phi = 0, fs = 16000):
        print("length = " + str(length) + " center = " + str(center) + " width = " + str(width) )
        width = width/5500
        sigma = sqrt(2*pi)/width
        ga = sg.gaussian(length, sigma)
        sinWave = np.array([cos(2*pi*i*center/fs) for i in range(0, length)])
        filt = ga * sinWave
        return filt
    #-----------/filter-----------

    #------------fft-----------
    def fft(self, data = None, fs = None, graph = False, saveName = None, sig = None):
        if fs is None:
            fs = self.fs

        if data is None:
            self.fftsig = []
            self.fftfreq = []
            if not len(self.windowData) == 0:
                cnt = 0
                for sig in self.windowData:
                    winFunc = sg.hamming(len(sig))  #窓関数
                    winSig = sig * winFunc
                    sig = np.abs(fftpack.fft(winSig))
                    self.fftsig.append(sig)#[:len(sig)/2])
                    freq = fftpack.fftfreq(len(sig), d = 1.0 / self.fs)
                    #freq = freq[:len(freq)/2]
                    self.fftfreq.append(freq)#[:len(freq)/2])
                    if graph or not(saveName == None):
                        plt.subplot(111)
                        plt.plot(freq[:len(sig)/2], sig[:len(sig)/2])
                        plt.ylim(0, 5000000)
                        plt.xlabel("frequency [Hz]")
                        plt.ylabel("amplitude spectrum")
                        if saveName == None:
                            plt.show()
                        else:
                            plt.savefig(saveName + str(cnt) +".png")
                            plt.close()
                            cnt += 1

        else:
            winFunc = sg.hamming(len(data))  #窓関数
            winSig = data * winFunc
            sig = np.abs(fftpack.fft(winSig))

            freq = fftpack.fftfreq(len(data), d = 1.0 / fs)
            #freq = freq[:len(freq)/2]
            if graph or not(saveName == None):
                plt.subplot(111)
                plt.plot(freq[:len(sig)/2], sig[:len(sig)/2])
                #plt.ylim(0, 5000000)
                plt.xlabel("frequency [Hz]")
                plt.ylabel("amplitude spectrum")
                if saveName == None:
                    plt.show()
                else:
                    plt.savefig(saveName + str(cnt) +".png")
                    plt.close()
                    cnt += 1

            return sig[:len(sig)/2], freq[:len(sig)/2]
    #-----------/fft-----------

    #------------cepstrum-----------
    def cepstrum(self, dim = 10,graph = False):
        self.ceps = []
        self.highCeps = []
        for spec, freq in zip(self.fftsig, self.fftfreq):
            AdftLog = 20 * np.log10(spec)
            cps = np.real(fftpack.ifft(AdftLog))
            cpsLif = np.array(cps)   # arrayをコピー
            cpsLifLif = np.array(cps)
            # 高周波成分を除く（左右対称なので注意）
            cpsLif[dim:len(cpsLif) - dim + 1] = 0
            # 高周波を通す。
            cpsLifLif[0:len(cpsLifLif)/2 - dim/2] = 0
            cpsLifLif[len(cpsLifLif)/2 + dim/2 + 1:len(cpsLif) - 1] = 0
            dftSpc = np.real(fftpack.fft(cpsLif))
            self.ceps.append(dftSpc[:len(freq)/2])
            dftSpcSpc = np.real(fftpack.fft(cpsLifLif))
            self.highCeps.append(dftSpcSpc[:len(freq)/2])
            if graph:
                plt.plot(freq[:len(freq)/2], AdftLog[:len(freq)/2])
                # 高周波成分を除いた声道特性のスペクトル包絡を重ねて描画
                plt.plot(freq[:len(freq)/2], dftSpc[:len(freq)/2], color="red")
                #plt.plot(freq[:len(freq)/2], dftSpcSpc[:len(freq)/2], color="green")
                plt.xlabel("frequency [Hz]")
                plt.ylabel("log amplitude spectrum")
                plt.show()
    #-----------/cepstrum-----------

    #------------mfcc-----------
    def preEmphasis(self, data, p = 0.97):
        return sg.lfilter([1.0, -p], 1, data)

    def hz_Mel(self,f):
        #Hzをmelに変換
        return 1127.01048 * np.log(f / 700.0 + 1.0)

    def mel_Hz(self,m):
        #melをhzに変換
        return 700.0 * (np.exp(m / 1127.01048) - 1.0)

    def melFilterBank(self, nfft, numChannels):
        fmax = self.fs / 2
        melmax = self.hz_Mel(fmax)
        nmax = floor(nfft / 2)
        df = self.fs / nfft
        dmel = melmax / (numChannels + 1)
        melcenters = np.arange(1, numChannels + 1) * dmel
        fcenters = self.mel_Hz(melcenters)
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


    def mfcc(self,data = None, dim = 20 ,mel = 20 ,graph = False, preEm = True):
        fg = data
        if data is None:
            data = self.windowData
        mfcc = []
        for windata in data:
            preEmData = windata
            if preEm:
                preEmData = self.preEmphasis(windata, 0.97)
            nfft = len(preEmData)
            winFunc = sg.hamming(len(preEmData))
            wavdata = preEmData * winFunc
            spec = np.abs(fftpack.fft(wavdata))[:nfft/2]
            fscale = fftpack.fftfreq(nfft, d = 1.0 / self.fs)[:nfft/2]

            # メルフィルタバンクを作成
            df = self.fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
            filterbank, fcenters = self.melFilterBank(nfft, mel)

            # 振幅スペクトルにメルフィルタバンクを適用
            mspec = np.log10(np.dot(spec, filterbank.T))

            # 離散コサイン変換
            ceps = fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)[:dim]
            mfcc.append(ceps)

            if graph:
                plt.subplot(311)
                plt.plot(fscale, np.log10(spec))
                plt.xlabel("frequency")

                plt.subplot(312)
                plt.plot(fcenters, mspec, "o-")
                plt.xlabel("frequency")

                plt.subplot(313)
                plt.plot(ceps)
                plt.xlabel("frequency")
                plt.show()
        if fg is None:
            self.mfccData = mfcc
        return mfcc
    #------------/mfcc-----------

    #------------lpc-----------
    def autocor(self, x, nlags=None):
        N = len(x)
        if nlags == None: nlags = N
        r = np.zeros(nlags)
        for lag in range(nlags):
            for n in range(N - lag):
                r[lag] += x[n] * x[n + lag]
        return r

    def levDur(self, R, order):
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


    def lpc(self,data = None, order = 32, graph = False, debug = False, nfft = 2048):
        if data is None:
            data = self.windowData
        self.lpcData = []
        self.lpc_fscale = []
        for sig in data:
            preEm = self.preEmphasis(sig, p = 0.97) * sg.hamming(len(sig))
            r = self.autocor(preEm, order + 1)
            a, e = self.levDur(r, order)

            #nfft = 2048#len(preEm)
            fscale = np.fft.fftfreq(nfft, d = 1.0 / self.fs)[:nfft/2]
            self.lpc_fscale.append(fscale)
            freqsig = np.abs(fftpack.fft(preEm,nfft))
            logspec = 20 * np.log10(freqsig)

            w, h = sg.freqz(np.sqrt(e), a, nfft, "whole")
            lpcspec = np.abs(h)
            loglpcspec = 20 * np.log10(lpcspec)
            self.lpcData.append(loglpcspec[:nfft/2])
            if graph:
                if debug:
                    plt.subplot(211)
                    plt.plot(sig)
                    plt.subplot(212)
                plt.plot(fscale, logspec[:nfft/2])
                plt.plot(fscale, loglpcspec[:nfft/2], "r", linewidth=2)

                plt.xlim((0, 10000))
                plt.show()
    #-----------/lpc-----------

    #-----------formant--------
    def get_formant(self, lpc, lpc_f):
        fol = []
        for i in range(1,len(lpc) - 1):
            if max(lpc[i-1:i+2]) == lpc[i]:
                fol.append(lpc_f[i])
        return fol

    def formant(self):
        self.lpc()
        self.formant = []

        for l, lf in zip(self.lpcData, self.lpc_fscale):
            f_ = get_formant(l, lf)
            self.formant.append(f_[:5])
    #-----------/formant-------







if __name__ == '__main__':
    music = MusiQ(sys.argv[1])
    #music.play()
    music.divide(11000,5500,silentCut = True)
    music.lpc(order = 32, graph = True)

    """bpsig = music.bpf(music.windowData[10], 1100, 2000)

    sig, freq = music.fft(data = bpsig, fs = 16000)
    orisig, orifreq = music.fft(data = music.windowData[10], fs = 16000)

    plt.plot(freq[:len(freq)/2], sig[:len(sig)/2], label = "processing")
    plt.plot(orifreq[:len(freq)/2], orisig[:len(sig)/2], label = "original")
    plt.legend(loc='best')
    plt.show()"""


    #music.lpc(16,graph = True)
    #music.mfcc(dim = 12, graph = True)
    #music.outputCSV(music.mfcc, "test")
    #music.fft()
    #music.cepstrum(glaph = True)
    #print(len(music.fftsig[0]))
    #print(len(music.data))
    #print(str(len(music.windowData))+"::"+str(len(music.windowData[0])))
    #print(len(music.windowData[0]))
    #print(len(music.windowData[49]))
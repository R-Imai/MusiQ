#coding:utf-8
#-------------------------------------------------------------------------------
#   Name:		MusiQ.py
#	Author:		R.Imai
#	Created:	2016 / 04 / 09
#	Last Date:	2016 / 04 / 12
#	Note:
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import wave
import sys
from scipy import fftpack
from scipy import signal as sg
import csv

argv = sys.argv
argNum = len(sys.argv)


class MusiQ:
    def __init__(self, filename, fs = 16000, graph = False):
        self.filename = filename
        self.importFile(self.filename, graph = graph)
        self.windowData = []
        self.fs = fs

    def importFile(self, rawfile, graph = False):
        f = open(rawfile, "rb")
        rawData = f.read()
        self.data = np.frombuffer(rawData, dtype="int16")
        if graph:
            plt.plot(self.data)
            plt.show()

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

    def divide(self, length, shift = 0):
        if shift == 0:
            shift = length
        for i in range(len(self.data)//shift):
            #print(self.data[i*shift : i*shift + length])
            self.windowData.append(self.data[i*shift : i*shift + length])

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
                    self.fftfreq.append(freq)
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
            fftsig = np.abs(fftpack.fft(winSig))#[:len(sig)/2])
            freq = fftpack.fftfreq(len(sig), d = 1.0 / fs)
            #freq = freq[:len(freq)/2]

            return fftsig, freq


    def cepstrum(self, dim = 10,glaph = False):
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
            self.ceps.append(dftSpc)
            dftSpcSpc = np.real(fftpack.fft(cpsLifLif))
            self.highCeps.append(dftSpcSpc)

            if glaph:
                plt.plot(freq[:len(freq)/2], AdftLog[:len(freq)/2])
                # 高周波成分を除いた声道特性のスペクトル包絡を重ねて描画
                plt.plot(freq[:len(freq)/2], dftSpc[:len(freq)/2], color="red")
                #plt.plot(freq[:len(freq)/2], dftSpcSpc[:len(freq)/2], color="green")
                plt.xlabel("frequency [Hz]")
                plt.ylabel("log amplitude spectrum")

                plt.show()

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
        nmax = nfft / 2
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


    def mfcc(self,dim = 20 ,mel = 20 ,graph = False):
        self.mfcc = []
        for windata in self.windowData:
            preEmData = self.preEmphasis(windata, 0.97)
            nfft = len(windata)
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
            self.mfcc.append(ceps)

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


    def lpc(self, order, graph = False):
        self.lpc = []
        for sig in self.windowData:
            preEm = self.preEmphasis(sig, p = 0.97) * sg.hamming(len(sig))
            r = self.autocor(preEm,order + 1)
            a, e = self.levDur(r, order)

            nfft = len(preEm)
            fscale = np.fft.fftfreq(nfft, d = 1.0 / 16000)[:nfft/2]

            freqsig = np.abs(fftpack.fft(preEm))
            logspec = 20 * np.log10(freqsig)

            w, h = sg.freqz(np.sqrt(e), a, nfft, "whole")
            lpcspec = np.abs(h)
            loglpcspec = 20 * np.log10(lpcspec)

            if graph:
                plt.plot(fscale, logspec[:nfft/2])
                plt.plot(fscale, loglpcspec[:nfft/2], "r", linewidth=2)

                plt.xlim((0, 10000))
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





if __name__ == '__main__':
    music = MusiQ(sys.argv[1],graph = True)
    #music.divide(256,128)
    #music.lpc(16)
    #music.mfcc(dim = 12, graph = True)
    #music.outputCSV(music.mfcc, "test")
    #music.fft()
    #music.cepstrum(glaph = True)
    #print(len(music.fftsig[0]))
    #print(len(music.data))
    #print(str(len(music.windowData))+"::"+str(len(music.windowData[0])))
    #print(len(music.windowData[0]))
    #print(len(music.windowData[49]))

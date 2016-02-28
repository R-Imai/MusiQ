#coding:utf-8
import numpy as np
import pylab
import wave
import sys
from scipy import fftpack
from scipy import signal as sg
import csv

argv = sys.argv
argNum = len(sys.argv)

def importWave(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    data = wf.readframes(wf.getnframes())
    data = np.frombuffer(data, dtype="int16") / 32768.0  # (-1, 1)に正規化
    wf.close()
    return data, float(fs)

def fft(data,n):
    winFunc = sg.hamming(len(data))  #窓関数
    wavdata = data * winFunc
    sig = fftpack.fft(wavdata, n)
    freq = fftpack.fftfreq(n, d = 1.0 / fs)
    return sig,freq


def cepstrum(data,fs,dim,glaph = False):
    t = np.arange(0.0, len(data) / fs, 1/fs)
    center = len(data) / 2  # 中心のサンプル番号
    cuttime = 0.04         # 切り出す長さ [s]
    wavdata = data[center - cuttime/2*fs : center + cuttime/2*fs]
    time = t[center - cuttime/2*fs : center + cuttime/2*fs]
    n = 2048
    dft, freq = fft(wavdata, n)

    Adft = np.abs(dft)              # 振幅スペクトル
    Pdft = np.abs(dft) ** 2         # パワースペクトル

    AdftLog = 20 * np.log10(Adft)    # 対数振幅スペクトル
    PdftLog = 10 * np.log10(Pdft)    # 対数パワースペクトル

    cps = np.real(fftpack.ifft(AdftLog))
    quefrency = time - min(time)

    cpsLif = np.array(cps)   # arrayをコピー
    cpsLifLif = np.array(cps)
    # 高周波成分を除く（左右対称なので注意）
    cpsLif[dim:len(cpsLif) - dim + 1] = 0
    # 高周波を通す。
    cpsLifLif[0:dim] = 0
    cpsLifLif[len(cpsLif) - dim + 1:len(cpsLif) - 1] = 0

    dftSpc = np.real(fftpack.fft(cpsLif, n))
    dftSpcSpc = np.real(fftpack.fft(cpsLifLif, n))

    if glaph:
        pylab.plot(freq[0:n/2], AdftLog[0:n/2])
        # 高周波成分を除いた声道特性のスペクトル包絡を重ねて描画
        pylab.plot(freq[0:n/2], dftSpc[0:n/2], color="red")
        pylab.plot(freq[0:n/2], dftSpcSpc[0:n/2], color="green")
        pylab.xlabel("frequency [Hz]")
        pylab.ylabel("log amplitude spectrum")
        pylab.xlim(0, 5000)

        pylab.show()

    return dftSpc

def outputCSV(data,filename):
    try:
        fp = open(filename + ".csv", 'w')
    except IOError:
        print (argv[2] + " cannot be opened.")
        exit()
    except Exception as e:
        print('type' + str(type(e)))
        exit()
    for i in range(len(data) - 1):
        fp.write(str(data[i]) + "\n")
    print("save " + filename +".csv")





if __name__ == '__main__':
    wavdata, fs = importWave(argv[1])
    cpsData = cepstrum(wavdata,fs,60)
    if argNum == 3:
        outputCSV(cpsData,argv[2])

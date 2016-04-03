#coding:utf-8
#-------------------------------------------------------------------------------
#   Name:		MFCC.py
#	Author:		R.Imai
#	Created:	2016 / 03 / 20
#	Last Date:	2016 / 04 / 03
#	Note:       <wavfile name><save name>
#-------------------------------------------------------------------------------
import numpy as np
from pylab import *
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

def pickupCenter(data,cuttime = 0.04):
    center = len(data) / 2  # 中心のサンプル番号
    wavdata = data[center - cuttime/2*fs : center + cuttime/2*fs]
    return wavdata

def preEmphasis(data, p):
    # プレエンファシスフィルタ
    # y(t) = x(t) - p*x(t-1)
    #   pはプリエンファシス係数(一般的に0.97らしい)
    return sg.lfilter([1.0, -p], 1, data)

def hz_Mel(f):
    #Hzをmelに変換
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel_Hz(m):
    #melをhzに変換
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数（Hz） ←サンプリング周波数の1/2のこと
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz_Mel(fmax)
    # 周波数インデックスの最大数
    nmax = nfft / 2
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / nfft
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel_Hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters



def fft(data,n):
    winFunc = sg.hamming(len(data))  #窓関数
    wavdata = data * winFunc
    sig = np.abs(fftpack.fft(wavdata, n))[:n/2]
    freq = fftpack.fftfreq(n, d = 1.0 / fs)[:n/2]
    return sig,freq

def mfcc(wavdata):
    preEmData = preEmphasis(wavdata, 0.97)

    spec,fscale = fft(preEmData,nfft)

    # メルフィルタバンクを作成
    numChannels = 20  # メルフィルタバンクのチャネル数
    df = fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
    filterbank, fcenters = melFilterBank(fs, nfft, numChannels)

    # 振幅スペクトルにメルフィルタバンクを適用
    mspec = np.log10(np.dot(spec, filterbank.T))

    # 離散コサイン変換
    ceps = fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)


    # 元の振幅スペクトルとフィルタバンクをかけて圧縮したスペクトルを表示
    subplot(311)
    plot(fscale, np.log10(spec))
    for c in np.arange(0, numChannels):
        plot(np.arange(0, nfft / 2) * df, filterbank[c])
    xlabel("frequency")
    xlim(0, 25000)

    subplot(312)
    plot(fcenters, mspec, "o-")
    xlabel("frequency")
    xlim(0, 25000)

    subplot(313)
    plot(ceps)
    xlabel("frequency")
    show()

    return ceps


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
    nfft = 2048
    wavdata, fs = importWave(argv[1])
    wavdata = pickupCenter(wavdata)

    resMfcc = mfcc(wavdata)
    outputCSV(resMfcc,"testMFCC")

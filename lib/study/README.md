# MusiQ
###cepstrum.py
waveファイルの中心に対してケプストラム分析を行う<br>
>ケプストラム分析の要約:<br>
>データ(実時間領域)→FFT(周波数領域)→log→IFFT(ケプストラム領域)→高周波と低周波に分ける→FFT(周波数領域)<br>
>参考URL:  http://aidiary.hatenablog.com/entry/20120211/1328964624

###MFCC.py
waveファイルの中心に対してMFCC(メル周波数ケプストラム係数)を行う。
>MFCCの要約:<br>
>データ→プリエンファシスフィルタ(差分みたいなやつ)→FFT→メルフィルタバンクをかける→log→離散コサイン変換<br>

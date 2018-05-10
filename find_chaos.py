import numpy as np
import json as js
import scipy as sp
import scipy.signal
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':

    i = 0
    plist = np.linspace(0.6203, 0.64659, 100)
    while i < 99:
        #i= 20
        A = plist[i] - (1-plist[i])
        print(plist[i])
        print()
        with open('data/rt2list_' + str(i)+'.json') as json_data:
            d = np.array(js.load(json_data))
            #timelist = np.linspace(0.0,50.0,25000)
            #plt.plot(np.linspace(0.0,50.0,25000), d, 'k')
            #plt.show()


        #fft = np.fft.fft(d[-10000:])
        #fftfreq = np.fft.fftfreq(timelist[-10000:].shape[-1])
        #plt.plot(fftfreq, fft)
        #plt.show()

            maximum = sp.signal.argrelmax(np.array(d))[0]
            minimum = sp.signal.argrelmin(np.array(d))[0]
        #print(type(maximum))
        #print(d[maximum])
        #print(timelist[maximum])
        #plt.scatter(timelist[maximum], d[maximum], s=8)
        #plt.show()
            plt.scatter(np.repeat(A, 5), d[minimum[-5:]], s=2, c='r')
            #plt.scatter(np.repeat(A, 200), d[maximum[-200:]], marker='+', s=50, c='k')
            plt.scatter(np.repeat(A, 5), d[maximum[-5:]], s=2, c='k')
        #plt.scatter(np.repeat(1, 500), d[maximum[-500:]], marker='x', s=50, c='k')
        #plt.scatter(np.repeat(1, 1000), d[maximum[-1000:]], marker='+', s=50, c='g')
            #print(len(maximum))
            #print(maximum)
        i += 1
    plt.show()
    #t = np.arange(256)
    #s = np.fft.fft(np.sin(t))
    #freq = np.fft.fftfreq(t.shape[-1])
    #print(t.shape[-1])
    #plt.plot(freq, s.real)
    #plt.show()
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Image_Processing():
    def __init__(self):
        pass

    def load_image(self, fname, color='BGR'):
        img = cv2.imread(fname)
        if img.all() == None:
            raise Exception('%s is not found.' %(fname))
        if color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def save_image(self, fname, img, format='JPEG', q=100):
        if format == 'JPEG':
            cv2.imwrite(fname, img, [cv2.IMWRITE_JPEG_QUALITY, q])
        else:
            cv2.imwrite(fname, img)
        return None

    def show_image(self, img, title='sample'):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def smoothing(self, img, level, method='gaussian'):
        if method == 'gaussian':
            img = cv2.GaussianBlur(img, (level,level), 0)
        elif method == 'averaging':
            img = cv2.blur(img, (level,level))
        elif method == 'median':
            img = cv2.medianBlur(img, level)
        else:
            raise Exception('%s method is not defined.' %(method))
        return img
    
    def binarization(self, img, threshold, bw=True):
        if bw:
            ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        else:
            ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
        return img

    def grayscale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def trimming(self, img, l, r, t, b):
        shape = img.shape
        left = int(0 + shape[1]*l)
        right = int(shape[1] - shape[1]*r)
        top = int(shape[0] - shape[0]*b)
        bottom = int(0 + shape[0]*t)
        img = img[bottom:top, left:right]
        return img

    def add_white_noise(self, img, level=1):
        wn = np.random.rand(img.shape[0], img.shape[1])
        img = img + wn * level
        return img

class Signal_Processing():
    def __init__(self):
        pass

    def image2signal(self, img, ch=0, dir=0, area=[0, 1], method='sum'):
        if img.ndim == 3:
            img = img[:, :, ch]
        s = img.shape
        start = int(0 + s[dir] * area[0])
        end = int(s[dir] * area[1])
        if dir == 0:
            img = img[start:end, :]
        else:
            img = img[:, start:end]
        if method == 'sum':
            signal = np.sum(img, axis=dir)
        elif method == 'average':
            signal = np.mean(img, axis=dir)
        elif method == 'min':
            signal = np.min(img, axis=dir)
        elif method == 'max':
            signal = np.max(img, axis=dir)
        elif method == 'std':
            signal = np.std(img, axis=dir)
        elif method == 'var':
            signal = np.var(img, axis=dir)
        return signal

    def show_signal(self, signal, c='black', xlim=None, ylim=None, title='Sample'):
        if type(c) is str:
            color = c
        if type(xlim) is list:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = 0
            xmax = len(signal)
        if type(ylim) is list:
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin = np.min(signal)
            ymax = np.max(signal)
        plt.plot(signal, color=color, linestyle='solid')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(title)
        plt.show()
        return None

    def show_2signal(self, signal1, signal2, c='black', xlim=None, ylim=None, title='Sample'):
        if type(c) is str:
            color = c
        if type(xlim) is list:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = 0
            xmax1 = len(signal1)
            xmax2 = len(signal2)
            xmax = max(xmax1, xmax2)
        if type(ylim) is list:
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin1 = np.min(signal1)
            ymax1 = np.max(signal1)
            ymin2 = np.min(signal2)
            ymax2 = np.max(signal2)
            ymin = min(ymin1, ymin2)
            ymax = max(ymax1, ymax2)
        plt.plot(signal1, color=color, linestyle='solid')
        plt.plot(signal2, color=color, linestyle='dashed')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(title)
        plt.show()
        return None

    def diff_signal(self, signal, n=1):
        signal = np.diff(signal, n=n)
        return signal

    def abs_signal(self, signal):
        signal = np.absolute(signal)
        return signal

    def search_peak(self, signal, area, threshold, dir=True, No=0):
        signal_ind = np.where(signal>threshold)[0]
        signal_ind = signal_ind[signal_ind >= area[0]]
        signal_ind = signal_ind[signal_ind <= area[1]]
        peak_ind = []
        for i in range(len(signal_ind)-2):
            first = signal[signal_ind[i]]
            second = signal[signal_ind[i+1]]
            thired = signal[signal_ind[i+2]]
            if first <= second and thired <= second:
                peak_ind.append(signal_ind[i+1])
        if dir:
            peak_ind = peak_ind[No]
        else:
            peak_ind = peak_ind[-1*(No+1)]
        return peak_ind

    def search_peak_max(self, signal, area, threshold):
        signal_ind = np.where(signal>threshold)[0]
        signal_ind = signal_ind[signal_ind >= area[0]]
        signal_ind = signal_ind[signal_ind <= area[1]]
        peak = np.argmax(signal[signal_ind])
        peak_ind = signal_ind[peak]
        return peak_ind

    def distance(self, point1, point2, coef):
        dist = abs(point2 - point1) * coef
        return dist

    def show_image_signal(self, img, signal, point1, point2, coef):
        dist = self.distance(point1, point2, coef)
        plt.imshow(img, origin='lower', cmap='gray_r')
        plt.xlim(0, img.shape[0])
        plt.ylim(0, img.shape[1])
        plt.plot(signal, color='blue', linestyle='solid')
        plt.vlines(point1, 0, img.shape[1], colors='red', linestyles='dashed')
        plt.vlines(point2, 0, img.shape[1], colors='red', linestyles='dashed')
        plt.text(img.shape[0]*0.7, img.shape[1]*0.9, '%f' %(dist), color='white')
        plt.show()

if __name__ == "__main__":
    fname = './lw.png'
    ctrl = Image_Processing()
    img = ctrl.load_image(fname, 'GRAY')
    img = ctrl.smoothing(img, 5)
    img = ctrl.add_white_noise(img, 100)
    signal = Signal_Processing()
    s = signal.image2signal(img, dir=0, area=[0.45, 0.55], method='average')
    d = signal.diff_signal(s, 1)
    dd = signal.diff_signal(d, 1)
    d = signal.abs_signal(d)
    peak1 = signal.search_peak(d, [100,200], 10, dir=False, No=0)
    peak2 = signal.search_peak(d, [300,400], 10, dir=True, No=0)
    dist = signal.distance(peak1, peak2, 1)
    signal.show_image_signal(img, s, peak1, peak2, 1)

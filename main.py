from helperFunctions import *
import numpy as np
from settings import *
from scipy.io import savemat, loadmat
from scipy.io.wavfile import write
import random
import matplotlib.pyplot as plt
random.seed(42069)

if __name__ == "__main__":
    trainBlock = qammod([random.randint(0, 1) for i in range(int(Nd*b))], M)  # To estimate channel

    bitStream = imageCapture.getImageStream()  # Capture image using webcam
    data_bits = bitStream.size
    print("data bits", data_bits)

    pad_amount = int(Nd*b*Ld*(math.ceil(bitStream.size/(Nd*b*Ld)))-bitStream.size)
    padding = [random.randint(0, 1) for i in range(pad_amount)]
    bitStream = np.append(bitStream, padding)  # pad bitStream to fit in data packages

    # generate QAM stream
    qamStream = qammod(bitStream, M)

    # generate OFDM stream
    ofdmStream = ofdm_mod(qamStream, trainBlock, N, L, Lt, Ld)
    ofdmStreamsize = ofdmStream.size
    print("len of transmitted OFDMstream", ofdmStream.size)

    #  add sync pulses
    simin = initparams(ofdmStream, fs, Nr)
    write("siminStream.wav", fs, simin)  # save the .wav file
    #### Now you can play the file simStream.wav and record it

    ## bitStream = np.load("bitStream.npy")
    ## simin = np.load("simin.npy")
    ## data_bits = bitStream.size
    ## ofdmStream = simulateAudioTransmission(simin, CHUNK, sync)

    rxOfdmStream = getDataStream()
    print("recorded ofdmStream of size", rxOfdmStream.size)

    rxQamStream = ofdm_demod(rxOfdmStream, trainBlock, N, L, Lt, Ld)
    rxBitStream = qamdemod(rxQamStream, M)
    rxBitStream = rxBitStream[:data_bits]

    # regenerate image from received bitStream
    imageCapture.getImage(rxBitStream, (48, 64, 3))

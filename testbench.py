from main import *
import numpy as np
from settings import *
from scipy.io import savemat, loadmat
from scipy.io.wavfile import write

# M = 16
# bitStream = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
#              0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
#              1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1 ,0 ,1 ,1 ,1 ,0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
#              0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
#              0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
# trainBlock = np.array([0.707106781186548 - 0.707106781186548j,
#                        -0.707106781186548 - 0.707106781186548j,
#                        0.707106781186548 + 0.707106781186548j,
#                        -0.707106781186548 - 0.707106781186548j,
#                        0.707106781186548 - 0.707106781186548j])
# bitStream = np.random.randint(0, 2, int(240*4*16))
# assert((qamdemod(qammod(bitStream, M), M) == bitStream).all())

if __name__ =="__main__":
    trainBlock = np.ravel(loadmat("imageTrainBlock.mat")['trainBlock'])
    # bitStream = np.ravel(loadmat("imageBitStream.mat")['bitStream'])
    # bitStream = imageCapture.getImageStream()
    # data_bits = bitStream.size
    data_bits = 73728
    print("data bits", data_bits)

    # pad_zeros = np.zeros(Nd*b*Ld*(math.ceil(bitStream.size/(Nd*b*Ld)))-bitStream.size)
    # bitStream = np.append(bitStream, pad_zeros)

    # qamStream = qammod(bitStream, M)
    # ofdmStream = ofdm_mod(qamStream, trainBlock, N, L, Lt, Ld, M)

    # simin = initparams(ofdmStream, fs, Nr)
    # # ofdmStream = alignIO(ofdmStream, sync=sync, Nr=200, length=data_bits)
    #
    # write("ofdmStreamTB.wav", fs, ofdmStream.astype(np.float32))
    # write("siminStreamTB.wav", fs, simin)
    # write("ofdmStreamTB16.wav", fs, ofdmStream.astype(np.int16))
    ofdmStream = getDataStream() #136400

    ofdmStream = ofdmStream[:68200] #68200 for picture 136400

    rxQamStream = ofdm_demod(ofdmStream, trainBlock, N, L, Lt, Ld, M)
    rxBitStream = qamdemod(rxQamStream, M)
    rxBitStream = rxBitStream[:data_bits]

    # BER = sum(np.bitwise_xor(bitStream, rxBitStream))/136400
    # print("BER", BER)
    imageCapture.getImage(rxBitStream, (48, 64, 3))
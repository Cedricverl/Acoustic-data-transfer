import numpy as np
import imageCapture, math, pyaudio
from settings import *

from qampoints import mapping_table, demapping_table
fs=20000

 #amount of zero padding samples between pulse and data stream
def getDataStream():

    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 20000
    CHUNK = 2048*2
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,rate=RATE, input=True,frames_per_buffer=CHUNK)
    print("recording...")
    xcorr = 0
    xcorr_max = 0
    dataFrames = np.zeros(0, np.float32)
    while xcorr_max < 30:
        raw_data = stream.read(CHUNK)
        decoded_data = np.frombuffer(raw_data, np.float32)
        xcorr = np.correlate(decoded_data, sync)
        xcorr_max = max(xcorr)
        xcorr_maxIndex = np.argmax(xcorr)
        if(xcorr_max) > 30:
            dataFrames = np.append(dataFrames, decoded_data[xcorr_maxIndex + sync.size + Nr - 20:])
    xcorr_max = 0
    print("recording actual data...")
    while xcorr_max < 30:
        raw_data = stream.read(CHUNK)
        decoded_data = np.frombuffer(raw_data, np.float32)
        xcorr = np.correlate(decoded_data, sync)
        xcorr_max = max(xcorr)
        xcorr_maxIndex = np.argmax(xcorr)
        if xcorr_max > 30:
            dataFrames = np.append(dataFrames, decoded_data[:xcorr_maxIndex+sync.size])
        else:
            dataFrames = np.append(dataFrames, decoded_data)
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    return dataFrames


def alignIO(out, sync, Nr, length):
    # Out: dataStream
    # pulse: sync pulse
    # Nr: amount of zero padding samples between pulse and data stream
    return out[:length] #known length of data stream


def ofdm_mod(qamStream, qamTrainblock, N, L, Lt, Ld, M):
    Nd = N//2-1
    Pdata = int(qamStream.size/Nd)
    P = Pdata + int(Pdata*Lt/Ld)
    packet = np.zeros((N, P), np.complex128)
    trainingIndex = [i for i in range(P) if i%(Lt+Ld) in range(Lt)]
    dataIndex = [i for i in range(P) if i not in trainingIndex]
    packet[1:Nd+1, trainingIndex] = np.transpose(np.tile(qamTrainblock, (P-Pdata, 1)))
    packet[1:Nd+1, dataIndex] = np.reshape(qamStream, (Nd, Pdata), order='F')
    packet[Nd+2:, :] = np.conj(np.flip(packet[1:Nd+1, :], axis=0))
    packetfft = np.fft.ifft(packet, axis=0)
    packet_wprefix = np.zeros((N+L, P), np.complex128)
    packet_wprefix[L:, :] = packetfft
    # packet_wprefix[0:L, :] = packet_wprefix[packet_wprefix.shape[0]-L-1::-1, :]
    # packet_wprefix[0:L, :] = packet_wprefix[packet_wprefix.shape[0]-1: packet_wprefix.shape[0]-L-1:-1, :]
    packet_wprefix[:L, :] = packet_wprefix[-L:, :]
    return np.ravel(packet_wprefix, order='F')


def ofdm_demod(ofdm_stream, qamTrainblock, N, L, Lt, Ld, M):
    P = ofdm_stream.size//(N+L)
    Nd = N//2-1
    packet_wprefix = np.reshape(ofdm_stream, (N+L, P), order='F')
    packet = packet_wprefix[L:, :]

    almostdemodulated = np.fft.fft(packet, axis=0)
    demodulated_cleaned = almostdemodulated[1:Nd+1, :]
    packets = np.array(np.split(demodulated_cleaned, P//(Lt+Ld), axis=1))
    trainingpackets = np.mean(packets[:, :, :Lt], axis=2)

    H_ests = trainingpackets/qamTrainblock
    H_ests = np.transpose(np.tile(np.transpose(H_ests), (8, 1, 1)))

    datapackets = packets[:, :, Lt:]
    datapackets = datapackets/H_ests

    # data_serial = np.ravel(datapackets, order='F')
    data_serial = np.ravel([np.ravel(datapacket, order='F') for datapacket in datapackets])
    return data_serial


def qammod(stream, M):
    b = int(np.log2(M))
    return np.array([mapping_table[M][tuple(x)] for x in np.reshape(stream, (-1, b))])


# assert((qammod(np.array([1,1,0,1,1,1,1,0,0,0,1,0]), 16) == [0.316227766016838 + 0.316227766016838j,0.316227766016838 - 0.948683298050514j,-0.948683298050514 - 0.948683298050514j]).all())

def qamdemod(QAMstream, M):
    constellation = np.array(list(demapping_table[M].keys()))
    dists = abs(np.reshape(QAMstream, (-1, 1)) - np.reshape(constellation, (1, -1)))
    hardDecision = constellation[dists.argmin(axis=1)]
    return np.concatenate([list(demapping_table[M][C]) for C in hardDecision])

def initparams(toplay, fs, Nr):
    toplay_scaled = toplay/max(abs(toplay))

    toplay_syncpulsed = np.concatenate((sync, np.zeros(Nr), toplay_scaled, sync))
    return toplay_syncpulsed.astype(np.float32)

# assert((qamdemod([0.316227766016838 + 0.316227766016838j,0.316227766016838 - 0.948683298050514j,-0.948683298050514 - 0.948683298050514j], 16) == np.array([1,1,0,1,1,1,1,0,0,0,1,0])).all())

if __name__ == "__main__":
    trainBlock = qammod(np.random.randint(0, 2, int(Nd*b)), M)  # To estimate channel
    # trainBlock = np.ravel(loadmat("imageTrainBlock.mat")['trainBlock'])

    # trainBlock = qammod(np.ones(int(Nd*b)), M)
    # bitStream = np.randdom.randint(0, 2, int(Nd*b*(Ld*100))) #Ld*2 so I have 2 full data packets

    bitStream = imageCapture.getImageStream()
    data_bitwidth = bitStream.size
    # data_bitwidth = 73782
    pad_zeros = np.zeros(Nd*b*Ld*(math.ceil(bitStream.size/(Nd*b*Ld)))-bitStream.size)
    bitStream = np.append(bitStream, pad_zeros)


    qamStream = qammod(bitStream, M)
    ofdmStream = ofdm_mod(qamStream, trainBlock, N, L, Lt, Ld, M)

    # simin = initparams(ofdmStream, fs, Nr)
    # ofdmStream = getDataStream() #136400
    # ofdmStream = alignIO(ofdmStream, sync=sync, Nr=200, length=data_bitwidth)

    # No channel modulation for now
    rxQamStream = ofdm_demod(ofdmStream, trainBlock, N, L, Lt, Ld, M)
    rxBitStream = qamdemod(rxQamStream, M)
    rxBitStream = rxBitStream[:data_bitwidth]

    # bitStream = np.ravel(loadmat("imageBitStream.mat")['bitStream'])
    # BER = sum(np.bitwise_xor(bitStream, rxBitStream))/136400
    # print("BER", BER)
    imageCapture.getImage(rxBitStream, (48, 64, 3))




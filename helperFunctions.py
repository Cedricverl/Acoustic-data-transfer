import numpy as np
import imageCapture, math, pyaudio
from settings import *
from qampoints import mapping_table, demapping_table


def getDataStream():
    FORMAT = pyaudio.paFloat32
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=fs, input=True, frames_per_buffer=CHUNK)
    print("recording...")
    xcorr_max = 0
    decodedData = np.zeros(CHUNK*3, np.float32)
    dataFrames = np.zeros(0, np.float32)

    while xcorr_max < threshold:
        raw_data = stream.read(CHUNK)
        decodedData = np.append(decodedData[CHUNK:], np.frombuffer(raw_data, np.float32))

        xcorr = np.correlate(decodedData[:CHUNK*2], sync)
        xcorr_max = max(xcorr)
        xcorr_maxIndex = np.argmax(xcorr)

        xcorr2 = abs(np.correlate(decodedData[CHUNK:], sync))
        xcorr_max2 = max(xcorr2)
        xcorr_maxIndex2 = np.argmax(xcorr2)
        if xcorr_max > threshold:
            if xcorr_max >= xcorr_max2:
                dataFrames = decodedData[xcorr_maxIndex + sync.size + Nr:]
            else:
                dataFrames = decodedData[CHUNK+xcorr_maxIndex2 + sync.size + Nr:]
    raw_data = stream.read(CHUNK*2)
    decodedData = np.append(np.zeros(CHUNK, np.float32), np.frombuffer(raw_data, np.float32))
    xcorr_max = 0
    print("recording actual data...")
    while xcorr_max < 30:
        raw_data = stream.read(CHUNK)
        decodedData = np.append(decodedData[CHUNK:], np.frombuffer(raw_data, np.float32))

        xcorr = np.correlate(decodedData[:CHUNK*2], sync)
        xcorr_max = max(xcorr)
        xcorr_maxIndex = np.argmax(xcorr)

        xcorr2 = abs(np.correlate(decodedData[CHUNK:], sync))
        xcorr_max2 = max(xcorr2)
        xcorr_maxIndex2 = np.argmax(xcorr2)
        if xcorr_max > threshold:
            if xcorr_max >= xcorr_max2:
                dataFrames = np.append(dataFrames, decodedData[:xcorr_maxIndex])
            else:
                dataFrames = np.append(dataFrames, decodedData[:CHUNK+xcorr_maxIndex2])
        else:
            dataFrames = np.append(dataFrames, decodedData[:CHUNK])
    stream.stop_stream()    # Stop recording
    stream.close()
    audio.terminate()
    return dataFrames


def alignIO(out, sync, Nr, length):
    # Out: dataStream
    # pulse: sync pulse
    # Nr: amount of zero padding samples between pulse and data stream
    return out[:length] #known length of data stream


def ofdm_mod(qamStream, qamTrainblock, N, L, Lt, Ld):
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

    packet_wprefix[:L, :] = packet_wprefix[-L:, :]
    return np.ravel(packet_wprefix, order='F')


def ofdm_demod(ofdm_stream, qamTrainblock, N, L, Lt, Ld):
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
    toplay_scaled = toplay/max(abs(toplay))  # Normalize OFDM stream
    toplay_syncpulsed = np.concatenate((sync, np.zeros(Nr), toplay_scaled, sync))  # add sync and padding to prevent bleeding
    return toplay_syncpulsed.astype(np.float32)  # convert complex array to real (imag part is 0 anyways)


def simulateAudioTransmission(ofdmStream, CHUNK, sync):
    ofdmStream = np.append(ofdmStream, np.zeros(CHUNK, np.float32))
    dataFrames = np.zeros(0, np.float32)
    xcorr_max = 0
    decodedData = np.zeros(CHUNK*3, np.float32)

    while xcorr_max < threshold:
        decodedData = np.append(decodedData[CHUNK:], ofdmStream[:CHUNK])
        ofdmStream = ofdmStream[CHUNK:]

        xcorr = abs(np.correlate(decodedData[:CHUNK*2], sync))
        xcorr_max = max(xcorr)
        xcorr_maxIndex = np.argmax(xcorr)

        xcorr2 = abs(np.correlate(decodedData[CHUNK:], sync))
        xcorr_max2 = max(xcorr2)
        xcorr_maxIndex2 = np.argmax(xcorr2)
        # print("xcorr_max", xcorr_max, "xcorr_maxI", xcorr_maxIndex, "xcorr_max2", xcorr_max2, "xcorr_maxI2", xcorr_maxIndex2)
        if(xcorr_max > threshold):
            if xcorr_max >= xcorr_max2:
                dataFrames = decodedData[xcorr_maxIndex + sync.size + Nr:]
            else:
                dataFrames = decodedData[CHUNK+xcorr_maxIndex2 + sync.size + Nr:]
    decodedData = np.append(np.zeros(CHUNK, np.float32), ofdmStream[:CHUNK*2])
    ofdmStream = ofdmStream[CHUNK*2:]
    xcorr_max = 0

    while xcorr_max < threshold:
        decodedData = np.append(decodedData[CHUNK:], ofdmStream[:CHUNK])
        ofdmStream = ofdmStream[CHUNK:]

        xcorr = abs(np.correlate(decodedData[:CHUNK*2], sync))
        xcorr_max = max(xcorr)
        xcorr_maxIndex = np.argmax(xcorr)

        xcorr2 = abs(np.correlate(decodedData[CHUNK:], sync))
        xcorr_max2 = max(xcorr2)
        xcorr_maxIndex2 = np.argmax(xcorr2)
        # print("xcorr_max", xcorr_max, "xcorr_maxI", xcorr_maxIndex, "xcorr_max2", xcorr_max2, "xcorr_maxI2", xcorr_maxIndex2)
        if xcorr_max > threshold:
            if xcorr_max >= xcorr_max2:
                dataFrames = np.append(dataFrames, decodedData[:xcorr_maxIndex])
            else:
                dataFrames = np.append(dataFrames, decodedData[:CHUNK+xcorr_maxIndex2])
        else:
            dataFrames = np.append(dataFrames, decodedData[:CHUNK])

    return dataFrames


if __name__ == "__main__":
    pass

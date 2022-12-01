import wave
import pyaudio

filename = "MyPictureSound.wav"
rate = 20000
dataFrames = simin

waveFile = wave.open(filename, 'wb')
waveFile.setnchannels(1)
waveFile.setsampwidth(pyaudio.paFloat32)
waveFile.setframerate(rate)
waveFile.writeframes(dataFrames)
waveFile.close()
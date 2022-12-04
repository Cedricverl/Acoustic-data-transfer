# Acoustic-data-transfer

Use your laptop speakers and microphone as transmitter and receiver for OFDM-modulated packages. 
Data will be modulated and saved as a .wav file. This can be played by your laptop or any other device with speakers in the proximity. 
The transmitted signal will be detected using a sync pulse and be demodulated to reconstruct te original data. 

This data can be any (binary) array. With the use of OpenCV, an image is taken using your webcam and this is converted to a binary stream. 

The data is transmitted using Lt training frames and Ld data frames at a time. 
A BER of ~12% is achieved using 4-QAM depending on the surrounding noise and audio equipment. 

TODO: 
- [x] Training Frames
- [ ] ON-OFF Bitloading
- [ ] Adaptive Bitloading
- [ ] Pilot Tones

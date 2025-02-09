import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
from scipy.io import loadmat
from sklearn.svm import LinearSVR
from scipy.io.wavfile import read, write
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.optimizers import SGD
import librosa
import os

## MACHINE LEARNING FOR AUDIO RECONSTRUCTION

def networks(md, X_train, y_train):
    if md != 'random_forest':
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    if md == 'lstm':
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=128)

    elif md == 'simple_rnn':
        model = Sequential()
        model.add(SimpleRNN(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(Dense(units=1))

        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=200, verbose=0)

    elif md == 'gru':
        model = Sequential()
        model.add(GRU(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(GRU(units=32, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum=0.6, nesterov=False), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)

    elif md == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

    elif md == 'svr':
        nsamples, nx, ny = X_train.shape
        X_train_2d = X_train.reshape((nsamples, nx*ny))
        y_train_2d = np.array(y_train)
        y_train_1d = y_train_2d.flatten()

        model = LinearSVR(verbose=0, C=1, epsilon=0, fit_intercept=True, intercept_scaling=1.0, max_iter=2000, random_state=None, tol=0.0001)
        model.fit(X_train_2d, y_train_1d)

    else:
        print('No Such Model')
        return None

    return model

def calculations(file_path, model_name):
    original_mat = pd.DataFrame(loadmat(file_path + 'original.mat')['y'])
    hidden_mat = pd.DataFrame(loadmat(file_path + 'hiddenData.mat')['KK'])
    stego_mat = pd.DataFrame(loadmat(file_path + 'stegoAudio.mat')['W'])
    stego_recon = pd.DataFrame(loadmat(file_path + 'reconstructedAudio.mat')['recon'])

    nan_index = stego_mat[0][stego_mat[0].apply(np.isnan)].index
    stego_mat.iloc[nan_index] = 0

    original_drop_mat = original_mat.copy()
    original_drop_mat.iloc[nan_index] = 0

    hidden_drop_mat = hidden_mat.copy()
    hidden_drop_mat.iloc[nan_index] = 0

    nan_index = nan_index[nan_index < len(stego_mat)]  # Limit the size of nan_index

    test_data = hidden_mat.iloc[nan_index]

    X_train = hidden_drop_mat
    y_train = stego_mat

    model = networks(model_name, X_train, y_train)
    if model is None:
        return

    X_test = np.array(test_data)

    if model_name in ['lstm', 'simple_rnn', 'gru']:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_recon_audio = model.predict(X_test)

    final_mat = stego_mat.copy()
    min_length = min(len(nan_index), len(predicted_recon_audio))

    for i in range(min_length):
        final_mat.iloc[nan_index[i]] = predicted_recon_audio[i]

    signalOut = final_mat.values
    reconstructed_audio = {"reconL": final_mat.values, "Fs": 44100}
    sp.savemat(file_path + model_name + "_reconstructed_audio.mat", reconstructed_audio)
    write(file_path + 'OutAudio/reconstructedgru.wav', 44100, signalOut.astype(np.float64))

    recon_mat = pd.DataFrame(final_mat.values)
    frames = pd.concat([original_mat, recon_mat, stego_mat, stego_recon], axis=1)
    frames.columns = ['original', model_name, 'stego_audio', 'stego_recon']
    print(frames.corr())

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    axes[0].plot(original_mat, color='blue', label='Original Audio')
    axes[0].set_title('Original Audio')
    axes[1].plot(recon_mat, color='red', label='Reconstructed Audio')
    axes[1].set_title('Reconstructed Audio')
    axes[2].plot(recon_mat, color='red', label='Reconstructed Audio (SVR)')
    axes[2].set_title('SVR Reconstructed Audio')
    axes[3].plot(stego_mat, color='green', label='Dropped Audio')
    axes[3].set_title('Dropped Audio')

    for ax in axes:
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.savefig(file_path + 'simple_RNN_plots/outputgru.jpg')
    plt.show()

# Run the model
model_name = 'gru'
file_path = 'C:/Users/HP/Desktop/ADSPProject/audio_reconstruction_project/'
#calculations(file_path, model_name)
print("Done")

folder_path = 'C:/Users/HP/Desktop/ADSPProject/audio_reconstruction_project/OutAudio'

# Get all .wav files in the folder
audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Create a figure for the plot
plt.figure(figsize=(10, 6))

# Iterate through each audio file and calculate FFT
for file in audio_files:
    file_path = os.path.join(folder_path, file)
    
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Compute the Short-Time Fourier Transform (STFT) for FFT
    D = librosa.stft(y)
    magnitude, phase = librosa.magphase(D)  # Get magnitude and phase
    fft = np.abs(magnitude)  # Magnitude spectrum (FFT)
    
    # Compute the frequency axis for plotting
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Plot the FFT for the current file on a logarithmic scale
    plt.plot(freqs, np.mean(fft, axis=1), label=file)

# Set logarithmic scale for the y-axis
plt.yscale('log')

# Add labels and title
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Log scale)')
plt.title('FFT of Multiple Audio Files (Logarithmic Scale)')
plt.legend(loc='upper right')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
#Calculating spectral centroid
'''
folder_path = 'C:/Users/HP/Desktop/ADSPProject/audio_reconstruction_project/OutAudio'
audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
plt.figure(figsize=(14, 10))

# Iterate through each audio file
for idx, file in enumerate(audio_files):
    file_path = os.path.join(folder_path, file)
    
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Compute the Short-Time Fourier Transform (STFT) for FFT
    D = librosa.stft(y)
    magnitude, phase = librosa.magphase(D)  # Get magnitude and phase
    fft = np.abs(magnitude)  # Magnitude spectrum (FFT)
    
    # Compute the spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Time axis for plotting
    time = librosa.times_like(fft)
    time_centroid = librosa.frames_to_time(range(len(spectral_centroid)), sr=sr)

    # Plot the FFT (log scale)
    plt.subplot(len(audio_files), 2, 2 * idx + 1)  # Plot FFT on the left
    librosa.display.specshow(librosa.amplitude_to_db(fft, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.title(f'FFT of {file}')
    plt.colorbar(format='%+2.0f dB')

    # Plot the Spectral Centroid
    plt.subplot(len(audio_files), 2, 2 * idx + 2)  # Plot Spectral Centroid on the right
    plt.plot(time_centroid, spectral_centroid, label=f'Spectral Centroid ({file})', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Centroid (Hz)')
    plt.title(f'Spectral Centroid of {file}')
    plt.legend()
    plt.grid(True)

# Adjust the layout
plt.tight_layout()
plt.savefig(r'C:\Users\HP\Desktop\ADSPProject\audio_reconstruction_project\FFT_plots\outputsignal2.jpg')
plt.show()
'''
from pydub import AudioSegment
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import librosa
matplotlib.use("Agg")
import os
import wave
# 读取音频文件 切割音频文件 10s/segment
def split_audio(input_file, output_folder, segment_length=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with wave.open(input_file, 'rb') as wf:
        # 获取音频文件的参数
        params = wf.getparams()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()

        segment_frames = segment_length * frame_rate

        segment_index = 0
        while True:
            frames = wf.readframes(segment_frames)
            if not frames:
                break
            segment_filename = os.path.join(output_folder, f'segment_{segment_index}.wav')
            with wave.open(segment_filename, 'wb') as segment_wf:
                segment_wf.setnchannels(channels)
                segment_wf.setsampwidth(sample_width)
                segment_wf.setframerate(frame_rate)
                segment_wf.writeframes(frames)
            segment_index += 1
    print(f'Audio split into {segment_index} segments.')



def wave2img(wave_path, save_path):
    '''convert the wave file to an image file'''
    audio = AudioSegment.from_file(wave_path, format="wav")
    array = audio.get_array_of_samples()
    plt.figure(figsize=(2.56, 2.56), dpi=100)
    plt.plot(array, color='black')
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    # Save the plot
    plt.savefig(save_path)

def draw_spectrogram(wav_name,save_name):
    '''draw the spectrogram of the audio file'''
    y, sr = librosa.load(wav_name)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(2.56, 2.56), dpi=100)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    # Save the plot
    # plt.savefig(save_name)
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)

    plt.close('all')
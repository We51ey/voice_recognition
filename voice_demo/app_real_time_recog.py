import pyaudio
import wave
import numpy as np
import logging
import cv2
import os
import sys
from utils.audio_process import draw_spectrogram
from utils.LBP import faster_calculate_LBP,calculate_distance
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout ,QLabel, QHBoxLayout, QLineEdit, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal,QTimer,Qt
from PyQt6.QtGui import QPixmap,QColor
class RealTimeRecognizer(QThread):
    update_sin = pyqtSignal()
    def __init__(self,parent=None):
        super().__init__(parent)
        self.FORMAT = pyaudio.paInt16  # 16-bit resolution
        self.CHANNELS = 1              # single channel
        self.RATE = 44100              # 44.1kHz sampling rate
        self.CHUNK = 1024              # 2^12 samples for buffer
        self.WINDOW_SIZE = 5        # Window size for evaluation
        self.OUTPUT_TXT_PATH = 'output/output.txt'
        self.OUTPUT_WAV_PATH = 'output/output.wav'
        self.MODEL_PATH = 'model/gray_all.txt'
        self.VOICE_SOURCE_FOLDER = 'voice_source/'
        self.VOICE_IMAGES_FOLDER = 'voice_images/'
        self.LABELS = os.listdir(self.VOICE_SOURCE_FOLDER)
        self.LABEL_VALUES = {label: i for i, label in enumerate(self.LABELS)}
        
        self.stopped = True
        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(
            format = self.FORMAT,
            channels = self.CHANNELS,
            rate = self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.callback
        )
        self.frames = []
        self.speaker = None
        self.valide_data = np.loadtxt(self.MODEL_PATH, delimiter=',')

    def process_image(self,text_name,img_path,label,img_type='gray'):
        '''process the image and save the LBP features to a file'''
        with open(text_name, 'a') as f:
            img = cv2.imread(img_path)
            # # vector= LBP(img, label,img_type)
            vector = faster_calculate_LBP(img, label,img_type)
            # np.savetxt(f, vector, fmt='%f', delimiter=',')
        return vector
    
    def find_key_by_value(self,dictionary, value) -> str:
        for key, val in dictionary.items():
            if val == value:
                return key
        return None  # not found

    def evaluate(self,train_data, test_data, distance_type):
        '''evaluate the model using test data'''
        accuracy = 0
        for data in test_data:
            distances = []
            for train_vector in train_data:
                distance = calculate_distance(data[1:], train_vector[1:], distance_type)
                distances.append([train_vector[0], distance])
            # sort the distances
            distances = sorted(distances, key=lambda x: x[1])
            np.argmin(distances)

            predicted_label = distances[0][0]
            min_distance = distances[0][1]
            # print(f"Predicted label: {predicted_label}, Actual label: {data[0]}, Distance: {min_distance}")
        return predicted_label


    def callback(self,in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        # 计算当前窗口中的数据总时长
        total_samples = sum(len(frame) for frame in self.frames) //2  # 每个样本占用2个字节
        total_time = total_samples / float(self.RATE)
        # total_time = len(self.frames) / float(self.RATE)
        
        # print(total_time)
        # 如果超过窗口大小，则移除最旧的数据
        while total_time > self.WINDOW_SIZE:
            oldest_frame = self.frames.pop(0)
            total_samples -= len(oldest_frame) // 2
            total_time = total_samples / float(self.RATE)
            
        return (in_data, pyaudio.paContinue)

    def run(self):
        '''reading audio from microphone and save the audio to a file'''
        # set the format of the audio
    
        valide_data = np.loadtxt(self.MODEL_PATH, delimiter=',')
        print("Recording started.")
        labels = []
        with open('output/output.txt', 'w') as file:
            pass
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
        if self.stream is None or self.stream.is_stopped():
            self.stream = self.audio.open(
                format = self.FORMAT,
                channels = self.CHANNELS,
                rate = self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.callback
            )
            self.stream.start_stream()
        try:
            self.stopped=False
            while not self.stopped:
                self.update_sin.emit()
                if len(self.frames) == 0:
                    continue
                with wave.open(self.OUTPUT_WAV_PATH, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(self.frames))
                img_name = self.OUTPUT_TXT_PATH.replace('.txt', '.png')
                draw_spectrogram(self.OUTPUT_WAV_PATH, img_name) # draw the spectrogram
                real_voice_data = self.process_image(self.OUTPUT_TXT_PATH,img_name, 0, 'gray')

                label = self.evaluate(valide_data, real_voice_data, distance_type='L2') # analyze the voice
                self.speaker = self.find_key_by_value(self.LABEL_VALUES, label)
                # print(f"Current Speaker: {self.find_key_by_value(self.LABEL_VALUES, label)}")
                labels.append(label)
        except KeyboardInterrupt:
            logging.info("Recording stopped.")
        finally:
            self.stop()
    def stop(self):
        self.stopped = True
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream=None
        self.frames = []


class VoiceRecognizerGUI(QWidget):
    stop_sin = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real Time Voice Recognizer")
        self.setGeometry(100, 100, 600, 300)
        
        # 左侧图片
        self.image_path = 'output/output.png'
        self.image_label = QLabel(self)
        pixmap = QPixmap(256, 256)
        pixmap.fill(QColor("white"))  # 填充背景颜色
        self.image_label.setPixmap(pixmap)

        # 右侧按钮
        self.speaker_label = QLabel("Speaker", self)
        # 居中
        self.speaker_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_button = QPushButton("start", self)
        self.stop_button = QPushButton("end", self)
        self.stop_button.setEnabled(False)  # 初始化时停止按钮不可用

        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        # 右侧布局
        vbox = QVBoxLayout()
        vbox.addWidget(self.speaker_label)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)

        # 主布局
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addLayout(vbox)

        self.recognizer_thread = RealTimeRecognizer()
        self.recognizer_thread.update_sin.connect(self.update_info)

        self.stop_sin.connect(self.recognizer_thread.stop)
        self.setLayout(hbox)


    def update_info(self):
        try:
            self.speaker_label.setText(f"Speaker: {self.recognizer_thread.speaker}" if self.recognizer_thread.speaker else "Speaker: None")
            new_pixmap = QPixmap(self.image_path)
            new_pixmap = new_pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(new_pixmap)
        except:
            pass
        
    def update_speaker_label(self):
        if not self.recognizer_thread.stopped:
            self.speaker_label.setText(f"Speaker: {self.recognizer_thread.speaker}")

    def start_recording(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.recognizer_thread.start()
        
    def stop_recording(self):
        self.stop_sin.emit()
        self.recognizer_thread.stopped = True
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceRecognizerGUI()
    window.show()
    sys.exit(app.exec())
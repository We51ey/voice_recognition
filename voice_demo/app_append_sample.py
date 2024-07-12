import sys
import os
import wave
import time
import pyaudio
import shutil
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLineEdit, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal
class RecorderThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path = None
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.stopped = False 
        
    def run(self):
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        self.frames = []
        print("Recording started")
        while not self.stopped:
            self.stopped = False
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            time.sleep(0.01)

        print("Recording stopped")
        stream.stop_stream()
        stream.close()

    def stop(self):
        # 保存录音文件
        segment_length=5
        # 计算每个片段的帧数
        frames_per_segment = int(self.RATE * segment_length/self.CHUNK)
        print(f"Frames per segment: {frames_per_segment}")
        # 分割音频并保存片段
        segment_count = 0
        
        for start in range(0, len(self.frames), frames_per_segment):
            end = start + frames_per_segment
            sample_width = self.audio.get_sample_size(self.FORMAT)
            segment_frames = self.frames[start * sample_width: end * sample_width]
            # random name
            random_file_name = os.path.join(self.file_path, str(time.time())+".wav")
            with wave.open(random_file_name, 'wb') as segment:
                segment.setnchannels(self.CHANNELS)
                segment.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                segment.setframerate(self.RATE)
                segment.writeframes(b''.join(segment_frames))
            segment_count += 1
        print(f"Saved {segment_count} segments")

class VoiceRecorderGUI(QWidget):
    sin = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Recorder")
        self.resize(600, 400)

        self.recording_name = None
        self.save_path = None
        self.voice_source = "voice_source"
        # 创建布局
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 创建左侧列表
        self.list_widget = QListWidget()
        left_layout.addWidget(self.list_widget)
        self.update_file_list()

        # 创建右侧输入框和按钮
        self.name_input = QLineEdit()
        self.start_button = QPushButton("start recording")
        self.stop_button = QPushButton("stop recording")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        right_layout.addWidget(self.name_input)
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)

        # 添加布局到主布局中
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 初始化 PyAudio
        self.audio = pyaudio.PyAudio()
        # 录音状态标志
        self.is_recording = False

        self.recorder_thread = RecorderThread()
        self.sin.connect(self.recorder_thread.stop)

    def update_file_list(self):
        self.list_widget.clear()
        """更新左侧列表中的文件夹名字"""
        folder_path = self.voice_source
        folders= os.listdir(folder_path)
        for folder in folders:
            self.list_widget.addItem(folder)
        # 你的更新文件夹列表的代码

    def start_recording(self):
        """开始录制声音"""
        if not self.is_recording:
            self.recording_name = self.name_input.text().strip()
            if not self.recording_name:
                return
            
            self.save_path = os.path.join(self.voice_source, self.recording_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)

            self.is_recording = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.recorder_thread.file_path = self.save_path
            self.recorder_thread.start()

    def stop_recording(self):
        """停止录制声音"""
        self.is_recording = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("Recording stopped")
        self.recorder_thread.stopped = True
        self.sin.emit()
        self.update_file_list()
    
    def closeEvent(self, event):
        """关闭窗口时停止录音"""
        if self.is_recording:
            self.stop_recording()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRecorderGUI()
    window.show()
    sys.exit(app.exec())

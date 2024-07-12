# Voice Recognition with LBP Algorithm
This project aims to recognize and detect the speaker in real-time using the Local Binary Pattern (LBP) algorithm applied to sound spectrograms. The implementation involves converting a speaker's voice into a spectrogram over a period, applying the LBP algorithm on the spectrogram, and matching the dynamic LBP-decomposed frequency data of the current speaker with the average LBP-decomposed frequency data of each label in the dataset to identify the speaker with the highest match.

## Introduction
This project is divided into two main components:

Voice Demo: Utilizes the PIL_PARK dataset to recognize parked cars using the LBP algorithm, achieving over 90% accuracy.(Tested)
Image Demo: Real-time detection and recognition of speakers using their voice spectrograms and the LBP algorithm.
The voice recognition functionality includes two main features:

Collecting samples from speakers.
Analyzing real-time spectrograms to recognize the speaker.

![无标题视频——使用Clipchamp制作](https://github.com/user-attachments/assets/cd9aa70b-6a82-46fa-9ccb-f074080a0519)


## Project Structure
The project is organized into two main directories:

voice_demo: Contains the implementation for recognizing parked cars using the PIL_PARK dataset.
image_demo: Contains the implementation for real-time voice recognition and sample collection.

## Installation
To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/voice_recognition.git
cd voice_recognition
pip install -r requirements.txt
```
## Usage
#### Collecting Speaker Samples
To collect samples from a speaker, run the following command:

```bash
python app_append_sample.py
```
This script will prompt the speaker to say certain phrases, record their voice, and save the spectrograms in the specified output directory.

#### Real-time Speaker Detection
To perform real-time speaker detection, run:
```bash
python app_real_time_recog.py
```
This script will analyze the real-time spectrogram of the speaker's voice and compare it with the pre-collected samples to identify the speaker.

# LBP Algorithm Overview
The Local Binary Pattern (LBP) algorithm is used to extract texture features from images. In this project, the LBP algorithm is applied to voice spectrograms to capture the unique features of a speaker's voice.

Steps:
1.  Convert the voice signal into a spectrogram.
2.  Apply the LBP algorithm on the spectrogram to extract features.
3.  Match the extracted features with the pre-collected samples to identify the speaker.

## Dataset
The voice_demo directory uses the ```PIL_PARK``` dataset to demonstrate the application of the LBP algorithm in recognizing parked cars.

## Results
The implementation of the LBP algorithm in the voice_demo directory achieved an accuracy of over ```90%``` in recognizing parked cars. The image_demo directory is designed for real-time speaker detection and recognition.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

For more information, please contact wl979159265@gmail.com.

# Real-time Doorbell Audio Classification With Raspberry Pi

This two-part project implements a real-time audio classification system running on a Raspberry Pi using a 1D Convolutional Neural Network (CNN). The system uses Python with Keras (TensorFlow backend) for the 1D CNN model, along with scikit-learn for label encoding and data splitting, plus librosa for audio feature extraction. The Firebase Cloud Messaging (FCM) system is used to send notifications to an Android app when a doorbell is detected and provide real-time alerts. The second part of the project is the Android app which also implements a logging system to track these notifications with timestamps.


## Features

- Real-time audio processing from Raspberry Pi microphone
- 1D Convolutional Neural Network for audio classification
- Firebase Cloud Messaging integration
- Android app for notification management
- Notification logging system with timestamps
- Continuous monitoring capability

## Requirements

### Hardware
- Raspberry Pi or alternatively a desktop environment is also supported (Windows, MacOS, Linux)
- USB microphone or compatible audio input device
- Android device for notification reception (Second part of the project)


### Python Dependencies
```bash
pip install numpy pandas librosa tensorflow sounddevice pygame firebase-admin
```

## Setup Guide

### Part 1: Raspberry Pi Setup

1. Install required Python packages:
```bash
sudo apt-get update
sudo apt-get install python3-pip portaudio19-dev
pip3 install numpy sounddevice librosa tensorflow firebase-admin scikit-learn
```

2. Set up the service:
```bash
# Copy the service file
sudo cp doorbell.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable doorbell
sudo systemctl start doorbell

# Check the service status
sudo systemctl status doorbell

```

### Part 2: Project Setup

3. Train the model:
```bash
python train_model.py
```

4. Configure Firebase credentials:
- Download your Firebase service account key
- Place it on the config folder
- Change the paths on python files

5. Start the detection system:
```bash
python raspberry_pi.py
```

Alternative for desktops using keras:
```bash
python keras_doorbell_detector.py
```


### Audio Processing Parameters
- Sample Rate: 22050 Hz
- Chunk Duration: 0.5 seconds
- Buffer Duration: 2.0 seconds
- MFCC Features: 40
- FFT Window Size: 1024 samples
- Hop Length: 512 samples


## License

This project is licensed under the MIT License

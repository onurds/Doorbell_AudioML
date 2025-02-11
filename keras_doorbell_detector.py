import numpy as np
import sounddevice as sd
import librosa
import queue
import time
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pygame.mixer
import firebase_admin
from firebase_admin import credentials, messaging

def initialize_firebase():
    
    if not firebase_admin._apps:
        cred = credentials.Certificate('config/doorbell-notification-14f52-ddf5adc0791f.json')
        return firebase_admin.initialize_app(cred)
    return firebase_admin.get_app()


class DoorbellDetector:
    def __init__(self, model_path='model/doorbell_classifier.h5', threshold=0.7):
        
        # Load the trained model
        self.model = load_model(model_path)
        self.threshold = threshold
        
        # Initialize label encoder
        self.le = LabelEncoder()
        self.le.fit(['background', 'doorbell'])
        
        # Initialize Firebase
        self.firebase_app = initialize_firebase()
        
        # Audio parameters
        self.sample_rate = 22050
        self.chunk_duration = 0.5  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.channels = 1
        
        # Buffer for collecting audio chunks
        self.audio_buffer = np.array([])
        self.buffer_duration = 2.0
        self.buffer_samples = int(self.sample_rate * self.buffer_duration)
        
        # Initialize audio queue
        self.audio_queue = queue.Queue()
        
        # Initialize pygame for alert sound
        pygame.mixer.init()
        
        # Instance variable to track the last notification time
        self.last_notification_time = 0
        self.notification_cooldown = 20  # Cooldown period to not send multiple notifications in (s)
        
    def process_audio(self, audio_data):
        
        try:
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure minimum length for FFT
            if len(audio_data) < 2048:
                audio_data = np.pad(audio_data, (0, 2048 - len(audio_data)))
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=40,
                n_fft=1024,  
                hop_length=512
            )
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return None

    def audio_callback(self, indata, frames, time, status):
        
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())

    def predict_audio(self, audio_data):
        
        features = self.process_audio(audio_data)
        if features is not None:
            pred_probs = self.model.predict(features, verbose=0)[0]
            pred_class = self.le.inverse_transform([np.argmax(pred_probs)])[0]
            confidence = float(pred_probs[self.le.transform(['doorbell'])[0]])
            return pred_class, confidence
        return None, 0.0

    def alert(self):
        
        current_time = time.time()
        
        # Check if enough time has passed since the last notification
        if current_time - self.last_notification_time < self.notification_cooldown:
            print("Notification cooldown active, skipping alert...")
            return
            
        print("\nDOORBELL DETECTED!")
        
        # Create message
        message = messaging.Message(
            notification=messaging.Notification(
                title='Doorbell Alert',
                body='Someone is at your door!'
            ),
            topic='doorbell_alerts'
        )
        
        # Send message
        try:
            response = messaging.send(message)
            print(f"Successfully sent notification: {response}")
            # Update the last notification time
            self.last_notification_time = current_time
        except Exception as e:
            print(f"Error sending notification: {e}")
        
        # Beep sound
        # sd.play(np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410)), 44100)
        # time.sleep(0.1)

    def start_listening(self):
        
        try:
            print("Starting doorbell detection...")
            
            # Start audio stream
            with sd.InputStream(
                callback=self.audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples
            ):
                while True:
                    try:
                        # Get audio data from queue
                        audio_chunk = self.audio_queue.get(timeout=1.0)
                        
                        # Add to buffer
                        self.audio_buffer = np.append(self.audio_buffer, audio_chunk.flatten())
                        
                        # Keep buffer at desired length
                        if len(self.audio_buffer) > self.buffer_samples:
                            # Process when buffer is full
                            pred_class, confidence = self.predict_audio(self.audio_buffer)
                            
                            if pred_class == 'doorbell' and confidence > self.threshold:
                                self.alert()
                            
                            # Reset buffer with overlap
                            overlap_samples = int(self.sample_rate * 0.1)  # 0.1 second overlap
                            self.audio_buffer = self.audio_buffer[-overlap_samples:]
                            
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Error in processing: {str(e)}")
                        continue
                    
        except Exception as e:
            print(f"Error in audio stream: {str(e)}")

def main():
    # Create detector instance
    detector = DoorbellDetector(threshold=0.7)
    
    # Start detection
    detector.start_listening()

if __name__ == "__main__":
    main()
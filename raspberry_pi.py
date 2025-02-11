import numpy as np
import sounddevice as sd
import librosa
import queue
import time
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, messaging
from sklearn.preprocessing import LabelEncoder
import signal
import sys
import os
import logging
from logging.handlers import RotatingFileHandler

class DoorbellDetector:
    def __init__(self, model_path='/home/pi/doorbell/model/doorbell_classifier.tflite', 
                 threshold=0.7,
                 firebase_cred_path='/home/pi/doorbell/config/doorbell-notification-14f52-ddf5adc0791f.json'):
       
        # Set up logging
        self.setup_logging()
        
        # Initialize TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.threshold = threshold
            logging.info("TFLite model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load TFLite model: {str(e)}")
            sys.exit(1)

        # Initialize label encoder
        self.le = LabelEncoder()
        self.le.fit(['background', 'doorbell'])

        # Initialize Firebase
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(firebase_cred_path)
                self.firebase_app = firebase_admin.initialize_app(cred)
            else:
                self.firebase_app = firebase_admin.get_app()
            logging.info("Firebase initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {str(e)}")
            sys.exit(1)

        # Audio parameters optimized for Raspberry Pi
        self.sample_rate = 22050
        self.chunk_duration = 0.5
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.channels = 1
        
        # Buffer configuration
        self.audio_buffer = np.array([])
        self.buffer_duration = 2.0
        self.buffer_samples = int(self.sample_rate * self.buffer_duration)
        self.audio_queue = queue.Queue()
        
        # Notification settings
        self.last_notification_time = 0
        self.notification_cooldown = 20
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
    
        log_dir = '/home/pi/doorbell/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        handler = RotatingFileHandler(
            f"{log_dir}/doorbell_detector.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[handler, logging.StreamHandler()]
        )

    def signal_handler(self, signum, frame):
        
        logging.info("Shutdown signal received. Cleaning up...")
        # Close audio stream if it exists
        if hasattr(self, 'stream'):
            self.stream.stop()
        sys.exit(0)

    def process_audio(self, audio_data):
        
        try:
            # Convert to mono if necessary
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure minimum length and handle silence
            if len(audio_data) < 2048:
                audio_data = np.pad(audio_data, (0, 2048 - len(audio_data)))
            
            # Check if the audio is too quiet
            if np.max(np.abs(audio_data)) < 0.01:
                return None
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=40,
                n_fft=1024,
                hop_length=512
            )
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled.reshape(1, -1, 1).astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            return None

    def predict_audio(self, audio_data):
        
        features = self.process_audio(audio_data)
        if features is not None:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], features)
                self.interpreter.invoke()
                pred_probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                pred_class = self.le.inverse_transform([np.argmax(pred_probs)])[0]
                confidence = float(pred_probs[self.le.transform(['doorbell'])[0]])
                return pred_class, confidence
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
        return None, 0.0

    def send_notification(self):
        
        current_time = time.time()
        
        if current_time - self.last_notification_time < self.notification_cooldown:
            logging.debug("Notification cooldown active")
            return
            
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title='Doorbell Alert',
                    body='Someone is at your door!'
                ),
                topic='doorbell_alerts'
            )
            response = messaging.send(message)
            logging.info(f"Notification sent successfully: {response}")
            self.last_notification_time = current_time
            
        except Exception as e:
            logging.error(f"Failed to send notification: {str(e)}")

    def audio_callback(self, indata, frames, time_info, status):
        """Handle incoming audio data"""
        if status:
            logging.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def start_listening(self):
        """Start the audio stream and process incoming sound"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                logging.info("Starting doorbell detection...")
                
                with sd.InputStream(
                    callback=self.audio_callback,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_samples,
                    device=None  # Let sounddevice choose the default input
                ) as self.stream:
                    
                    while True:
                        try:
                            audio_chunk = self.audio_queue.get(timeout=1.0)
                            self.audio_buffer = np.append(self.audio_buffer, audio_chunk.flatten())
                            
                            if len(self.audio_buffer) > self.buffer_samples:
                                pred_class, confidence = self.predict_audio(self.audio_buffer)
                                
                                if pred_class == 'doorbell' and confidence > self.threshold:
                                    logging.info(f"Doorbell detected with confidence: {confidence}")
                                    self.send_notification()
                                
                                # Keep a small overlap for continuous detection
                                overlap_samples = int(self.sample_rate * 0.1)
                                self.audio_buffer = self.audio_buffer[-overlap_samples:]
                                
                        except queue.Empty:
                            continue
                        except Exception as e:
                            logging.error(f"Error processing audio chunk: {str(e)}")
                            continue
                            
            except Exception as e:
                retry_count += 1
                logging.error(f"Audio stream error (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(5)  # Wait before retrying
                
        logging.critical("Max retries reached. Exiting...")
        sys.exit(1)

def main():
    # Create detector instance
    detector = DoorbellDetector()
    
    # Start detection
    detector.start_listening()

if __name__ == "__main__":
    main()
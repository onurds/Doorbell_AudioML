import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D

def load_audio_data(data_path):
   
    features = []
    labels = []
    
    # Target sampling rate
    target_sr = 22050
    
    # For all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file)
                # Get the class label from the parent directory name
                label = os.path.basename(root)
                
                try:
                    # Load and resample the audio file
                    audio, sample_rate = librosa.load(file_path, sr=target_sr)
                    
                    # Extract MFCC features
                    mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
                    mfccs_scaled = np.mean(mfccs.T, axis=0)
                    
                    # Append features and labels
                    features.append(mfccs_scaled)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return np.array(features), np.array(labels)

def build_model(input_shape, num_classes):
    
    # Building the 1D CNN model
    
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    # Second convolutional layer
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def make_prediction(model, le, file_path):
    
    # Make prediction on a single audio file
    
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return le.inverse_transform(predicted_class_index)[0]

def main():
    
    data_path = "sounds"
    
    print("Loading and preprocessing audio files...")
    features, labels = load_audio_data(data_path)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_onehot, test_size=0.2, random_state=42, stratify=labels_onehot
    )
    
    # Reshape the data for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build and compile the model
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape, len(le.classes_))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    print("Training the model...")
    batch_size = 32
    epochs = 50
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              epochs=epochs,
              validation_data=(X_test, y_test),
              verbose=1)
    
    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Save the model
    model.save('model/doorbell_classifier.h5')
    print("Model saved as 'doorbell_classifier.h5'")
    
    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    with open('model/doorbell_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model converted and saved as 'doorbell_classifier.tflite'")
    
    return model, le

if __name__ == "__main__":
    model, le = main()
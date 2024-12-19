import os

import librosa
import numpy as np

CLASS_MAPPINGS = {
    "0": "piccolo",
    "1": "clarinet",
    "2": "bass",
    "3": "flute",
    "4": "oboe",
    "5": "cello",
    "6": "violin",
    "7": "sax",
    "8": "trumpet",
}


def predict_class(audio_file, model):
    try:
        class_mappings = CLASS_MAPPINGS

        # Save the uploaded file
        temp_file_path = "temp_audio_file.wav"
        with open(temp_file_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Load the audio
        audio_data, sr = librosa.load(temp_file_path, sr=None)
        print(f"Audio loaded: {audio_data.shape} samples at {sr} Hz")

        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=128
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_resized = librosa.util.fix_length(
            mel_spectrogram, size=128, axis=1
        )
        mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=(0, -1))

        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs_resized = librosa.util.fix_length(mfccs, size=128, axis=1)
        mfccs_resized = np.expand_dims(mfccs_resized, axis=(0, -1))

        inputs = [mel_spectrogram_resized, mfccs_resized]

        # Prediction
        predictions = model.predict(inputs)
        print(f"Raw predictions: {predictions}")

        # Convert raw model output to percentage
        predictions_percentage = predictions[0] * 100
        print(f"Predictions in percentage: {predictions_percentage}")

        # Map predictions
        result = {}
        for i in range(len(predictions_percentage)):
            class_name = class_mappings.get(str(i), f"Unknown Class {i+1}")
            result[class_name] = f"{predictions_percentage[i]:.2f}%"

        sorted_result = dict(
            sorted(
                result.items(), key=lambda item: float(item[1].strip("%")), reverse=True
            )
        )

        print("Mapped Predictions (sorted):", sorted_result)
        return sorted_result

    except Exception as e:
        print(f"Error in predict_class: {e}")
        return {"error": "Prediction failed"}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

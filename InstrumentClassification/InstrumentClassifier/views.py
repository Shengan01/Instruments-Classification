import tensorflow as tf
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .utils import predict_class

model = tf.keras.models.load_model("../models/instrument_classifier_model.h5")

ALLOWED_EXTENSIONS = {"wav", "mp3"}
MAX_FILE_SIZE = 800 * 1024 * 1024


def index(request):
    return render(request, "index.html")


@csrf_exempt
def upload(request):
    if request.method == "POST" and request.FILES.get("file"):
        audio_file = request.FILES["file"]
        file_extension = audio_file.name.split(".")[-1].lower()

        if file_extension not in ALLOWED_EXTENSIONS:
            return JsonResponse({"error": "Invalid file format"}, status=400)

        if audio_file.size > MAX_FILE_SIZE:
            return JsonResponse({"error": "File too large"}, status=400)

        print(f"Received file: {audio_file.name}, {audio_file.size} bytes")

        try:
            predictions = predict_class(audio_file, model)
            return JsonResponse({"predictions": predictions})
        except Exception as e:
            return JsonResponse({"error": "Prediction failed"}, status=500)
    else:
        return JsonResponse({"error": "No file provided"}, status=400)

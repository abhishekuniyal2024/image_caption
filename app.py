from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import pickle

# Paths to your model and tokenizer files
model_path = '/mnt/data/abhishek/data2/Ducat Machine Learning/image caption generater/model.keras'
tokenizer_path = '/mnt/data/abhishek/data2/Ducat Machine Learning/image caption generater/tokenizer.pkl'
feature_extractor_path = '/mnt/data/abhishek/data2/Ducat Machine Learning/image caption generater/feature_extractor.keras'

# Load models and tokenizer once at startup
caption_model = load_model(model_path)
feature_extractor = load_model(feature_extractor_path)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34
img_size = 224

# Initialize FastAPI app
app = FastAPI()

# Root endpoint to confirm API is running
@app.get("/")
def read_root():
    return {"message": "Image Caption Generator API is running. Use /docs to test."}

# Helper function to generate caption from image array
def generate_caption(image_array):
    image_features = feature_extractor.predict(image_array, verbose=0)

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return caption

# POST endpoint to receive image file and return generated caption
@app.post("/generate-caption")
async def get_caption(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((img_size, img_size))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        caption = generate_caption(image_array)
        return JSONResponse(content={"caption": caption})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

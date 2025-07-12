# IMPORTS 

import pandas as pd
import os, json, csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

# PATHS
base_path = "kaggle_model/"
resnet_weights_path = base_path + "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
model_path = base_path + "model.h5"
captions_path = base_path + "captions.csv"

MAX_LEN = 20

# Load captions
captions = pd.read_csv(captions_path)



# LOAD MODEL AND DATA

# Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(captions['productDisplayName'])
vocab_size = len(tokenizer.word_index) + 1

# Models
cnn = ResNet50(weights=resnet_weights_path, include_top=False, pooling='avg')
model = tf.keras.models.load_model(model_path)





# Extract features from image

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return cnn.predict(x)

# def generate_caption(image_path):
#     feature = extract_features(image_path)
#     in_text = '<start>'
#     for _ in range(MAX_LEN):
#         seq = tokenizer.texts_to_sequences([in_text])[0]
#         seq = pad_sequences([seq], maxlen=MAX_LEN)
#         yhat = model.predict([feature, seq], verbose=0)
#         next_word = tokenizer.index_word[np.argmax(yhat)]
#         if next_word == '<end>' or next_word is None:
#             break
#         in_text += ' ' + next_word
#     return in_text


# def generate_caption_topk(image_path, k=3):

#     # 1. Extract CNN features
#     feature = extract_features(image_path)

#     # 2. Start caption
#     in_text = "<start>"
#     topk_history = []

#     for _ in range(7):
#         # Tokenise current partial caption
#         seq = tokenizer.texts_to_sequences([in_text])[0]
#         seq = pad_sequences([seq], maxlen=MAX_LEN)

#         # Predict next-token distribution
#         yhat = model.predict([feature, seq], verbose=0)[0]  # shape = (vocab_size,)

#         # --- keep the k highest probabilities ---
#         topk_idx = np.argsort(yhat)[-k:][::-1]              # indices, highest → lowest
#         topk_tokens = [(tokenizer.index_word.get(idx, "<unk>"), float(yhat[idx]))
#                        for idx in topk_idx]
#         topk_history.append(topk_tokens)

#         # Greedy choice for caption string
#         next_word = topk_tokens[0][0]

#         if next_word == "<end>" or next_word is None:
#             break
#         in_text += " " + next_word

#     # Strip the leading <start> token for readability
#     caption = in_text.replace("<start> ", "")

#     return caption, topk_history



def generate_caption_topk_norepeat(image_path, k=3):
    # ---- feature vector -----------------------------------------------------
    feature = extract_features(image_path)
    # ---- book-keeping --------------------------------------------------------
    in_tokens   = ["<start>"]          # growing caption as a list
    used_words  = set()                # set of used words
    topk_history = []

    # ---- decoding loop -------------------------------------------------------
    for _ in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([" ".join(in_tokens)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)

        yhat = model.predict([feature, seq], verbose=0)[0]   # (vocab_size,)

        # best-k indices, high → low
        topk_idx = np.argsort(yhat)[-k:][::-1]
        topk_tokens = [(tokenizer.index_word.get(i, "<unk>"), float(yhat[i]))
                       for i in topk_idx]
        topk_history.append(topk_tokens)

        # pick the highest-probability *new* word
        next_word = None
        for tok, _ in topk_tokens:
            if tok not in used_words:          # never used before?
                next_word = tok
                break
        # stop if no fresh word or reached the <end> token
        if next_word is None or next_word == "<end>":
            break
        # update caption state
        in_tokens.append(next_word)
        used_words.add(next_word)
    # convert list → string and drop <start>
    caption = " ".join(in_tokens[1:])

    return caption, topk_history



# APP
# ────────────────────────────────────────────────────────────────────
# 4.  Flask app
# ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"API": "running", "send": "POST /predict"}), 200

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400
#     try:
#         file = request.files["image"]            # <FileStorage>
#         if file.filename == "":
#             return jsonify({"error": "Empty filename"}), 400

#         # 2) save to a temp file --------------------------------------------------
#         #    This gives us a real filesystem path for the existing caption code.
#         suffix = os.path.splitext(secure_filename(file.filename))[1] or ".jpg"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             file.save(tmp.name)
#             tmp_path = tmp.name

#         try:
#             # 3) call your unchanged caption function ----------------------------
#             caption, topk = generate_caption_topk_norepeat(tmp_path, k=1)
#         finally:
#             # 4) clean up the temp file ------------------------------------------
#             os.remove(tmp_path)

#         # 5) respond --------------------------------------------------------------
#         print("Caption:", caption)
#         return jsonify({"caption": caption}), 200
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"error": str(e)}), 500
# # ──────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save to a temp file
        suffix = os.path.splitext(secure_filename(file.filename))[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Generate caption
            caption, topk = generate_caption_topk_norepeat(tmp_path, k=1)
        finally:
            os.remove(tmp_path)

        # Remove <unk> tokens
        caption_cleaned = " ".join([word for word in caption.split() if word != "<unk>"])

        print("Caption:", caption_cleaned)
        return jsonify({"caption": caption_cleaned}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)



# image_pth = 'test.jpg'  

# caption, topk = generate_caption_topk_norepeat("test.jpg", k=1)

# print("Caption:", caption)
# for step, candidates in enumerate(topk, 1):
#     print(f"Step {step}:")
#     for token, prob in candidates:
#         print(f"  {token:15s}  p={prob:.4f}")




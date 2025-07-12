# import tensorflow as tf
# import numpy as np
# import cv2
# import json

# # Load model and vocab
# model = tf.keras.models.load_model("model.keras")

# with open("index_to_word.json", "r") as f:
#     index_to_word = json.load(f)

# with open("word_to_index.json", "r") as f:
#     word_to_index = json.load(f)

# MAX_LENGTH = 9
# START_TOKEN = word_to_index.get("<start>", 482)
# END_TOKEN = word_to_index.get("<end>", 481)

# def preprocess_image(filepath):
#     image = cv2.imread(filepath)
#     image = cv2.resize(image, (180, 180))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image.astype('float32') / 255.0
#     return np.expand_dims(image, axis=0)

# def generate_caption(image):
#     caption_input = np.zeros((1, MAX_LENGTH), dtype=np.int32)
#     caption_input[0, 0] = START_TOKEN
#     tokens = []  # Stores all predicted tokens, including special ones
#     result = []  # Stores only valid words for the final caption

#     for i in range(1, MAX_LENGTH):
#         preds = model.predict([image, caption_input], verbose=0)
#         probs = tf.nn.softmax(preds[0, i]).numpy()
#         top_indices = probs.argsort()[-5:][::-1]
#         for idx in top_indices:
#             print(f"  {idx}: {index_to_word.get(str(idx), '<unk>')} ({probs[idx]:.4f})")
#         token_id = np.argmax(probs)
#         word = index_to_word.get(str(token_id), "<unk>")

#         tokens.append(word)  # Collect every token predicted
#         print(f"Step {i}: token_id={token_id}, word={word}")

#         if word == "<end>":
#             break
#         if word not in ["<start>", "<unk>"]:
#             result.append(word)
#             caption_input[0, i] = token_id
#         else:
#             caption_input[0, i] = 0  # Avoid leaving a gap

#     print(" All tokens:", " ".join(tokens))
#     return " ".join(result).strip().capitalize() + "."
# # Run on image
# image = preprocess_image("10014.jpg")
# caption = generate_caption(image)
# print(" Final Caption:", caption)


#!/usr/bin/env python3
"""
Run:  python test_caption.py path/to/image.jpg
Outputs: "Blue Nike Cap."
────────────────────────────────────────────────────────────────
Folder must contain:
  • model.h5                              (your saved caption model)
  • resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
  • index_to_word.json / word_to_index.json   (re-created via step-1)
"""

import sys, json, numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ─── paths ─────────────────────────────────────────────────────
MODEL_PATH   = "kaggle_model/model.h5"
RESNET_WGTS  = "kaggle_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
IMG_SIZE     = (224, 224)               # ResNet-50 default
MAX_LEN      = 20                      

# ─── load tokenizer mapping ───────────────────────────────────
with open("kaggle_model/index_to_word.json") as f:
    index_to_word = json.load(f)
with open("kaggle_model/word_to_index.json") as f:
    word_to_index = json.load(f)

START_ID = word_to_index.get("<start>") or word_to_index.get("<sos>", 1)
END_ID   = word_to_index.get("<end>")   or word_to_index.get("<eos>", 2)

# ─── feature extractor (same as notebook) ─────────────────────
cnn = ResNet50(weights=None, include_top=False, pooling="avg")
cnn.load_weights(RESNET_WGTS)

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img).astype("float32"), 0)
    arr = preprocess_input(arr)              # ResNet scaling
    return cnn.predict(arr, verbose=0)       # (1, 2048)

# ─── load caption model ───────────────────────────────────────
cap_model = tf.keras.models.load_model(MODEL_PATH)

# ─── greedy decoding loop ─────────────────────────────────────
# def generate_caption(img_feat):
#     seq = np.zeros((1, MAX_LEN), dtype=np.float32)
#     seq[0, 0] = float(START_ID)
#     words = []

#     for t in range(1, MAX_LEN):
#         preds = cap_model.predict([img_feat, seq], verbose=0)[0, t]
#         token = int(np.argmax(preds))
#         if token == END_ID:
#             break
#         word = index_to_word.get(str(token), "<unk>")
#         if word == "<unk>":
#             break
#         words.append(word)
#         seq[0, t] = float(token)
#     return (" ".join(words)).capitalize() + "."

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
# tokenizer.fit_on_texts(captions['productDisplayName'])
# vocab_size = len(tokenizer.word_index) + 1

def generate_caption(image_path):
    feature = extract_features(image_path)
    in_text = '<start>'
    for _ in range(MAX_LEN):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = model.predict([feature, seq], verbose=0)
        next_word = tokenizer.index_word[np.argmax(yhat)]
        if next_word == '<end>' or next_word is None:
            break
        in_text += ' ' + next_word
    return in_text
# ─── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_caption.py image.jpg"); sys.exit(1)
    features = extract_features(sys.argv[1])
    print(generate_caption(features))
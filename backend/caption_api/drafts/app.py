# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import json
# import cv2

# # Load index_to_word mapping
# with open("model_test_2/index_to_word.json", "r") as f:
#     index_to_word = json.load(f)
# print("Vocabulary size:", len(index_to_word))

# # Add "<start>" and "<end>" tokens if not present
# if "0" not in index_to_word:
#     index_to_word["0"] = "<start>"
# if "1" not in index_to_word:
#     index_to_word["1"] = "<end>"

# # Reverse dictionary for convenience
# word_to_index = {v: int(k) for k, v in index_to_word.items()}

# app = Flask(__name__)

# # Load TFLite model
# # interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
# # interpreter.allocate_tensors()
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# # Load the model
# model = tf.keras.models.load_model("model_test_2/model.keras")

# # Show model summary to understand architecture
# model.summary()

# # Get input and output shapes
# input_shape = model.input_shape
# output_shape = model.output_shape
# input_details = input_shape
# output_details = output_shape


# print(f"\nReal model input shape: {input_shape}")
# print(f"Real model output shape: {output_shape}")

# interpreter = model

# print("\n MODEL INPUTS / OUTPUTS")
# for i, inp in enumerate(model.inputs):
#     print(f"Inp[{i}] name={inp.name:<20} shape={inp.shape} dtype={inp.dtype}")

# print("Server is running and model is ready...")

# # def preprocess_image(image_bytes):
# #     try:
# #         img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((180, 180))
# #     except Exception as e:
# #         print("Error loading image:", e)
# #         return None
# #     img = np.array(img).astype('float32') / 255.0
# #     img = np.expand_dims(img, axis=0)
# #     return img


# def preprocess_image(image_bytes):
#     try:
#         # Convert bytes to a NumPy array
#         nparr = np.frombuffer(image_bytes, np.uint8)

#         # Decode the image using OpenCV
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if image is None:
#             raise ValueError("Could not decode image from bytes.")

#         # Resize and convert to RGB (OpenCV loads in BGR by default)
#         image = cv2.resize(image, (180, 180))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Normalize pixel values to [0, 1]
#         image = image.astype('float32') / 255.0

#         # Add batch dimension
#         image = np.expand_dims(image, axis=0)
#         return image

#     except Exception as e:
#         print("Error preprocessing image:", e)
#         return None


# @app.route('/', methods=['GET'])
# def test():
#     return jsonify({'API RUNNING': True, 'message': 'Upload an image to get a caption.'}), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image = request.files['image'].read()
#     input_image = preprocess_image(image)
#     if input_image is None:
#         return jsonify({'error': 'Invalid image format'}), 400

#     print("Input 0 shape:", input_details[0]['shape'])
#     print("Input 1 shape:", input_details[1]['shape'])

#     if list(input_details[0]['shape'][1:]) == [180, 180, 3]:
#         image_index = input_details[0]['index']
#         text_index = input_details[1]['index']
#         max_length = input_details[1]['shape'][1]
#     else:
#         image_index = input_details[1]['index']
#         text_index = input_details[0]['index']
#         max_length = input_details[0]['shape'][1]

#     caption_input = np.zeros((1, max_length), dtype=np.float32)
#     caption_input[0, 0] = word_to_index.get("<start>", 0)

#     import random

#     def sample_with_temperature(probs, temperature=1.0):
#         # Apply temperature scaling
#         probs = np.log(probs + 1e-10) / temperature
#         exp_probs = np.exp(probs)
#         probs = exp_probs / np.sum(exp_probs)
#         return np.random.choice(len(probs), p=probs)

#     result = []
#     temperature = 1.0  # You can tweak this (e.g., 0.7 for more confident, 1.2 for more random)

#     for i in range(1, max_length):
#         interpreter.set_tensor(image_index, input_image)
#         interpreter.set_tensor(text_index, caption_input)
#         interpreter.invoke()

#         output = interpreter.get_tensor(output_details[0]['index'])
#         logits = output[0, i]

#         probs = tf.nn.softmax(logits).numpy()

#         print(f"Step {i}: Vocabulary tokens with probability > 0.001:")
#         for idx, prob in enumerate(probs):
#             if prob > 0.001:
#                 print(f"  Token ID {idx}: {index_to_word.get(str(idx), '<unk>')} - Probability: {prob:.4f}")

#         token_id = sample_with_temperature(probs, temperature)
#         word = index_to_word.get(str(token_id), '<unk>')

#         print(f"Step {i}: token_id={token_id}, sampled word={word}")

#         if word == "<end>":
#             print("End token predicted, stopping.")
#             break
#         if word == "<start>" and i > 1:
#             print("Repeated start token detected, stopping.")
#             break   
#         if word != "<start>":
#             result.append(word)

#         caption_input[0, i] = token_id

#     caption = ' '.join(result).capitalize() + "."
#     print("Final caption:", caption)
#     # caption = "Blue Nike Cap"
#     return jsonify({'caption': caption})

# @app.route('/predict', methods=['POST'])
# def predict():
#     # ---------------------------------------------------------------
#     # 1. Get & preprocess image
#     # ---------------------------------------------------------------
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image_bytes = request.files['image'].read()
#     image_input = preprocess_image(image_bytes)  # shape (1, 180, 180, 3)
#     if image_input is None:
#         return jsonify({'error': 'Invalid image format'}), 400

#     # If the model expects features, uncomment ↓ and comment the line above
#     # image_input = cnn_feature_extractor(image_bytes)  # shape (1, 2048)

#     # ---------------------------------------------------------------
#     # 2. Prepare caption seed
#     # ---------------------------------------------------------------
#     # If the sequence length is variable (None), hard-code MAX_LEN from training
#     MAX_LEN = model.inputs[1].shape[1] or 9        # fallback to 9 if None
#     caption_seq = np.zeros((1, MAX_LEN), dtype=np.int32)
#     caption_seq[0, 0] = word_to_index.get("<start>", 0)

#     # ---------------------------------------------------------------
#     # 3. Generate caption
#     # ---------------------------------------------------------------
#     temperature = 1.0
#     result = []

#     def sample_softmax(logits, T=1.0):
#         probs = tf.nn.softmax(logits / T).numpy()
#         return np.random.choice(len(probs), p=probs)

#     for t in range(1, MAX_LEN):
#         preds = model.predict([image_input, caption_seq], verbose=0)
#         logits = preds[0, t]                        # (VOCAB_SIZE,)
#         token_id = sample_softmax(logits, temperature)
#         word = index_to_word.get(str(token_id), "<unk>")

#         if word in {"<end>", "<start>"}:
#             break
#         result.append(word)
#         caption_seq[0, t] = token_id

#     caption = " ".join(result).strip().capitalize() + "."
#     return jsonify({"caption": caption})

# if __name__ == '__main__':
#     # app.run(debug=True, port=5001)
#     app.run(host='0.0.0.0', debug=True ,  port=5001)


# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import json
# import cv2

# app = Flask(__name__)

# # Load vocab files
# with open("index_to_word.json", "r") as f:
#     index_to_word = json.load(f)

# with open("word_to_index.json", "r") as f:
#     word_to_index = json.load(f)

# # Constants
# MAX_LENGTH = 9
# START_TOKEN = word_to_index.get("<start>", 482)
# END_TOKEN = word_to_index.get("<end>", 481)

# # Load Keras model
# model = tf.keras.models.load_model("model.keras")
# print(" Keras model loaded.")

# def preprocess_image(image_bytes):
#     try:
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if image is None:
#             raise ValueError("Could not decode image.")
#         image = cv2.resize(image, (180, 180))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = image.astype('float32') / 255.0
#         return np.expand_dims(image, axis=0)
#     except Exception as e:
#         print("Error preprocessing image:", e)
#         return None

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({'API RUNNING': True}), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image = request.files['image'].read()
#     image_input = preprocess_image(image)
#     if image_input is None:
#         return jsonify({'error': 'Invalid image format'}), 400

#     caption_input = np.zeros((1, MAX_LENGTH), dtype=np.float32)
#     caption_input[0, 0] = float(START_TOKEN)

#     result = []
#     repeated_start_count = 0

#     print(" Starting caption generation...\n")

#     for i in range(1, MAX_LENGTH):
#         predictions = model.predict([image_input, caption_input], verbose=0)
#         logits = predictions[0, i]
#         probs = tf.nn.softmax(logits).numpy()

#         token_id = np.argmax(probs)
#         word = index_to_word.get(str(token_id), "<unk>")

#         print(f"Step {i}: token_id={token_id}, word={word}, prob={probs[token_id]:.4f}")

#         # Show top-5 predictions for this step
#         top5 = np.argsort(probs)[-5:][::-1]
#         print("Top 5 predictions:")
#         for tid in top5:
#             tok = index_to_word.get(str(tid), "<unk>")
#             print(f"  {tid}: {tok} ({probs[tid]:.4f})")

#         if word == "<end>":
#             print(" Stopping: <end> token predicted.")
#             break
#         if word == "<start>":
#             repeated_start_count += 1
#             if repeated_start_count > 2:
#                 print(" Looping on <start>, stopping.")
#                 break
#             continue
#         if word == "<unk>":
#             print(" Unknown token, skipping.")
#             continue

#         result.append(word)
#         caption_input[0, i] = float(token_id)

#     if not result:
#         caption = ""
#     else:
#         caption = " ".join(result).strip().capitalize() + "."

#     print(f" Final caption: {caption}\n")
#     return jsonify({'caption': caption})

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)



"""
Flask caption-generation API
CNN-encoder + LSTM-decoder saved as model_test_2/model.keras
Assumes the model was trained on 180×180 RGB images rescaled to [0,1]
----------------------------------------------------------------------
Run:  python app.py
POST: curl -F "image=@blue_cap.jpg" http://127.0.0.1:5001/predict
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json

# ────────────────────────────────────────────────────────────────────
# 1.  Vocabulary
# ────────────────────────────────────────────────────────────────────
# ───────── vocab & special IDs ──────────────────────────
with open("model_test_2/index_to_word.json", "r") as f:
    index_to_word = json.load(f)
word_to_index = {v: int(k) for k, v in index_to_word.items()}

def find_special_id(token_names):
    for tok in token_names:
        for k, v in index_to_word.items():
            if v.lower() == tok.lower():
                return int(k)
    return None

START_ID = find_special_id(["<start>", "<sos>", "<bos>"]) or 0
END_ID   = find_special_id(["<end>", "<eos>"])

print(f"START_ID={START_ID}  END_ID={END_ID}")

# ------------ Place directly under the line that opens index_to_word.json --------
# Detect the IDs that were actually used during training
def _find_id(token_variants):
    for t in token_variants:
        for k, v in index_to_word.items():
            if v.lower() == t.lower():
                return int(k)
    return None

START_ID = _find_id(["<start>", "<sos>", "<bos>"]) or 0
END_ID   = _find_id(["<end>", "<eos>", "</s>"])

print(f"[INFO]  START_ID = {START_ID}   END_ID = {END_ID}")
# ---------------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────
# 2.  Model
# ────────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("model_test_2/model.keras")
print("  Model summary")
model.summary()

IMG_SHAPE  = model.inputs[0].shape[1:4]    # (180, 180, 3)
MAX_LEN    = model.inputs[1].shape[1] or 8 # fallback if None

import json, re

def rebuild_vocab_from_model(loaded_model):
    # 1. Check for a TextVectorization layer
    for lyr in loaded_model.layers:
        if isinstance(lyr, tf.keras.layers.TextVectorization):
            vocab = lyr.get_vocabulary()           # list of strings
            return {str(i): w for i, w in enumerate(vocab)}

    # 2. Fall back: look for vocab in Embedding config (TF 2.14+)
    for lyr in loaded_model.layers:
        if isinstance(lyr, tf.keras.layers.Embedding):
            cfg = lyr.get_config()
            if "vocabulary" in cfg:                # new Keras API
                vocab = cfg["vocabulary"]          # already list[str]
                return {str(i): w for i, w in enumerate(vocab)}
    
    return None

auto_vocab = rebuild_vocab_from_model(model)
if auto_vocab:
    print(f"[INFO]  Recovered vocab of size {len(auto_vocab)} from model.")
    index_to_word = auto_vocab
    # Optional: overwrite JSON on disk so you keep it next run
    with open("model_test_2/index_to_word_recovered.json", "w") as f:
        json.dump(index_to_word, f)
else:
    print(" Could not extract vocabulary from model. "
          "You will need the original tokenizer JSON.")
    

word_to_index = {w: int(i) for i, w in index_to_word.items()}

START_ID = word_to_index.get("<start>") or word_to_index.get("<sos>") or 0
END_ID   = word_to_index.get("<end>")   or word_to_index.get("<eos>")

print(f"[INFO]  START_ID={START_ID}  END_ID={END_ID}")
# ────────────────────────────────────────────────────────────────────
# 3.  Pre-processing (match training!)
#     If you used tf.keras.applications preprocessing,
#     swap the /255.0 line for that function.
# ────────────────────────────────────────────────────────────────────
# def preprocess_image(image_bytes):
#     """Bytes → (1, H, W, 3) float32  scaled to [0,1]"""
#     img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Could not decode image.")
#     img = cv2.resize(img, (IMG_SHAPE[1], IMG_SHAPE[0]))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
#     return np.expand_dims(img, 0)
from tensorflow.keras.applications.resnet       import preprocess_input as cnn_preproc
def preprocess_image(b):
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (180, 180))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img = cnn_preproc(img)        # <-- critical difference
    return np.expand_dims(img, 0)


try:
    SAMPLE_PATH = "test.jpg"   # <-- drop the same image in your project folder
    with open(SAMPLE_PATH, "rb") as f:
        img_bytes = f.read()

    # All candidate preprocess functions you might have trained with
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preproc
    from tensorflow.keras.applications.resnet       import preprocess_input as res_preproc
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preproc

    def raw255(x): return x.astype("float32") / 255.0

    candidates = [
        ("raw /255", raw255),
        ("EfficientNet", eff_preproc),
        ("ResNet",       res_preproc),
        ("MobileNetV2",  mob_preproc),
    ]

    blue_id = next(int(k) for k, v in index_to_word.items() if v.lower() == "blue")

    seq0 = np.zeros((1, MAX_LEN), dtype=np.float32)
    seq0[0, 0] = float(START_ID)

    print("\n🔬  PROBABILITY OF 'blue' AFTER STEP-1")
    for name, fn in candidates:
        # preprocess image with candidate fn
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (180, 180))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = fn(img)                       # candidate scaling
        img = np.expand_dims(img, 0)

        logits = model.predict([img, seq0], 0)[0, 1]
        prob   = tf.nn.softmax(logits)[blue_id].numpy()
        print(f"{name:<12}:  {prob:.4f}")
except Exception as e:
    print(f"[DEBUG] Skipping auto-probe: {e}")

# ────────────────────────────────────────────────────────────────────
# 4.  Flask app
# ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"API": "running", "send": "POST /predict"}), 200

def sample_argmax(logits):            # greedy
    return int(np.argmax(logits))

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    img = preprocess_image(request.files["image"].read())

    MAX_LEN = model.inputs[1].shape[1] or 8      # keep
    seq = np.zeros((1, MAX_LEN), dtype=np.float32)
    seq[0, 0] = float(START_ID)

    words = []
    for t in range(1, MAX_LEN):
        logits = model.predict([img, seq], verbose=0)[0, t]

        # ---------- DEBUG block (prints once at t==1) ----------
        if t == 1:
            top = np.argsort(logits)[-10:][::-1]
            print("TOP-10 step-1:",
                [f"{id}:{index_to_word.get(str(id),'?')}" for id in top])
        # -------------------------------------------------------

        tok  = int(np.argmax(logits))               # greedy decode
        if END_ID is not None and tok == END_ID:
            break
        word = index_to_word.get(str(tok), "<unk>")
        if word == "<unk>":
            break                                   # safety stop
        words.append(word)
        seq[0, t] = float(tok)
    caption = " ".join(words).capitalize() + "."
    return jsonify({"caption": caption})
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
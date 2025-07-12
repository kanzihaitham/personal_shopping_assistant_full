import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT DETAILS:")
for i, detail in enumerate(input_details):
    print(f"Input {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")

print("\nOUTPUT DETAILS:")
for i, detail in enumerate(output_details):
    print(f"Output {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")
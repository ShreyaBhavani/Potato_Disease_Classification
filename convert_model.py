import tensorflow as tf

# Load keras model
model = tf.keras.models.load_model(
    r"E:\Potato_disease\saved_models\1\1.keras"
)

# Export for TensorFlow Serving
model.export(r"E:\Potato_disease\saved_models\2")

print("Model converted successfully!")
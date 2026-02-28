import tensorflow as tf

# Load keras model (existing .keras file)
model = tf.keras.models.load_model(
    r"E:\Potato_disease\saved_models\keras_model\1.keras"
)

# Export for TensorFlow Serving
model.export(r"E:\Potato_disease\saved_models\potato_disease_model")

print("Model converted successfully!")
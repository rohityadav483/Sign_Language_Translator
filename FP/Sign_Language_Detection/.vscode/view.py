from tensorflow.keras.models import load_model

print("Script started")

model = load_model('cnn8grps_rad1_model.h5')

# Show model architecture
model.summary()

# View layer configurations
for layer in model.layers:
    print(f"\nLayer: {layer.name}")
    print("Config:", layer.get_config())
    weights = layer.get_weights()
    print(f"Weights shape: {[w.shape for w in weights]}")

for layer in model.layers:
    print(f"\n🔸 Layer: {layer.name}")
    print("Config:", layer.get_config())


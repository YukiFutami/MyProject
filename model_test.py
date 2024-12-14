from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_dir):
    print("[INFO] Loading the model...")
    model = load_model(model_path)

    print("[INFO] Preparing test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print("[INFO] Evaluating the model...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"[RESULTS] Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model('trained_model.h5', 'dataset/test')


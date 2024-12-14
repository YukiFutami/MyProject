import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model():
    print("[INFO] Creating the model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3クラス分類
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("[INFO] Model created successfully.")
    return model

def prepare_data(train_dir, validation_dir):
    print("[INFO] Preparing the data...")

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    print("[INFO] Data prepared successfully.")
    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator, epochs=5):
    print("[INFO] Starting training...")

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    print("[INFO] Training complete.")
    
    # モデルの保存
    print("[INFO] Saving the trained model...")
    model.save('trained_model.h5')  # モデルを保存
    print("[INFO] Model saved as 'trained_model.h5'")
    
    return history

def main():
    train_dir = 'dataset/train'  # 訓練データのディレクトリ
    validation_dir = 'dataset/validation'  # 検証データのディレクトリ

    model = create_model()
    train_generator, validation_generator = prepare_data(train_dir, validation_dir)
    history = train_model(model, train_generator, validation_generator, epochs=5)

if __name__ == "__main__":
    main()


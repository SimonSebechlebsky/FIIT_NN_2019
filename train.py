import os
import argparse

from tensorflow import keras

from model import get_model
from data_manipulation import get_image_generators,  IMAGE_DIM, ROOT_DIR
from utils import logger

CHECKPOINT_DIR =  os.path.join(ROOT_DIR, 'best_models')


def get_run_name(config):
    return f'{config["run_name"]}-lr{config["learning_rate"]}'


def get_callbacks(config):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, f'{get_run_name(config)}.h5'),
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(ROOT_DIR, f'logs/{get_run_name(config)}'),
            histogram_freq=1,
            write_grads=False,
            write_images=False,
            update_freq='epoch')
    ]
    return callbacks


def train_model(config):
    train_generator, test_generator = get_image_generators(config)
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.build(input_shape=(None, IMAGE_DIM, IMAGE_DIM, 3))
    model.summary()
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        epochs=config['epochs'],
                        callbacks=get_callbacks(config)
                        )


def get_summary():
    model = get_model()
    model.build(input_shape=(None, IMAGE_DIM, IMAGE_DIM, 3))
    model.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-lr', '--learning-rate', type=float, required=True)
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-r', '--run-name', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=32)

    args = parser.parse_args()
    args_dict = vars(args)
    logger.info(f'Starting training with arguments: {str(args_dict)}')
    train_model(args_dict)

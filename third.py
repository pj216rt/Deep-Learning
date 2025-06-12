#underfitting and overfitting
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

#translation:
#if there has not been an improvement in the validation loss of 0.001 ove the last 20 epochs, stop training and
#keep the best model up to that point
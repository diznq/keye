# Keye

Machine learning based eye-tracking using webcam

## Creating dataset
In order to create dataset, use `src/capture.py` after which your eye will follow mouse cursor position (for now Windows only).
After you are done, press Q to stop capturing

## Training model
To train model, use `model.ipynb` and run the training.

## Predictions
To make predictions, run `src/pred.py` after which based on prediction, mouse cursor will be set to predicted position.
# AI-Based Hand Gesture Recognition System

## Project Overview
This project uses computer vision and machine learning to recognize predefined hand gestures through a webcam in real time.

## Gestures Supported
- FIST
- OPEN_PALM
- PEACE
- THUMBS_UP

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy and Pandas
- Scikit-learn
- Random Forest Classifier
- Matplotlib

## How It Works
1. The webcam captures live video.
2. MediaPipe detects hand landmarks.
3. The landmark coordinates are used as model features.
4. A trained Random Forest classifier predicts the gesture.
5. The predicted gesture is displayed on the video window.

## Project Setup
1. Install Python 3.12.
2. Create and activate a virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run `python collect_data.py` to collect gesture samples.
5. Run `python train_model.py` to train the classifier.
6. Run `python app.py` to start real-time recognition.

## Evaluation
The training script generates a classification report and confusion matrix. Actual evaluation results should be documented after running the model.

## Dataset
The dataset is collected from webcam hand landmarks and stored in `data/gesture_data.csv`.

## Future Improvements
- Collect samples from multiple people and lighting conditions.
- Test on separate recordings to estimate real-world performance.
- Add more gestures and a graphical user interface.

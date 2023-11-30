# Face Emotion Recognition using Oneshot Learning Strategy

Face Emotion Recognition is a simple web application built with Flask for recognizing human faces and detecting their emotions. The project utilizes OpenCV for image processing, capturing frames from a camera, and haarcascade_frontalface_default to crop the coordinates of the detected faces.

## Features

- Face recognition: The application implements a one-shot learning approach by extracting embeddings values from a pretrained Facenet model for comparison with detected faces.
- Emotion classification: The application uses a fine-tuned Facenet model with the jonathanoheix/face-expression-recognition-dataset from Kaggle for classifying emotions.

## Usage

To use this application, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required dependencies as mentioned in the project documentation.
3. Run the Flask application.
4. Access the application through a web browser.
5. Allow camera access and view real-time face detection and emotion recognition.

## Project Structure

The project structure is as follows:

- `app.py`: The main Flask application script.
- `templates/`: Directory containing HTML templates for rendering web pages.
- `haarcascade_frontalface_default.xml`: XML file for face detection using Haar cascade classifier.
- `/model_embedding_save/`: Directory for storing the pretrained Facenet model and emotion classification model.
- `requirements.txt`: File specifying the required dependencies.
- `trainning/`: Directory for main work, trainning model extracting feature embeddings.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request.

## Support

If you encounter any issues or have any questions or concerns, please feel free to contact the project maintainers via the repository's issue tracker.



![Alt Text](https://github.com/khanhvovan2002/face-emotion-regconition_/blob/main/OKE.gif)

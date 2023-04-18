import cv2
from flask import Flask, render_template, Response, request
import torch
from facenet_pytorch import InceptionResnetV1
import pickle
from torchvision import transforms
from model import MyModel
# Initialize Flask app
app = Flask(__name__)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
state_dict = torch.load('model_embedding_save/emotion.pt', map_location=torch.device('cpu'))
facenet_expression_model = MyModel()
facenet_expression_model.load_state_dict(state_dict)
facenet_expression_model.eval()

normalize = transforms.Compose([
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


# Open a video capture object to access the camera
cap = cv2.VideoCapture(0)  # 0 indicates the default camera
similarity_threshold = torch.tensor(0.750)
known_embeddings = {}
labels = ['sad', 'disgust', 'neutral', 'happy', 'surprise', 'angry', 'fear']
with open('model_embedding_save/face_embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)
label = ""
def generate_frames():
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.14, minNeighbors=6, minSize=(200, 200))

        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            # Extract the face region
            face = frame[y:y+h, x:x+w]
            # Convert the face to RGB and resize it
            face = cv2.resize(face, (160, 160))  # Resize face to match input size of the model
            face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float()  # Convert face to torch tensor
            face = face / 255.0
            # face = normalize(face)
            face = normalize(face)
            # Extract face embeddings using Facenet model
            with torch.no_grad():
                emb = facenet_model(face)   
                emb_ = facenet_expression_model(face)
            similarity_scores = torch.cosine_similarity(emb, torch.stack(list(known_embeddings.values())).squeeze(), dim=1)
            max_values, id = similarity_scores.max(dim=0)
            comparison = max_values > similarity_threshold
            if comparison:
                recognized = True
                face_id = list(known_embeddings.keys())[id]
                face_id = face_id.split('_')[0]
                predictions = torch.argmax(emb_, dim=1)  # Get the predicted class label
                label = labels[predictions]

            else:
                recognized = False
                face_id = None
            # Draw rectangles around the detected faces, with different colors based on recognition status
            color = (0, 255, 0) if recognized else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw face name on the frame
            cv2.putText(frame, face_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, label, (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    app.run(port = 5005)
    cap.release()
    cv2.destroyAllWindows()
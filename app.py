import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms
import io

app = Flask(__name__)

data_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    # Set CC to 3 so the images retain the grayscale after transforming to tensor
    transforms.Grayscale(num_output_channels=3),
    # Turn the image into a torch.Tensor
    transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.googlenet(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load('model_google3.pth', map_location=torch.device('cpu')))
model.eval()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    img_up = request.files['photo_input']
    img_data = img_up.read()
    img_stream = io.BytesIO(img_data)
    img = Image.open(img_stream)
    img = data_transform(img)

    # Perform inference
    with torch.no_grad():
        output = model(img.unsqueeze(0))  # Add a batch dimension

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(output, dim=1).squeeze().tolist()

    # Determine predicted label and corresponding percentage
    print(probabilities[0], probabilities[1])
    predicted_label = 'girl' if probabilities[0] > probabilities[1] else 'boy'
    percentage = round(max(probabilities) * 100, 1)  # Get the highest probability
    if predicted_label == 'boy':
        return render_template('boy.html', predicted_label=predicted_label, percentage=percentage)
    else:
        return render_template('girl.html', predicted_label=predicted_label, percentage=percentage)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

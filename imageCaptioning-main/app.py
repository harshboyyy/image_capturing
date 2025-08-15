from flask import Flask, request, render_template
import os
from utils import extract_image_features, greedy_generator

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            features = extract_image_features(filepath)
            caption = greedy_generator(features)
            image = os.path.join('static/uploads', filename)
    return render_template('index.html', caption=caption, image=image)

if __name__ == '__main__':
    app.run(debug=True)

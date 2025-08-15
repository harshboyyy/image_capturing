# 🖼️ Image Captioning Web App using Flask

This is an image captioning web application built using **Flask** and a trained **deep learning model**. Upload an image, and the app generates a caption describing its content.

---

## 🚀 Features

- Uses **InceptionV3** for image feature extraction (CNN encoder)
- Uses **LSTM** for caption generation (decoder)
- Built with **TensorFlow/Keras**
- Web interface built using **Flask** and **HTML/CSS**
- Pretrained tokenizer and model weights are used locally

---

## 🧠 Model Overview

The model follows the encoder-decoder architecture:

- **Encoder**: Pretrained InceptionV3 extracts a 2048-dimensional feature vector from the input image.
- **Decoder**: A stacked LSTM model uses the image features and previously generated words to produce the next word in the caption.

---

## 🗂️ Project Structure

ImageCaptioningApp/
├── app.py # Flask backend
├── utils.py # Feature extraction and caption generation logic
├── tokenizer.pkl # Trained tokenizer used during training
├── caption_model.weights.h5 # Trained model weights (NOT included in GitHub)
├── static/
│ └── uploads/ # Folder where uploaded images are saved
├── templates/
│ └── index.html # Web UI template
├── .gitignore # To ignore virtual environment and weights
├── requirements.txt # Required Python libraries
└── README.md # Project documentation

---

## 🛠️ Installation

### 1. Clone the repository

```bash 
git clone https://github.com/SaranshGupta02/imageCaptioning
```

### 1.  Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
🖼️ How It Works
Upload an image via the web interface.

The image is passed through InceptionV3 to extract features.

These features are passed to an LSTM-based model to generate a caption.

The generated caption is displayed below the image.



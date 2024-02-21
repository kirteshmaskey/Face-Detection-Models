// require('@tensorflow/tfjs-node');
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const canvas = require('canvas');
const faceapi = require('face-api.js');

const app = express();
const PORT = process.env.PORT || 3000;

// faceapi setup and canvas setup
faceapi.env.monkeyPatch({
  Canvas: canvas.Canvas,
  Image: canvas.Image,
  ImageData: canvas.ImageData,
});

// Configure multer for file upload
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB file size limit
});

// Serve static files from the "public" folder
app.use(express.static(path.join(__dirname, 'public')));

// Set up face-api.js
async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk('models/');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('models/');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('models/');
}

// Load models on server start
loadModels();

// Render the HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Handle image upload and face detection
app.post('/detect', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send('No image file uploaded');
    }

    const imagePath = path.join(__dirname, req.file.path);
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);

    const detections = await faceapi.detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (detections.length === 0) {
      return res.send('No faces detected!');
    }

    // Calculate average confidence score for all detections
    let totalConfidence = 0;
    for (const detection of detections) {
      totalConfidence += detection.detection._score;
    }
    const averageConfidence = totalConfidence / detections.length;

    const result = `Number of faces detected: ${detections.length}<br>Model Accuracy: ${averageConfidence}`;

    // Delete uploaded image after processing
    fs.unlinkSync(imagePath);

    res.send(result);
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

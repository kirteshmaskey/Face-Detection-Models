require("@tensorflow/tfjs")
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const blazeface = require('@tensorflow-models/blazeface');

const app = express();
const PORT = process.env.PORT || 3001;

// Configure multer for file upload
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB file size limit
});

// Serve static files from the "public" folder
app.use(express.static(path.join(__dirname, 'public')));

// Load the Blazeface model
let model;
blazeface.load().then((loadedModel) => {
  model = loadedModel;
  console.log('Blazeface model loaded successfully');
}).catch((error) => {
  console.error('Error loading Blazeface model:', error);
});

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

    if (!model) {
      return res.status(500).send('Blazeface model not loaded');
    }

    const imagePath = path.join(__dirname, req.file.path);
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);

    const predictions = await model.estimateFaces(canvas);

    if (predictions.length === 0) {
      return res.send('No faces detected!');
    }

    // Calculate average confidence score for all detections
    let totalConfidence = 0;
    for (const prediction of predictions) {
      totalConfidence += prediction.probability;
    }
    const averageConfidence = totalConfidence / predictions.length;

    const result = `Number of faces detected: ${predictions.length}<br>Model Accuracy: ${averageConfidence}`;

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

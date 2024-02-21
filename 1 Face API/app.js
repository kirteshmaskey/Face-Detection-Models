const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
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
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "./uploads");
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname)); 
  },
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 1024 * 1024 * 10, // Limit file size to 1MB
  },
});

// Serve static files from the "public" folder
app.use(express.static(path.join(__dirname, 'public')));


const MODEL_URL = path.join(__dirname, 'models');
// Set up face-api.js
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL),
  faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL),
]).then(() => {
  console.log("loaded all the models from the disk");
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

    const filename = req.file.filename;

    // Load the uploaded image using face-api.js
    const imageBuffer = fs.readFileSync(`./uploads/${filename}`);
    const image = new canvas.Image();
    image.src = imageBuffer;

    // Detect faces in the uploaded image and compute face descriptors
    const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({
      minConfidence: 0.5,
    });

    const detectionResult  = await faceapi
      .detectAllFaces(image, faceDetectionOptions)
      .withFaceLandmarks();
      // .withFaceDescriptors();

      if (detectionResult.length === 0) {
        // No face detected, delete the uploaded image and send a message
        fs.unlinkSync(`./uploads/${filename}`); // Delete the image

        return res.send("No face detected in the uploaded image.");
      }

    const accuracy = detectionResult[0].detection.score;
    const response = `Number of faces detected: ${detectionResult.length}<br>Model Accuracy: ${accuracy}`;

    console.log(accuracy);
    // Delete uploaded image after processing
    fs.unlinkSync(`./uploads/${filename}`);

    res.send(response);
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

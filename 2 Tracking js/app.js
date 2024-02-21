const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const tracking = require('tracking');

const app = express();
const PORT = process.env.PORT || 3000;

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

    // Use Tracking.js for face detection
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);

    const tracker = new tracking.ObjectTracker(['face']);
    tracker.setInitialScale(4);
    tracker.setStepSize(2);
    tracker.setEdgesDensity(0.1);

    tracking.track(canvas, tracker);

    tracker.on('track', (event) => {
      const results = event.data;
      if (results.length === 0) {
        return res.send('No faces detected!');
      }

      const accuracy = results[0].confidence;
      const result = `Number of faces detected: ${results.length}<br>Model Accuracy: ${accuracy}`;

      // Delete uploaded image after processing
      fs.unlinkSync(imagePath);

      res.send(result);
    });
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
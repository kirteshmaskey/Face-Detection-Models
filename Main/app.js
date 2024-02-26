let tf
// require("@tensorflow/tfjs")

const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { createCanvas, loadImage } = require("canvas");
const canvas = require("canvas");
const faceapi = require("face-api.js");
const blazeface = require("@tensorflow-models/blazeface");
// const tracking = require('tracking');

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
app.use(express.static(path.join(__dirname, "public")));

const MODEL_URL = path.join(__dirname, "models");
// Set up face-api.js
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL),
  faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL),
  faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_URL),
  faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL),
]).then(() => {
  console.log("loaded all the face-api models from the disk");
});

// Load the Blazeface model
let model;
blazeface
  .load()
  .then((loadedModel) => {
    model = loadedModel;
    console.log("Blazeface model loaded successfully");
  })
  .catch((error) => {
    console.error("Error loading Blazeface model:", error);
  });

// Render the HTML file
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

const FaceAPIModel = async (req) => {
  try {
    const filename = req.file.filename;
  
    // Load the uploaded image using face-api.js
    const imageBuffer = fs.readFileSync(`./uploads/${filename}`);
    const image = new canvas.Image();
    image.src = imageBuffer;
  
    // Detect faces in the uploaded image and compute face descriptors
    const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({
      minConfidence: 0.5,
    });
  
    const detectionResult = await faceapi
      .detectAllFaces(image, faceDetectionOptions)
      .withFaceLandmarks();
    // .withFaceDescriptors();
  
    if (detectionResult.length === 0) {
      // No face detected
      const response = {
        model: "Face API",
        result: "No face detected",
      };
      return response;
    }
  
    const accuracy = detectionResult[0].detection.score;
    const response = {
      model: "Face API",
      numberOfFaces: detectionResult.length,
      modelAccuracy: accuracy,
    };
  
    return response;
  }catch(err) {
    const response = {
      model: "Face API",
      error: err.message,
    };
    return response;
  }
};

const tinyFaceDetectorModel = async (req) => {
  try {
    const imagePath = path.join(__dirname, req.file.path);
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
  
    const detectionResult = await faceapi
      .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();
  
    if (detectionResult.length === 0) {
      const response = {
        model: "Tiny Face Detector",
        result: "No face detected",
      };
      return response;
    }
  
    // Calculate average confidence score for all detections
    let totalConfidence = 0;
    for (const detection of detectionResult) {
      totalConfidence += detection.detection._score;
    }
    const averageConfidence = totalConfidence / detectionResult.length;
  
    const response = {
      model: "Tiny Face Detector",
      numberOfFaces: detectionResult.length,
      modelAccuracy: averageConfidence,
    };
  
    return response;
  }catch(err) {
    const response = {
      model: "Tiny Face Detector",
      error: err.message,
    };
    return response;
  }
};

const blazeFaceModel = async (req) => {
  if (!model) {
    const response = {
      model: "BlazeFace",
      error: "Blazeface model not loaded",
    };
    return response;
  }
  try {
    const imagePath = path.join(__dirname, req.file.path);
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);

    const detectionResult = await model.estimateFaces(canvas);

    if (detectionResult.length === 0) {
      const response = {
        model: "Blaze Face",
        result: "No face detected",
      };
      return response;
    }

    // Calculate average confidence score for all detections
    let totalConfidence = 0;
    for (const prediction of detectionResult) {
      totalConfidence += prediction.probability;
    }
    const averageConfidence = totalConfidence / detectionResult.length;

    const response = {
      model: "Face API",
      numberOfFaces: detectionResult.length,
      modelAccuracy: averageConfidence,
    };

    return response;
  } catch (err) {
    const response = {
      model: "Blaze Face",
      error: err.message,
    };
    return response;
  }
  
};

const trackingJsModel = async (req) => {
  try {
    const imagePath = path.join(__dirname, req.file.path);

    // Use Tracking.js for face detection
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);

    const tracker = new tracking.ObjectTracker(["face"]);
    tracker.setInitialScale(4);
    tracker.setStepSize(2);
    tracker.setEdgesDensity(0.1);

    tracking.track(canvas, tracker);

    tracker.on("track", (event) => {
      const detectionResult = event.data;
      if (detectionResult.length === 0) {
        // No face detected
        const response = {
          model: "Traking js",
          result: "No face detected",
        };
        return response;
      }

      const accuracy = detectionResult[0].confidence;

      const response = {
        model: "Traking js",
        numberOfFaces: detectionResult.length,
        modelAccuracy: accuracy,
      };

      return response;
    });
  } catch (error) {
    console.error(error);
    res.status(500).send("Internal Server Error");
  }
};

// Handle image upload and face detection for all models
app.post("/detect", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send("No image file uploaded");
    }

    const faceAPIRes = await FaceAPIModel(req);
    const tinyFaceDetectorAcc = await tinyFaceDetectorModel(req);
    const blazeFaceAcc = await blazeFaceModel(req);

    // const trakingJsRes = trackingJsModel(req);

    const result = {
      faceAPI: faceAPIRes,
      tinyFaceDetector: tinyFaceDetectorAcc,
      blazeFace: blazeFaceAcc,

      // trackingJs: trakingJsRes
    };

    // Delete uploaded image after processing
    fs.unlinkSync(`./uploads/${req.file.filename}`);

    console.log(result);
    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).send("Internal Server Error");
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

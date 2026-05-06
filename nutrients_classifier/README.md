# 🌿 NutriVision AI

NutriVision AI is a full-stack, AI-powered dietary tracking platform that combines computer vision, global databases, and conversational LLMs to seamlessly log and analyze your nutritional intake.

## 🚀 Features

*   **Computer Vision Pipeline:** Snap a photo of your meal. A custom-trained Convolutional Neural Network (CNN) paired with YOLOv8n object detection instantly identifies your food and calculates portions.
*   **Barcode Scanner:** A native, mobile-friendly barcode scanner built with `react-webcam` and `html5-qrcode`. It pings the OpenFoodFacts global API to log any packaged food item.
*   **Live USDA Database:** A lightning-fast Pandas backend searching a 7,000+ item USDA fallback database in real-time.
*   **Conversational AI Nutritionist:** A floating React widget powered by **Google Gemini 2.5 Flash**. It reads your active SQLite meal logs and provides personalized, context-aware dietary advice.
*   **Analytics Dashboard:** Beautiful, animated data visualization using `recharts` to track your 7-day rolling macro distributions and caloric intake.
*   **Secure Authentication:** Multi-user support with hashed passwords and protected routing.

## 🛠️ Technology Stack

*   **Frontend:** React, Vite, React Router, Recharts, Lucide Icons, Vanilla CSS (Glassmorphism UI).
*   **Backend:** Python, FastAPI, SQLAlchemy, SQLite, Pandas.
*   **AI/ML:** TensorFlow, OpenCV, Ultralytics YOLOv8, Google Generative AI (Gemini).
*   **DevOps:** Docker, Docker Compose, Nginx.

## ⚙️ Quick Start (Docker)

1. Clone the repository.
2. Insert your Gemini API Key in `api.py`.
3. Build and launch the containerized application:
```bash
docker-compose up --build
```
4. Access the frontend at `http://localhost:80` and the API at `http://localhost:8000`.

## 📸 Architecture
*Fully decoupled architecture with RESTful API communication and stateless frontend.*

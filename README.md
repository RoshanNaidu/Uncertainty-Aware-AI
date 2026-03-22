# AI That Knows When It's Wrong

A complete, publication-ready Deep Ensemble pipeline for uncertainty quantification using CIFAR-10, paired with a modern Next.js web application.

[Full Walkthrough Documentation](brain/a8720bf5-0565-40cc-9875-b5adeac35b2d/walkthrough.md)

This project trains $M=5$ WideResNet-28-10 models entirely from scratch to create a Deep Ensemble. The backend framework evaluates the ensemble's mean prediction probability alongside measures of Total Uncertainty (Predictive Entropy) and Epistemic Uncertainty (Mutual Information — isolating model disagreement). 

Out-of-Distribution (OOD) testing takes place using CIFAR-100 to show the model generating appropriately high uncertainty when evaluating classes it hasn't seen before.

## Architecture Structure

- `/backend`: PyTorch WideResNet architecture, training pipelines, evaluation suites, and FastAPI endpoints.
- `/frontend`: Next.js App Router providing a premium glassmorphic UI, file-drop zones, Recharts histograms showing ensemble variance, and dynamic Confidence Progress markers.

## 🚀 Quick Setup

### 1. The Backend (PyTorch + FastAPI)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Train the Ensemble (Takes substantial time)
```bash
# Full training of 5 models on CIFAR-10
python train.py --epochs 200 --ensemble-size 5

# Sanity check training (1 epoch)
python train.py --epochs 1 --ensemble-size 1
```

*(Note: Without checkpoints, the API serves semi-random predictions for front-end structure testing).*

#### Start the Inference API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. The Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000).

---

## ☁️ Deployment Instructions

### Backend (Render or Railway)
- Use the included `requirements.txt`.
- Set the start command to: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- *Note:* Each WideResNet-28-10 comprises ~36.5M parameters. Ensure the deploy tier provides roughly 4GB of RAM to comfortably load all 5 models concurrently.

### Frontend (Vercel)
- Push repository to GitHub and link to Vercel.
- Framework Preset: Next.js
- Environment Variable: Set `NEXT_PUBLIC_API_URL` to your live backend endpoint.

## Technologies
- **Backend:** PyTorch, Torchvision, FastAPI, Uvicorn, Scikit-Learn. 
- **Frontend:** Next.js (App Router), React, Tailwind CSS v4, Recharts, Framer Motion. 

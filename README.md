# AI That Knows When It's Wrong

A complete, publication-ready Deep Ensemble pipeline for uncertainty quantification using CIFAR-10, paired with a modern Next.js web application.

This project trains $M=3$ WideResNet-28-10 models entirely from scratch to create a Deep Ensemble. The backend framework evaluates the ensemble's mean prediction probability alongside measures of Total Uncertainty (Predictive Entropy) and Epistemic Uncertainty (Mutual Information — isolating model disagreement). 

Out-of-Distribution (OOD) testing takes place using CIFAR-100 to show the model generating appropriately high uncertainty when evaluating classes it hasn't seen before.

## Architecture Structure

- `/backend`: PyTorch WideResNet architecture, training pipelines, evaluation suites, and FastAPI endpoints.
- `/frontend`: Next.js App Router providing a premium glassmorphic UI, file-drop zones, Recharts histograms showing ensemble variance, and dynamic Confidence Progress markers.

### 1. The Backend (PyTorch + FastAPI)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Train the Ensemble (Takes substantial time)
```bash
# Full training of 3 models on CIFAR-10
python train.py --epochs 200 --ensemble-size 3

# Sanity check training (1 epoch)
python train.py --epochs 1 --ensemble-size 1
```

*(Note: Without checkpoints, the API serves semi-random predictions for front-end structure testing).*

#### Start the Inference API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

"use client";

import { useState, useCallback } from "react";
import ImageUpload from "./components/ImageUpload";
import PredictionDisplay from "./components/PredictionDisplay";
import TopProbabilities from "./components/TopProbabilities";
import EnsembleChart from "./components/EnsembleChart";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "/api";

export default function Home() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = useCallback(async (file) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to get prediction");
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <div className="gradient-bg min-h-screen">
      {/* Header */}
      <header className="border-b border-white/[0.06] bg-black/20 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-lg">
              🧠
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                AI That Knows When It&apos;s Wrong
              </h1>
              <p className="text-[11px] text-white/30">
                Deep Ensemble Uncertainty Quantification · WideResNet-28-10 ·
                CIFAR-10
              </p>
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-2 text-xs text-white/30">
            <span className="px-2 py-1 rounded-md bg-white/[0.04] border border-white/[0.06]">
              M = 5 models
            </span>
            <span className="px-2 py-1 rounded-md bg-white/[0.04] border border-white/[0.06]">
              36.5M params each
            </span>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Hero text */}
        <div className="text-center mb-8">
          <h2 className="text-2xl sm:text-3xl font-bold mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Uncertainty-Aware Predictions
          </h2>
          <p className="text-white/40 max-w-lg mx-auto text-sm">
            Upload any image to see how a deep ensemble of 5 independently
            trained neural networks quantifies its own uncertainty.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left column — Input */}
          <div className="lg:col-span-4 space-y-6">
            <ImageUpload onUpload={handleUpload} isLoading={isLoading} />

            {/* How it works */}
            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <span>💡</span> How It Works
              </h3>
              <div className="space-y-2.5 text-xs text-white/40">
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-mono">1.</span>
                  <span>
                    Your image is resized to 32×32 and fed through{" "}
                    <strong className="text-white/60">5 independent</strong>{" "}
                    WideResNet-28-10 models
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-mono">2.</span>
                  <span>
                    Each model produces its own softmax probability distribution
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-mono">3.</span>
                  <span>
                    <strong className="text-white/60">
                      Predictive Entropy
                    </strong>{" "}
                    measures total uncertainty;{" "}
                    <strong className="text-white/60">
                      Mutual Information
                    </strong>{" "}
                    isolates model disagreement (epistemic uncertainty)
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-mono">4.</span>
                  <span>
                    Out-of-distribution images should trigger{" "}
                    <strong className="text-white/60">higher uncertainty</strong>
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right column — Results */}
          <div className="lg:col-span-8 space-y-6">
            {/* Loading state */}
            {isLoading && (
              <div className="glass-card p-12 flex flex-col items-center gap-4">
                <div className="spinner"></div>
                <p className="text-sm text-white/40 animate-pulse-soft">
                  Running ensemble inference...
                </p>
              </div>
            )}

            {/* Error state */}
            {error && (
              <div className="glass-card p-6 border border-red-500/20 bg-red-500/5 animate-fade-in">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">❌</span>
                  <div>
                    <p className="text-sm font-medium text-red-400">
                      Prediction Failed
                    </p>
                    <p className="text-xs text-white/40 mt-1">{error}</p>
                  </div>
                </div>
                <p className="text-xs text-white/30 mt-3">
                  Make sure the backend is running:{" "}
                  <code className="text-indigo-400">
                    uvicorn main:app --port 8000
                  </code>
                </p>
              </div>
            )}

            {/* Results */}
            {result && !isLoading && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <PredictionDisplay result={result} />
                  <TopProbabilities top5={result.top5} />
                </div>
                <EnsembleChart
                  ensemblePredictions={result.ensemble_predictions}
                  predictedIndex={result.predicted_index}
                />
              </>
            )}

            {/* Empty state */}
            {!result && !isLoading && !error && (
              <div className="glass-card p-16 flex flex-col items-center gap-4 text-center">
                <div className="text-5xl opacity-20">🔬</div>
                <p className="text-white/30 text-sm">
                  Upload an image to see uncertainty-aware predictions
                </p>
                <p className="text-white/20 text-xs max-w-md">
                  Try uploading CIFAR-10 classes (airplane, car, bird, cat,
                  deer, dog, frog, horse, ship, truck) for confident predictions,
                  or non-CIFAR images for high uncertainty.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-white/20">
          <p>
            Deep Ensemble UQ • WideResNet-28-10 • CIFAR-10 In-Distribution •
            CIFAR-100 OOD
          </p>
          <div className="flex items-center gap-4">
            <span>PyTorch + FastAPI + Next.js</span>
            <span>•</span>
            <span>M = 5 ensemble members</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

"use client";

import { useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import ImageUpload from "./components/ImageUpload";
import PredictionDisplay from "./components/PredictionDisplay";
import TopProbabilities from "./components/TopProbabilities";
import EnsembleChart from "./components/EnsembleChart";
import GradCAMDisplay from "./components/GradCAMDisplay";
import ValidationInsight from "./components/ValidationInsight";

const ParticleBackground = dynamic(
  () => import("./components/ParticleBackground"),
  { ssr: false }
);

const API_URL = process.env.NEXT_PUBLIC_API_URL || "/api";

export default function Home() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gradcamResult, setGradcamResult] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const lastFileRef = useRef(null);

  const fetchGradCAM = useCallback(async (file, targetIndex) => {
    setGradcamLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target_class", targetIndex.toString());

      const response = await fetch(`${API_URL}/gradcam`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setGradcamResult(data);
      } else {
        console.warn("Grad-CAM request failed:", response.status);
        setGradcamResult(null);
      }
    } catch (err) {
      console.warn("Grad-CAM request error:", err);
      setGradcamResult(null);
    } finally {
      setGradcamLoading(false);
    }
  }, []);

  const handleUpload = useCallback(async (file) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setGradcamResult(null);
    lastFileRef.current = file;

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

      // Trigger Grad-CAM after prediction succeeds
      fetchGradCAM(file, data.predicted_index);
    } catch (err) {
      setError(err.message || "Failed to get prediction");
    } finally {
      setIsLoading(false);
    }
  }, [fetchGradCAM]);

  return (
    <div className="app-shell min-h-screen">
      <ParticleBackground />

      {/* Header */}
      <header className="header-bar border-b border-white/[0.06] bg-black/30 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
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
              M = 3 models
            </span>
            <span className="px-2 py-1 rounded-md bg-white/[0.04] border border-white/[0.06]">
              36.5M params each
            </span>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="content-layer max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Hero text */}
        <div className="text-center mb-8">
          <h2 className="text-2xl sm:text-3xl font-bold mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Uncertainty-Aware Predictions
          </h2>
          <p className="text-white/40 max-w-lg mx-auto text-sm">
            Upload any image to see how a deep ensemble of 3 independently
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
                <div className="w-5 h-5 rounded bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                  <svg className="w-3 h-3 text-amber-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zm4.657 2.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM4 11a1 1 0 100-2H3a1 1 0 000 2h1zm7 5a1 1 0 01-1 1v1a1 1 0 102 0v-1a1 1 0 01-1-1zm-4.95-1.536a1 1 0 00-1.414 1.414l.707.707a1 1 0 001.414-1.414l-.707-.707zm9.9 0l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zM10 6a4 4 0 100 8 4 4 0 000-8z" />
                  </svg>
                </div>
                How It Works
              </h3>
              <div className="space-y-2.5 text-xs text-white/40">
                <div className="flex gap-2">
                  <span className="text-indigo-400 font-mono">1.</span>
                  <span>
                    Your image is resized to 32×32 and fed through{" "}
                    <strong className="text-white/60">3 independent</strong>{" "}
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
                  <div className="w-8 h-8 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </div>
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
                <GradCAMDisplay
                  gradcamResult={gradcamResult}
                  isLoading={gradcamLoading}
                />
                <EnsembleChart
                  ensemblePredictions={result.ensemble_predictions}
                  predictedIndex={result.predicted_index}
                />
                <ValidationInsight result={result} />
              </>
            )}

            {/* Empty state */}
            {!result && !isLoading && !error && (
              <div className="glass-card p-16 flex flex-col items-center gap-4 text-center">
                <div className="w-12 h-12 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center">
                  <svg className="w-6 h-6 text-white/20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                  </svg>
                </div>
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
      <footer className="content-layer border-t border-white/[0.06] mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-white/20">
          <p>
            Deep Ensemble UQ · WideResNet-28-10 · CIFAR-10 In-Distribution ·
            CIFAR-100 OOD
          </p>
          <div className="flex items-center gap-4">
            <span>PyTorch + FastAPI + Next.js</span>
            <span>·</span>
            <span>M = 3 ensemble members</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

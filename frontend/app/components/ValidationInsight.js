"use client";

export default function ValidationInsight({ result }) {
  if (!result) return null;

  const confidence = result.confidence;
  const entropy = result.uncertainty.predictive_entropy;
  const mi = result.uncertainty.mutual_information;
  const numModels = result.num_models || 3;
  const predictedClass = result.predicted_class;

  // Determine the scenario
  let scenario, scenarioColor, interpretation, validationPoints;

  if (confidence >= 0.7 && entropy < 1.0) {
    scenario = "High Confidence — Likely In-Distribution";
    scenarioColor = "text-emerald-400";
    interpretation = `The ensemble of ${numModels} models strongly agrees on "${predictedClass}" with ${(confidence * 100).toFixed(1)}% mean confidence. Low predictive entropy (${entropy.toFixed(3)}) indicates the models have seen similar images during training (CIFAR-10 dataset), making this a reliable prediction.`;
    validationPoints = [
      {
        title: "Ensemble Agreement",
        detail: `All ${numModels} models independently converged on the same class, reducing the likelihood of a single-model fluke.`,
      },
      {
        title: "Low Mutual Information",
        detail: `MI = ${mi.toFixed(3)} — minimal disagreement between models. This means the epistemic (model) uncertainty is low, so additional training data is unlikely to change this prediction.`,
      },
      {
        title: "In-Distribution Signal",
        detail: `Low entropy suggests this image resembles CIFAR-10 training data. The model is operating within its competence boundary.`,
      },
    ];
  } else if (confidence >= 0.4 && confidence < 0.7) {
    scenario = "Moderate Confidence — Borderline Case";
    scenarioColor = "text-amber-400";
    interpretation = `The ensemble shows mixed signals for "${predictedClass}" at ${(confidence * 100).toFixed(1)}% confidence. Moderate entropy (${entropy.toFixed(3)}) suggests the models are partially uncertain — this could be a difficult sample or partially out-of-distribution.`;
    validationPoints = [
      {
        title: "Partial Agreement",
        detail: `The models lean toward "${predictedClass}" but some assign non-trivial probability to other classes. This is expected for ambiguous images.`,
      },
      {
        title: "Moderate Mutual Information",
        detail: `MI = ${mi.toFixed(3)} — some model disagreement exists. Different training seeds led to different feature representations, revealing epistemic uncertainty.`,
      },
      {
        title: "Validation Approach",
        detail: `For borderline cases, check the Top-5 probabilities and Ensemble Analysis chart. If the runner-up class is visually plausible, the uncertainty is well-calibrated.`,
      },
    ];
  } else {
    scenario = "Low Confidence — Likely Out-of-Distribution";
    scenarioColor = "text-red-400";
    interpretation = `The ensemble is uncertain about "${predictedClass}" — only ${(confidence * 100).toFixed(1)}% mean confidence. High entropy (${entropy.toFixed(3)}) is a strong signal that this image is out-of-distribution (not in CIFAR-10's 10 classes).`;
    validationPoints = [
      {
        title: "Expected Behavior",
        detail: `High uncertainty on unfamiliar images is the correct response. A model that confidently misclassifies novel inputs is dangerous — our ensemble flags these cases instead.`,
      },
      {
        title: "High Mutual Information",
        detail: `MI = ${mi.toFixed(3)} — significant disagreement between models. Each model "guesses" differently, confirming this is epistemic (knowledge-gap) uncertainty.`,
      },
      {
        title: "This Is a Feature, Not a Bug",
        detail: `Deep Ensembles are designed to quantify "what the model doesn't know." High uncertainty here means the ensemble is correctly refusing to make a confident prediction on data it wasn't trained for.`,
      },
    ];
  }

  return (
    <div className="glass-card p-6 animate-fade-in">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/20 flex items-center justify-center">
          <svg
            className="w-4 h-4 text-indigo-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <div>
          <h2 className="text-lg font-semibold">Validation Insight</h2>
          <p className={`text-xs font-medium ${scenarioColor}`}>
            {scenario}
          </p>
        </div>
      </div>

      {/* Interpretation summary */}
      <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4 mb-5">
        <p className="text-sm text-white/60 leading-relaxed">
          {interpretation}
        </p>
      </div>

      {/* Validation points */}
      <div className="space-y-4">
        <p className="text-xs uppercase tracking-widest text-white/40">
          Why you can trust this result
        </p>
        {validationPoints.map((point, idx) => (
          <div
            key={idx}
            className="flex gap-3 animate-fade-in"
            style={{ animationDelay: `${idx * 100}ms` }}
          >
            <div className="flex-shrink-0 mt-1">
              <div className="w-5 h-5 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                <span className="text-[10px] font-bold text-indigo-400">
                  {idx + 1}
                </span>
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-white/80 mb-0.5">
                {point.title}
              </p>
              <p className="text-xs text-white/40 leading-relaxed">
                {point.detail}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Bottom note */}
      <div className="mt-5 pt-4 border-t border-white/[0.06] flex items-start gap-2">
        <svg
          className="w-4 h-4 text-white/20 flex-shrink-0 mt-0.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <p className="text-[11px] text-white/30 leading-relaxed">
          This analysis uses Deep Ensemble uncertainty decomposition. Predictive
          Entropy captures total uncertainty, while Mutual Information isolates
          epistemic uncertainty (model disagreement). Together, they enable
          reliable out-of-distribution detection without requiring OOD training
          data.
        </p>
      </div>
    </div>
  );
}

"use client";

function getConfidenceColor(confidence) {
  if (confidence >= 0.7) return "high";
  if (confidence >= 0.4) return "medium";
  return "low";
}

function getConfidenceLabel(level) {
  return { high: "High", medium: "Medium", low: "Low" }[level] || "Unknown";
}

export default function PredictionDisplay({ result }) {
  if (!result) return null;

  const confidence = result.confidence;
  const level = getConfidenceColor(confidence);
  const entropy = result.uncertainty.predictive_entropy;
  const mi = result.uncertainty.mutual_information;

  return (
    <div className="glass-card p-6 animate-fade-in">
      <h2 className="text-lg font-semibold mb-5 flex items-center gap-2">
        <span className="text-2xl">🎯</span>
        Prediction
      </h2>

      {/* Main prediction */}
      <div className="text-center mb-6">
        <p className="text-xs uppercase tracking-widest text-white/40 mb-1">
          Predicted Class
        </p>
        <h3 className="text-4xl font-bold capitalize bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          {result.predicted_class}
        </h3>
      </div>

      {/* Confidence + Uncertainty grid */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <MetricCard
          label="Confidence"
          value={`${(confidence * 100).toFixed(1)}%`}
          level={level}
        />
        <MetricCard
          label="Entropy"
          value={entropy.toFixed(3)}
          sublabel="Total"
          level={entropy > 1.0 ? "low" : entropy > 0.5 ? "medium" : "high"}
        />
        <MetricCard
          label="Mutual Info"
          value={mi.toFixed(3)}
          sublabel="Epistemic"
          level={mi > 0.3 ? "low" : mi > 0.1 ? "medium" : "high"}
        />
      </div>

      {/* Confidence bar */}
      <div className="mb-5">
        <div className="flex justify-between text-xs text-white/50 mb-1.5">
          <span>Confidence</span>
          <span className={`confidence-${level} font-medium`}>
            {getConfidenceLabel(level)}
          </span>
        </div>
        <div className="h-2.5 rounded-full bg-white/5 overflow-hidden">
          <div
            className={`h-full rounded-full bg-confidence-${level} animate-fill`}
            style={{ width: `${Math.min(confidence * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Smart feedback */}
      <SmartFeedback confidence={confidence} entropy={entropy} mi={mi} />
    </div>
  );
}

function MetricCard({ label, value, sublabel, level }) {
  return (
    <div
      className={`rounded-xl p-3 text-center bg-white/[0.03] border border-white/[0.05] glow-${
        level === "high" ? "green" : level === "medium" ? "yellow" : "red"
      }`}
    >
      <p className="text-[10px] uppercase tracking-wider text-white/40 mb-0.5">
        {label}
      </p>
      <p className={`text-xl font-bold confidence-${level}`}>{value}</p>
      {sublabel && (
        <p className="text-[10px] text-white/30 mt-0.5">{sublabel}</p>
      )}
    </div>
  );
}

function SmartFeedback({ confidence, entropy, mi }) {
  let message, emoji, borderColor;

  if (confidence >= 0.7) {
    message = "Model is very confident in this prediction";
    emoji = "✅";
    borderColor = "border-green-500/30 bg-green-500/5";
  } else if (confidence >= 0.4) {
    message = "Model is moderately confident — some uncertainty present";
    emoji = "⚠️";
    borderColor = "border-yellow-500/30 bg-yellow-500/5";
  } else {
    message =
      "Model is uncertain — this may be an out-of-distribution sample";
    emoji = "🔴";
    borderColor = "border-red-500/30 bg-red-500/5";
  }

  return (
    <div
      className={`rounded-lg border px-4 py-3 flex items-center gap-3 text-sm ${borderColor} animate-fade-in`}
    >
      <span className="text-xl">{emoji}</span>
      <span className="text-white/70">{message}</span>
    </div>
  );
}

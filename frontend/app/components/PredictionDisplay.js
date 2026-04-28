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
        <div className="w-6 h-6 rounded bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        </div>
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
    emoji = "check";
    borderColor = "border-green-500/30 bg-green-500/5";
  } else if (confidence >= 0.4) {
    message = "Model is moderately confident — some uncertainty present";
    emoji = "warn";
    borderColor = "border-yellow-500/30 bg-yellow-500/5";
  } else {
    message =
      "Model is uncertain — this may be an out-of-distribution sample";
    emoji = "alert";
    borderColor = "border-red-500/30 bg-red-500/5";
  }

  return (
    <div
      className={`rounded-lg border px-4 py-3 flex items-center gap-3 text-sm ${borderColor} animate-fade-in`}
    >
      <span className="text-xl flex-shrink-0">
        {emoji === "check" ? (
          <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        ) : emoji === "warn" ? (
          <svg className="w-5 h-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        ) : (
          <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )}
      </span>
      <span className="text-white/70">{message}</span>
    </div>
  );
}

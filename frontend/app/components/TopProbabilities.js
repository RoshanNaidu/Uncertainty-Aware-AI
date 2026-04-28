"use client";

const CIFAR10_CLASSES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
];

function getBarColor(probability) {
  if (probability >= 0.5) return "bg-indigo-500";
  if (probability >= 0.2) return "bg-indigo-400/70";
  return "bg-indigo-400/40";
}

export default function TopProbabilities({ top5 }) {
  if (!top5 || top5.length === 0) return null;

  const maxProb = top5[0]?.probability || 1;

  return (
    <div className="glass-card p-6 animate-fade-in">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <div className="w-6 h-6 rounded bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16 8v8m-4-5v5m-4-2v2m12-15a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h12z" />
          </svg>
        </div>
        Top-5 Probabilities
      </h2>

      <div className="space-y-3 stagger">
        {top5.map((item, idx) => (
          <div key={item.class_index} className="animate-fade-in">
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-sm font-medium capitalize flex items-center gap-2">
                <span className="text-xs text-white/30 font-mono w-4">
                  {idx + 1}.
                </span>
                {item.class_name}
              </span>
              <span className="text-xs font-mono text-white/50">
                {(item.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
              <div
                className={`h-full rounded-full ${getBarColor(
                  item.probability
                )} animate-fill`}
                style={{
                  width: `${Math.max(
                    (item.probability / maxProb) * 100,
                    2
                  )}%`,
                  animationDelay: `${idx * 80}ms`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

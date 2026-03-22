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
        <span className="text-2xl">📊</span>
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

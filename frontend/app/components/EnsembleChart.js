"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

const CIFAR10_CLASSES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
];

const COLORS = [
  "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#818cf8",
];

export default function EnsembleChart({ ensemblePredictions, predictedIndex }) {
  if (!ensemblePredictions || ensemblePredictions.length === 0) return null;

  const numModels = ensemblePredictions.length;

  // Build data: for the predicted class, show each model's probability
  const modelData = ensemblePredictions.map((modelProbs, idx) => ({
    name: `M${idx + 1}`,
    probability: modelProbs[predictedIndex],
  }));

  const meanProb =
    modelData.reduce((sum, d) => sum + d.probability, 0) / numModels;

  // Build class distribution data: average across models for all classes
  const classData = CIFAR10_CLASSES.map((cls, idx) => {
    const avg =
      ensemblePredictions.reduce((sum, m) => sum + m[idx], 0) / numModels;
    return {
      name: cls.charAt(0).toUpperCase() + cls.slice(1),
      probability: avg,
      isTop: idx === predictedIndex,
    };
  }).sort((a, b) => b.probability - a.probability);

  return (
    <div className="glass-card p-6 animate-fade-in">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <div className="w-6 h-6 rounded bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        </div>
        Ensemble Analysis
      </h2>

      {/* Per-model prediction for top class */}
      <div className="mb-6">
        <p className="text-xs uppercase tracking-widest text-white/40 mb-3">
          Per-Model Confidence — <span className="capitalize text-indigo-400">{CIFAR10_CLASSES[predictedIndex]}</span>
        </p>
        <div className="h-[180px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={modelData} barCategoryGap="20%">
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.05)"
                vertical={false}
              />
              <XAxis
                dataKey="name"
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(20,20,35,0.95)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  fontSize: 12,
                }}
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Prob"]}
              />
              <ReferenceLine
                y={meanProb}
                stroke="#a855f7"
                strokeDasharray="5 5"
                strokeWidth={2}
                label={{
                  value: `Mean: ${(meanProb * 100).toFixed(1)}%`,
                  fill: "#a855f7",
                  fontSize: 11,
                  position: "right",
                }}
              />
              <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                {modelData.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Class distribution */}
      <div>
        <p className="text-xs uppercase tracking-widest text-white/40 mb-3">
          Mean Prediction Distribution (All Classes)
        </p>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={classData} barCategoryGap="15%">
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.05)"
                vertical={false}
              />
              <XAxis
                dataKey="name"
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                tickLine={false}
                angle={-30}
                textAnchor="end"
                height={50}
              />
              <YAxis
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(20,20,35,0.95)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  fontSize: 12,
                }}
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Avg Prob"]}
              />
              <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                {classData.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={entry.isTop ? "#6366f1" : "rgba(99,102,241,0.25)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model agreement indicator */}
      <div className="mt-4 flex items-center gap-2 text-xs text-white/40">
        <div className="w-3 h-3 rounded-full bg-purple-500"></div>
        <span>Purple dashed line = ensemble mean</span>
        <span className="mx-2">•</span>
        <span>{numModels} models in ensemble</span>
      </div>
    </div>
  );
}

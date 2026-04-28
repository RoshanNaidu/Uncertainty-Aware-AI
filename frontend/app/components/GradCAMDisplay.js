"use client";

export default function GradCAMDisplay({ gradcamResult, isLoading }) {
  if (isLoading) {
    return (
      <div className="glass-card p-6 animate-fade-in">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="text-2xl">🔥</span>
          Grad-CAM Visualization
        </h2>
        <div className="flex flex-col items-center gap-4 py-8">
          <div className="gradcam-shimmer rounded-xl w-full h-[260px]"></div>
          <p className="text-sm text-white/40 animate-pulse-soft">
            Computing Grad-CAM heatmap...
          </p>
        </div>
      </div>
    );
  }

  if (!gradcamResult) return null;

  return (
    <div className="glass-card p-6 animate-fade-in">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <span className="text-2xl">🔥</span>
        Grad-CAM Visualization
      </h2>

      <div className="flex flex-col md:flex-row gap-6 items-center">
        {/* Heatmap image */}
        <div className="relative group">
          <div className="rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-indigo-500/10 gradcam-glow">
            <img
              src={`data:image/png;base64,${gradcamResult.gradcam_image}`}
              alt={`Grad-CAM heatmap for ${gradcamResult.target_class}`}
              className="w-[260px] h-[260px] object-cover"
              id="gradcam-heatmap-image"
            />
          </div>
          {/* Class badge */}
          <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 text-xs font-semibold whitespace-nowrap shadow-lg">
            Target: <span className="capitalize">{gradcamResult.target_class}</span>
          </div>
        </div>

        {/* Explanation */}
        <div className="flex-1 space-y-4">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-white/80">
              What is Grad-CAM?
            </h3>
            <p className="text-xs text-white/40 leading-relaxed">
              <strong className="text-white/60">Gradient-weighted Class Activation Mapping</strong>{" "}
              highlights the image regions most important for the model&apos;s prediction.
              Warmer colours (red/yellow) indicate high-relevance areas; cooler colours
              (blue/green) indicate low relevance.
            </p>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm" style={{ background: "linear-gradient(90deg, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)" }}></div>
              <span className="text-[10px] text-white/30 uppercase tracking-wider">
                Low → High importance
              </span>
            </div>
          </div>

          {/* Info pill */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06] text-xs text-white/40">
            <span>🧠</span>
            <span>
              Generated from <strong className="text-white/60">Model 1</strong> (ensemble member)
              using the final convolutional layer
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

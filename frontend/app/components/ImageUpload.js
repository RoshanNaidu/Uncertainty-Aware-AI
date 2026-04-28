"use client";

import { useRef, useState, useCallback } from "react";

export default function ImageUpload({ onUpload, isLoading }) {
  const fileInputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState("");

  const handleFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) return;
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
      onUpload(file);
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  return (
    <div className="glass-card p-6 glow-blue">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <div className="w-6 h-6 rounded bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        Upload Image
      </h2>

      <div
        className={`dropzone rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all min-h-[220px] ${
          dragOver ? "drag-over" : ""
        }`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {preview ? (
          <div className="flex flex-col items-center gap-4 animate-fade-in">
            <div className="relative w-40 h-40 rounded-lg overflow-hidden border border-white/10">
              <img
                src={preview}
                alt="Uploaded preview"
                className="w-full h-full object-cover"
              />
            </div>
            <p className="text-sm text-white/50 truncate max-w-[200px]">
              {fileName}
            </p>
            {isLoading && (
              <div className="flex items-center gap-2 text-indigo-400 text-sm">
                <div className="spinner !w-4 !h-4 !border-2"></div>
                Analyzing...
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-white/40">
            <svg
              className="w-12 h-12"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-sm font-medium">
              Drop an image here or{" "}
              <span className="text-indigo-400">click to browse</span>
            </p>
            <p className="text-xs text-white/30">
              JPEG, PNG — will be resized to 32×32 for CIFAR-10
            </p>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
      />

      {preview && !isLoading && (
        <button
          className="mt-4 w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 transition-colors text-sm font-medium"
          onClick={() => fileInputRef.current?.click()}
        >
          Upload Different Image
        </button>
      )}
    </div>
  );
}

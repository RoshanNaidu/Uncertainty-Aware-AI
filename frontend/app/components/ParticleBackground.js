"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function ParticleBackground() {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 300;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0a0a12, 1);
    containerRef.current.appendChild(renderer.domElement);

    // Particle system
    const PARTICLE_COUNT = 90;
    const SPREAD = 400;
    const CONNECTION_DISTANCE = 100;

    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    const palette = [
      new THREE.Color(0x6366f1), // indigo
      new THREE.Color(0x8b5cf6), // violet
      new THREE.Color(0xa855f7), // purple
      new THREE.Color(0x818cf8), // light indigo
      new THREE.Color(0x6d28d9), // deep violet
    ];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * SPREAD;
      positions[i3 + 1] = (Math.random() - 0.5) * SPREAD;
      positions[i3 + 2] = (Math.random() - 0.5) * SPREAD * 0.5;

      velocities[i3] = (Math.random() - 0.5) * 0.15;
      velocities[i3 + 1] = (Math.random() - 0.5) * 0.15;
      velocities[i3 + 2] = (Math.random() - 0.5) * 0.08;

      const color = palette[Math.floor(Math.random() * palette.length)];
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;
    }

    // Points geometry
    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );
    pointsGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(colors, 3)
    );

    const pointsMaterial = new THREE.PointsMaterial({
      size: 3,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const points = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(points);

    // Lines for connections
    const lineGeometry = new THREE.BufferGeometry();
    const maxLines = PARTICLE_COUNT * PARTICLE_COUNT;
    const linePositions = new Float32Array(maxLines * 6);
    const lineColors = new Float32Array(maxLines * 6);
    lineGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(linePositions, 3)
    );
    lineGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(lineColors, 3)
    );

    const lineMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    // Mouse tracking
    const mouse = { x: 0, y: 0 };
    const handleMouseMove = (e) => {
      mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.y = -(e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener("mousemove", handleMouseMove);

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    // Animation loop
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      const posAttr = pointsGeometry.attributes.position;
      const posArr = posAttr.array;

      // Update particle positions
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;
        posArr[i3] += velocities[i3];
        posArr[i3 + 1] += velocities[i3 + 1];
        posArr[i3 + 2] += velocities[i3 + 2];

        // Boundary wrapping
        const halfSpread = SPREAD / 2;
        if (posArr[i3] > halfSpread) posArr[i3] = -halfSpread;
        if (posArr[i3] < -halfSpread) posArr[i3] = halfSpread;
        if (posArr[i3 + 1] > halfSpread) posArr[i3 + 1] = -halfSpread;
        if (posArr[i3 + 1] < -halfSpread) posArr[i3 + 1] = halfSpread;
        if (posArr[i3 + 2] > halfSpread * 0.5) posArr[i3 + 2] = -halfSpread * 0.5;
        if (posArr[i3 + 2] < -halfSpread * 0.5) posArr[i3 + 2] = halfSpread * 0.5;
      }
      posAttr.needsUpdate = true;

      // Update connections
      let lineIdx = 0;
      const lp = lines.geometry.attributes.position.array;
      const lc = lines.geometry.attributes.color.array;

      for (let i = 0; i < PARTICLE_COUNT; i++) {
        for (let j = i + 1; j < PARTICLE_COUNT; j++) {
          const i3 = i * 3;
          const j3 = j * 3;
          const dx = posArr[i3] - posArr[j3];
          const dy = posArr[i3 + 1] - posArr[j3 + 1];
          const dz = posArr[i3 + 2] - posArr[j3 + 2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < CONNECTION_DISTANCE) {
            const alpha = 1 - dist / CONNECTION_DISTANCE;
            const li = lineIdx * 6;

            lp[li] = posArr[i3];
            lp[li + 1] = posArr[i3 + 1];
            lp[li + 2] = posArr[i3 + 2];
            lp[li + 3] = posArr[j3];
            lp[li + 4] = posArr[j3 + 1];
            lp[li + 5] = posArr[j3 + 2];

            // Fade color by distance
            const r = 0.39 * alpha;
            const g = 0.36 * alpha;
            const b = 0.96 * alpha;
            lc[li] = r; lc[li + 1] = g; lc[li + 2] = b;
            lc[li + 3] = r; lc[li + 4] = g; lc[li + 5] = b;

            lineIdx++;
          }
        }
      }

      // Clear remaining line slots
      for (let i = lineIdx * 6; i < lp.length; i++) {
        lp[i] = 0;
        lc[i] = 0;
      }

      lines.geometry.attributes.position.needsUpdate = true;
      lines.geometry.attributes.color.needsUpdate = true;
      lines.geometry.setDrawRange(0, lineIdx * 2);

      // Subtle camera movement based on mouse
      camera.position.x += (mouse.x * 30 - camera.position.x) * 0.02;
      camera.position.y += (mouse.y * 30 - camera.position.y) * 0.02;
      camera.lookAt(scene.position);

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      pointsGeometry.dispose();
      pointsMaterial.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}

import { Canvas } from '@react-three/fiber';
import { FilesetResolver, HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { useEffect, useRef, useState } from 'react';
import EffectsCanvas from './EffectsCanvas';

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastSwitchRef = useRef(0);

  const [videoReady, setVideoReady] = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const [effectIndex, setEffectIndex] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);

  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const boxRef = useRef<[number, number, number, number]>([0, 0, 0, 0]);

  useEffect(() => {
    let mounted = true;

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
          audio: false,
        });

        if (!videoRef.current) return;
        videoRef.current.srcObject = stream;

        await videoRef.current.play();

        if (!mounted || !videoRef.current) return;
        setVideoReady(true);
        setVideoEl(videoRef.current);

        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
        );

        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          numHands: 2,
          runningMode: 'VIDEO',
        });

        if (!mounted) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        setModelsReady(true);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown setup error.';
        setErrorMsg(message);
      }
    };

    start();

    return () => {
      mounted = false;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!videoReady || !modelsReady || !videoRef.current || !overlayRef.current || !landmarkerRef.current) return;

    const video = videoRef.current;
    const canvas = overlayRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const loop = () => {
      if (!video.videoWidth || !video.videoHeight || !landmarkerRef.current) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const result = landmarkerRef.current.detectForVideo(video, performance.now());
      const landmarks = result.landmarks;

      if (landmarks.length === 2) {
        const a = landmarks[0][9];
        const b = landmarks[1][9];

        const ax = 1 - a.x;
        const ay = a.y;
        const bx = 1 - b.x;
        const by = b.y;

        const centerX = (ax + bx) * 0.5;
        const centerY = (ay + by) * 0.5;
        const dx = ax - bx;
        const dy = ay - by;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 0.1) {
          boxRef.current = [0, 0, 0, 0];
          const now = Date.now();
          if (now - lastSwitchRef.current > 1000) {
            setEffectIndex((prev) => (prev + 1) % 6);
            lastSwitchRef.current = now;
          }
        } else {
          const width = dist * 1.2;
          const height = width * 0.8;
          boxRef.current = [
            Math.max(0, centerX - width / 2),
            Math.max(0, centerY - height / 2),
            Math.min(1, centerX + width / 2),
            Math.min(1, centerY + height / 2),
          ];
        }
      } else {
        boxRef.current = [0, 0, 0, 0];
      }

      drawSkeletons(ctx, landmarks, canvas.width, canvas.height);
      rafRef.current = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [videoReady, modelsReady]);

  if (errorMsg) {
    return (
      <div className="h-screen w-screen bg-zinc-950 text-white flex items-center justify-center">
        <div className="rounded-xl border border-red-500/40 bg-zinc-900/90 px-6 py-8 text-center max-w-md">
          <div className="text-red-400 text-4xl mb-3">⚠</div>
          <p className="text-red-200 mb-5">{errorMsg}</p>
          <button
            className="px-4 py-2 rounded bg-red-500 text-white hover:bg-red-400"
            onClick={() => window.location.reload()}
            type="button"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  if (!videoReady || !modelsReady || !videoEl) {
    return (
      <div className="h-screen w-screen bg-zinc-950 text-white flex flex-col items-center justify-center gap-4">
        <div className="h-12 w-12 rounded-full border-4 border-zinc-600 border-t-white animate-spin" />
        <p className="text-zinc-200 text-lg">{!videoReady ? 'Waiting for Camera...' : 'Loading AI Models...'}</p>
        <video ref={videoRef} className="hidden" playsInline muted />
      </div>
    );
  }

  return (
    <div className="relative h-screen w-screen bg-zinc-950 overflow-hidden">
      <video ref={videoRef} className="hidden" playsInline muted />
      <Canvas orthographic camera={{ position: [0, 0, 1], zoom: 1 }} gl={{ antialias: true }}>
        <EffectsCanvas video={videoEl} boxRef={boxRef} effectIndex={effectIndex} />
      </Canvas>
      <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  );
}

function drawSkeletons(
  ctx: CanvasRenderingContext2D,
  hands: NormalizedLandmark[][],
  width: number,
  height: number,
) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillStyle = '#ffffff';

  for (const hand of hands) {
    for (const [start, end] of HAND_CONNECTIONS) {
      const a = hand[start];
      const b = hand[end];
      if (!a || !b) continue;

      const ax = (1 - a.x) * width;
      const ay = a.y * height;
      const bx = (1 - b.x) * width;
      const by = b.y * height;

      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    for (const point of hand) {
      const x = (1 - point.x) * width;
      const y = point.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

import { useFrame, useThree } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

import type { MutableRefObject } from 'react';

type EffectsCanvasProps = {
  video: HTMLVideoElement;
  boxRef: MutableRefObject<[number, number, number, number]>;
  effectIndex: number;
};

const vertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

const fragmentShader = `
precision highp float;
uniform sampler2D uTexture;
uniform float uTime;
uniform vec4 uBox;
uniform float uEffect;
uniform vec2 uResolution;
varying vec2 vUv;

float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float snoise(vec2 p) {
  const float K1 = 0.366025404;
  const float K2 = 0.211324865;
  vec2 i = floor(p + (p.x + p.y) * K1);
  vec2 a = p - i + (i.x + i.y) * K2;
  vec2 o = step(a.yx, a.xy);
  vec2 b = a - o + K2;
  vec2 c = a - 1.0 + 2.0 * K2;
  vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
  vec3 n = h * h * h * h * vec3(dot(a, hash2(i + 0.0)), dot(b, hash2(i + o)), dot(c, hash2(i + 1.0)));
  return dot(n, vec3(70.0));
}

float luminance(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 burningGradient(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 0.33) {
    return mix(vec3(0.1, 0.0, 0.0), vec3(1.0, 0.0, 0.0), t / 0.33);
  }
  if (t < 0.66) {
    return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.33) / 0.33);
  }
  return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.66) / 0.34);
}

void main() {
  vec2 uv = vec2(1.0 - vUv.x, vUv.y);
  vec3 baseColor = texture2D(uTexture, uv).rgb;

  vec2 pixel = vec2(1.0) / uResolution;

  float burnNoise = snoise(uv * 5.0 + vec2(0.0, uTime * 0.7));
  vec2 burnUv = uv + vec2(burnNoise * 0.03, burnNoise * 0.02);
  vec3 burnSample = texture2D(uTexture, clamp(burnUv, 0.0, 1.0)).rgb;
  float burnLum = luminance(burnSample);

  float glowNoise = snoise(uv * 200.0 + uTime * 0.5) * 0.15;

  float thermalLum = luminance(baseColor);

  float aspect = uResolution.x / uResolution.y;
  float gridCount = 80.0;
  vec2 grid = vec2(gridCount, gridCount / aspect);
  vec2 gridUv = floor(uv * grid) / grid;
  vec3 pixelSample = texture2D(uTexture, clamp(gridUv, 0.0, 1.0)).rgb;
  float pixelLum = luminance(pixelSample);
  vec2 cellUv = fract(uv * grid) - 0.5;
  float cellDist = length(cellUv);

  float glitchN = snoise(vec2(uv.y * 12.0, uTime * 2.0));
  float shift = glitchN * 0.02;
  vec3 glitchColor = vec3(
    texture2D(uTexture, clamp(uv + vec2(shift, 0.0), 0.0, 1.0)).r,
    texture2D(uTexture, clamp(uv + vec2(-shift * 0.5, 0.0), 0.0, 1.0)).g,
    texture2D(uTexture, clamp(uv + vec2(shift * 0.75, 0.0), 0.0, 1.0)).b
  );

  vec3 s00 = texture2D(uTexture, uv + vec2(-pixel.x, -pixel.y)).rgb;
  vec3 s01 = texture2D(uTexture, uv + vec2(0.0, -pixel.y)).rgb;
  vec3 s02 = texture2D(uTexture, uv + vec2(pixel.x, -pixel.y)).rgb;
  vec3 s10 = texture2D(uTexture, uv + vec2(-pixel.x, 0.0)).rgb;
  vec3 s12 = texture2D(uTexture, uv + vec2(pixel.x, 0.0)).rgb;
  vec3 s20 = texture2D(uTexture, uv + vec2(-pixel.x, pixel.y)).rgb;
  vec3 s21 = texture2D(uTexture, uv + vec2(0.0, pixel.y)).rgb;
  vec3 s22 = texture2D(uTexture, uv + vec2(pixel.x, pixel.y)).rgb;

  float l00 = luminance(s00);
  float l01 = luminance(s01);
  float l02 = luminance(s02);
  float l10 = luminance(s10);
  float l12 = luminance(s12);
  float l20 = luminance(s20);
  float l21 = luminance(s21);
  float l22 = luminance(s22);

  float gx = -l00 - 2.0 * l10 - l20 + l02 + 2.0 * l12 + l22;
  float gy = -l00 - 2.0 * l01 - l02 + l20 + 2.0 * l21 + l22;
  float sobel = length(vec2(gx, gy));

  if (uBox.z <= 0.0 || uv.x < uBox.x || uv.x > uBox.z || uv.y < uBox.y || uv.y > uBox.w) {
    gl_FragColor = vec4(baseColor, 1.0);
    return;
  }

  float border = 0.005;
  if (abs(uv.x - uBox.x) < border || abs(uv.x - uBox.z) < border || abs(uv.y - uBox.y) < border || abs(uv.y - uBox.w) < border) {
    gl_FragColor = vec4(mix(baseColor, vec3(1.0), 0.6), 1.0);
    return;
  }

  vec3 finalColor = baseColor;

  if (uEffect < 0.5) {
    finalColor = burningGradient(burnLum);
  } else if (uEffect < 1.5) {
    float lum = pow(luminance(baseColor), 1.2) * 1.5;
    float core = smoothstep(0.5 + glowNoise, 0.7 + glowNoise, lum);
    float halo = smoothstep(0.2 + glowNoise, 0.6 + glowNoise, lum);
    vec3 glowCol = mix(vec3(0.0), vec3(0.4, 0.9, 1.0), halo);
    finalColor = mix(glowCol, vec3(1.0), core);
  } else if (uEffect < 2.5) {
    float t = clamp((thermalLum - 0.1) * 1.2, 0.0, 1.0);
    if (t < 0.25) {
      finalColor = mix(vec3(0.0, 0.0, 0.2), vec3(0.1, 0.0, 1.0), t / 0.25);
    } else if (t < 0.5) {
      finalColor = mix(vec3(0.1, 0.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) / 0.25);
    } else if (t < 0.75) {
      finalColor = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.9, 0.0), (t - 0.5) / 0.25);
    } else {
      finalColor = mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) / 0.25);
    }
  } else if (uEffect < 3.5) {
    float on = step(cellDist, 0.35);
    float bright = step(0.25, pixelLum);
    vec3 dotColor = vec3(0.0, bright, 0.0);
    finalColor = mix(vec3(0.0, 0.1, 0.0), dotColor, on);
  } else if (uEffect < 4.5) {
    finalColor = glitchColor;
    finalColor -= sin(uv.y * 800.0 + uTime * 10.0) * 0.05;
  } else if (uEffect < 5.5) {
    vec3 edgeColor = vec3(0.1, 1.0, 0.8) * sobel * 2.5;
    finalColor = edgeColor + baseColor * 0.3;
  }

  gl_FragColor = vec4(finalColor, 1.0);
}
`;

export default function EffectsCanvas({ video, boxRef, effectIndex }: EffectsCanvasProps) {
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const { size } = useThree();

  const texture = useMemo(() => {
    const t = new THREE.VideoTexture(video);
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
    t.wrapS = THREE.ClampToEdgeWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  }, [video]);

  useFrame(({ clock }) => {
    if (!matRef.current) return;
    matRef.current.uniforms.uTime.value = clock.getElapsedTime();
    matRef.current.uniforms.uBox.value.set(...boxRef.current);
    matRef.current.uniforms.uEffect.value = effectIndex;
    matRef.current.uniforms.uResolution.value.set(size.width, size.height);
  });

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          uTexture: { value: texture },
          uTime: { value: 0 },
          uBox: { value: new THREE.Vector4(0, 0, 0, 0) },
          uEffect: { value: 0 },
          uResolution: { value: new THREE.Vector2(size.width, size.height) },
        }}
      />
    </mesh>
  );
}

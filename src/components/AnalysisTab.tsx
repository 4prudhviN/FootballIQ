import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  Lightbulb,
  ArrowRight,
  RefreshCw,
  Play,
  Target,
  Crosshair,
  Dumbbell,
  Heart,
  Footprints,
} from "lucide-react";

import type { AnalysisResult, PillarTab } from "../types";
import PillarTechnical from "./PillarTechnical";
import PillarSetPiece from "./PillarSetPiece";
import PillarGymPlyo from "./PillarGymPlyo";
import PillarStamina from "./PillarStamina";

interface AnalysisTabProps {
  results: AnalysisResult[];
  onUploadMore: () => void;
}

/* ── Sub-navigation ──────────────────────────────────── */
const PILLAR_ITEMS: {
  id: PillarTab;
  label: string;
  shortLabel: string;
  icon: React.ReactNode;
}[] = [
  { id: "technical", label: "Technical Skills", shortLabel: "Technical", icon: <Footprints size={16} /> },
  { id: "setpiece", label: "Set-Piece Mastery", shortLabel: "Set-Piece", icon: <Crosshair size={16} /> },
  { id: "gymplyo", label: "Gym & Plyometrics", shortLabel: "Gym & Plyo", icon: <Dumbbell size={16} /> },
  { id: "stamina", label: "Stamina & Cardio", shortLabel: "Stamina", icon: <Heart size={16} /> },
];

/* ── Skeleton ────────────────────────────────────────── */
interface Joint { x: number; y: number }
interface SkeletonData { joints: Joint[]; connections: [number, number][] }

const SKELETON: SkeletonData = {
  joints: [
    { x: 0.50, y: 0.28 }, { x: 0.55, y: 0.28 }, { x: 0.52, y: 0.52 }, { x: 0.58, y: 0.50 },
    { x: 0.53, y: 0.78 }, { x: 0.59, y: 0.76 }, { x: 0.48, y: 0.15 }, { x: 0.54, y: 0.15 },
    { x: 0.46, y: 0.40 }, { x: 0.61, y: 0.38 },
  ],
  connections: [[0,2],[1,3],[2,4],[3,5],[6,0],[7,1],[0,1],[2,3],[4,5],[6,7],[0,8],[1,9]],
};

interface SkeletonOverlayOptions {
  joints: boolean;
  bones: boolean;
  angles: boolean;
  trajectories: boolean;
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  data: SkeletonData,
  w: number,
  h: number,
  options: SkeletonOverlayOptions,
) {
  ctx.clearRect(0, 0, w, h);

  const toPx = (j: Joint) => ({ x: j.x * w, y: j.y * h });

  /* ── Trajectory vectors ── */
  if (options.trajectories) {
    data.joints.forEach((j, idx) => {
      if (idx >= 8) return;
      const p = toPx(j);
      const dx = (Math.random() - 0.5) * 30;
      const dy = Math.random() * 30 + 10;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + dx, p.y + dy);
      ctx.strokeStyle = "oklch(0.55 0.18 145 / 0.3)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(p.x + dx, p.y + dy);
      ctx.lineTo(p.x + dx - 5, p.y + dy - 4);
      ctx.lineTo(p.x + dx - 5, p.y + dy + 4);
      ctx.closePath();
      ctx.fillStyle = "oklch(0.55 0.18 145 / 0.3)";
      ctx.fill();
    });
  }

  /* ── Bones ── */
  if (options.bones) {
    data.connections.forEach(([i, j], idx) => {
      const a = toPx(data.joints[i]), b = toPx(data.joints[j]);
      const isGhost = idx >= 10;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = isGhost
        ? "oklch(0.75 0.16 85 / 0.35)"
        : "oklch(0.55 0.18 145 / 0.85)";
      ctx.lineWidth = isGhost ? 2 : 3;
      ctx.setLineDash(isGhost ? [4, 6] : []); ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  /* ── Joints ── */
  if (options.joints) {
    data.joints.forEach((j, idx) => {
      const p = toPx(j);
      const isGhost = idx >= 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, isGhost ? 3 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = isGhost
        ? "oklch(0.75 0.16 85 / 0.5)"
        : "oklch(0.55 0.18 145)";
      ctx.fill();
      if (!isGhost) {
        ctx.strokeStyle = "oklch(0 0 0 / 0.6)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }

  /* ── Joint Angles ── */
  if (options.angles) {
    const hip = toPx(data.joints[2]);
    const knee = toPx(data.joints[4]);
    const ankle = toPx(data.joints[6]);
    ctx.beginPath();
    ctx.arc(knee.x, knee.y, 18, Math.atan2(hip.y - knee.y, hip.x - knee.x), Math.atan2(ankle.y - knee.y, ankle.x - knee.x));
    ctx.strokeStyle = "oklch(0.6 0.22 85 / 0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "oklch(0.6 0.22 85)";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText("167°", knee.x + 22, knee.y - 6);

    const shoulder = toPx(data.joints[0]);
    const midHip = toPx(data.joints[2]);
    ctx.beginPath();
    ctx.arc(midHip.x, midHip.y, 14, -Math.PI / 2, Math.atan2(shoulder.y - midHip.y, shoulder.x - midHip.x));
    ctx.strokeStyle = "oklch(0.5 0.16 30 / 0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "oklch(0.5 0.16 30)";
    ctx.fillText("12°", midHip.x - 30, midHip.y - 6);
  }
}

/* ── VideoPlayer ─────────────────────────────────────── */
function VideoPlayer({ label, subtitle, gradient, skeleton, skeletonOptions, videoUrl }: {
  label: string;
  subtitle?: string;
  gradient: string;
  skeleton?: boolean;
  skeletonOptions?: SkeletonOverlayOptions;
  videoUrl?: string | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoPlaying, setVideoPlaying] = useState(false);

  const playVideo = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.play().catch(() => {});
      setVideoPlaying(true);
    }
  }, []);

  useEffect(() => {
    if (!skeleton || !canvasRef.current || !skeletonOptions) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const resize = () => {
      const parent = canvas.parentElement!;
      const w = parent.clientWidth, h = parent.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
      ctx.scale(dpr, dpr);
      drawSkeleton(ctx, SKELETON, w, h, skeletonOptions);
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [skeleton, skeletonOptions]);

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-border bg-black/60 shadow-lg">
      <div className={`aspect-video w-full ${gradient} flex items-center justify-center bg-cover bg-center relative`}>
        {videoUrl ? (
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              className="absolute inset-0 h-full w-full object-contain"
              playsInline
              onPlay={() => setVideoPlaying(true)}
              onPause={() => setVideoPlaying(false)}
            />
            {/* Custom play overlay on top of paused video */}
            {!videoPlaying && (
              <button
                onClick={playVideo}
                className="absolute inset-0 z-10 flex items-center justify-center bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 cursor-pointer"
                aria-label={`Play ${label}`}
              >
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm transition-all duration-200 hover:bg-white/30 active:scale-95">
                  <Play size={24} className="ml-0.5 text-white" fill="white" />
                </span>
              </button>
            )}
          </>
        ) : (
          <>
            <div className="absolute inset-0 opacity-[0.06]">
              <div className="h-full w-full" style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(255,255,255,0.06) 40px, rgba(255,255,255,0.06) 41px), repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(255,255,255,0.06) 40px, rgba(255,255,255,0.06) 41px)" }} />
            </div>
            <div className="flex flex-col items-center gap-3 relative z-10">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white/10">
                <Play size={22} className="ml-0.5 text-white/40" fill="white" />
              </div>
              <p className="text-xs font-medium text-white/40">No video recorded</p>
            </div>
          </>
        )}
      </div>
      {skeleton && skeletonOptions && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full pointer-events-none"
          aria-label="Skeleton pose overlay"
        />
      )}
      <div className="flex items-center justify-between border-t border-border bg-surface/90 px-4 py-3 backdrop-blur-sm">
        <div>
          <p className="text-sm font-semibold text-foreground">{label}</p>
          {subtitle && <p className="text-xs text-muted">{subtitle}</p>}
        </div>
        <span className="flex h-2 w-2 rounded-full bg-success animate-pulse" />
      </div>
    </div>
  );
}

/* ── Skeleton Overlay Toggle Row ─────────────────────── */
function OverlayToggles({
  options,
  onChange,
}: {
  options: SkeletonOverlayOptions;
  onChange: (key: keyof SkeletonOverlayOptions) => void;
}) {
  const items: { key: keyof SkeletonOverlayOptions; label: string }[] = [
    { key: "joints", label: "Joints" },
    { key: "bones", label: "Bones" },
    { key: "angles", label: "Joint Angles" },
    { key: "trajectories", label: "Trajectory Vectors" },
  ];

  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-border/50 bg-surface/50 px-4 py-2.5">
      {items.map((item) => (
        <label
          key={item.key}
          className="flex cursor-pointer items-center gap-2 text-[11px] font-medium text-muted select-none"
        >
          <input
            type="checkbox"
            checked={options[item.key]}
            onChange={() => onChange(item.key)}
            className="peer sr-only"
          />
          <span className="flex h-4 w-4 items-center justify-center rounded border border-border bg-background transition-all duration-150 peer-checked:border-primary peer-checked:bg-primary/20 peer-focus-visible:ring-2 peer-focus-visible:ring-ring/50">
            {options[item.key] && (
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            )}
          </span>
          <span className="peer-checked:text-foreground transition-colors duration-150">{item.label}</span>
        </label>
      ))}
    </div>
  );
}

/* ── Target Heatmap ──────────────────────────────────── */
function TargetHeatmap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const goalLeft = w * 0.15;
    const goalRight = w * 0.85;
    const goalTop = h * 0.1;
    const goalBottom = h * 0.85;
    const goalW = goalRight - goalLeft;
    const goalH = goalBottom - goalTop;

    ctx.fillStyle = "oklch(0.15 0.05 145 / 0.4)";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "oklch(0.6 0.1 145 / 0.6)";
    ctx.lineWidth = 3;
    ctx.strokeRect(goalLeft, goalTop, goalW, goalH);

    ctx.strokeStyle = "oklch(0.55 0.18 145 / 0.4)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(goalLeft - 5, goalTop - 5, goalW + 10, goalH + 10);
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(w / 2, goalTop);
    ctx.lineTo(w / 2, goalBottom);
    ctx.strokeStyle = "oklch(0.4 0.08 145 / 0.3)";
    ctx.lineWidth = 1;
    ctx.stroke();

    for (let row = 1; row < 3; row++) {
      const y = goalTop + (goalH / 3) * row;
      ctx.beginPath();
      ctx.moveTo(goalLeft, y);
      ctx.lineTo(goalRight, y);
      ctx.strokeStyle = "oklch(0.4 0.08 145 / 0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = "oklch(0.5 0.1 145 / 0.3)";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "center";
    const corners = [
      { x: goalLeft + goalW * 0.17, y: goalTop + goalH * 0.12, label: "TL" },
      { x: goalRight - goalW * 0.17, y: goalTop + goalH * 0.12, label: "TR" },
      { x: goalLeft + goalW * 0.17, y: goalBottom - goalH * 0.08, label: "BL" },
      { x: goalRight - goalW * 0.17, y: goalBottom - goalH * 0.08, label: "BR" },
    ];
    corners.forEach((c) => { ctx.fillText(c.label, c.x, c.y); });

    const impacts = [
      { x: 0.2, y: 0.25, r: 10, v: 0.9 },
      { x: 0.8, y: 0.18, r: 14, v: 0.7 },
      { x: 0.5, y: 0.35, r: 8, v: 0.5 },
      { x: 0.12, y: 0.5, r: 6, v: 0.4 },
      { x: 0.9, y: 0.42, r: 11, v: 0.8 },
      { x: 0.3, y: 0.12, r: 9, v: 0.6 },
      { x: 0.7, y: 0.55, r: 5, v: 0.3 },
      { x: 0.45, y: 0.22, r: 16, v: 1.0 },
      { x: 0.25, y: 0.4, r: 7, v: 0.45 },
    ];

    impacts.forEach((imp) => {
      const px = goalLeft + imp.x * goalW;
      const py = goalTop + imp.y * goalH;
      const radius = imp.r;
      const grad = ctx.createRadialGradient(px, py, 0, px, py, radius * 2.5);
      grad.addColorStop(0, `oklch(${0.55 + imp.v * 0.15} 0.2 145 / ${0.35 * imp.v})`);
      grad.addColorStop(0.5, `oklch(${0.5 + imp.v * 0.1} 0.18 145 / ${0.15 * imp.v})`);
      grad.addColorStop(1, "oklch(0.5 0.18 145 / 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, radius * 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `oklch(${0.55 + imp.v * 0.15} 0.18 145 / ${0.6 + imp.v * 0.3})`;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = `oklch(0.55 0.18 145 / ${0.2 * imp.v})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, py, radius * 0.8, 0, Math.PI * 2);
      ctx.stroke();
    });

    ctx.fillStyle = "oklch(0.4 0.05 145 / 0.4)";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${impacts.length} impacts logged`, w - 8, h - 6);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full rounded-xl"
      style={{ minHeight: 200 }}
      aria-label="Football goal impact heatmap showing target zone matrix"
    />
  );
}

/* ── Severity badge ──────────────────────────────────── */
function severityColor(severity: "high" | "medium" | "low") {
  return severity === "high"
    ? "bg-destructive/15 text-destructive"
    : severity === "medium"
      ? "bg-accent/15 text-accent-foreground"
      : "bg-success/15 text-success";
}

/* ── Main ────────────────────────────────────────────── */
export default function AnalysisTab({ results, onUploadMore }: AnalysisTabProps) {
  const latest = results.find((r) => r.status === "completed") ?? null;
  const videoUrl = latest?.videoUrl ?? null;
  const fileName = latest?.fileName ?? null;
  const warnings = latest?.warnings ?? [];
  const hasWarnings = warnings.length > 0;
  const [activePillar, setActivePillar] = useState<PillarTab>("technical");

  const [overlayOptions, setOverlayOptions] = useState<SkeletonOverlayOptions>({
    joints: true,
    bones: true,
    angles: false,
    trajectories: false,
  });

  const toggleOverlay = useCallback((key: keyof SkeletonOverlayOptions) => {
    setOverlayOptions((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const feedbackItems = hasWarnings
    ? warnings.map((w, i) => ({ id: i, title: w, severity: (i === 0 ? "high" : i === 1 ? "medium" : "low") as const }))
    : [
        { id: 0, title: "Body leaning too far back", severity: "high" as const },
        { id: 1, title: "Knee stability — slight inward collapse detected", severity: "medium" as const },
        { id: 2, title: "Gait asymmetry — stride length imbalance", severity: "low" as const },
      ];

  const technical = latest?.technical ?? {
    coneAgility: "4.2s",
    changeOfDirectionSpeed: "5.8 m/s",
    touchTightness: "92%",
    wallReboundAccuracy: "87%",
    firstTouchControlCushion: "0.36 m/s²",
    weakFootRatio: "78/22",
    shotVelocity: "88 km/h",
    launchAngleElevation: "14°",
    apexTargetAccuracy: "81%",
  };
  const setPieces = latest?.setPieces ?? {
    curveRate: "420 RPM",
    wallClearance: "0.8m",
    targetCornerAccuracy: "76%",
    penaltyForce: "920 N",
    goalkeeperDeceptionIndex: "64%",
    gaitVectorConsistency: "88%",
    throwInSpineFlexion: "18°",
    elbowFlexionAcceleration: "4.2 m/s²",
    releasePointTrajectory: "24.5m",
  };
  const gymPlyo = latest?.gymPlyo ?? {
    barVelocity: "0.65 m/s",
    eccentricControlRatio: "0.82",
    leftRightForceBalancing: "92%",
    verticalJumpHeight: "58 cm",
    groundContactTime: "180 ms",
    reactiveStrengthIndex: "2.4",
    hipExtensionAngle: "168°",
    kneeFlexionAngle: "92°",
  };
  const stamina = latest?.stamina ?? {
    maxSprintVelocity: "31.2 km/h",
    strideAsymmetry: "4.2%",
    fatigueDropoff: "12% after minute 3",
    strideLengthConsistency: "94%",
    accelerationBurstTime: "1.8s",
    workToRestRatio: "2.3:1",
    vo2MaxEstimate: "52 mL/kg/min",
    heartRateRecovery: "22 bpm drop in 1 min",
    lactateThresholdPace: "4:30/km",
    strideFrequency: "178 spm",
    strideLength: "1.24 m",
    verticalOscillation: "7.2 cm",
    groundContactBalance: "49.2% left/right",
    brakingForceSymmetry: "92%",
    strideExtensionAsymmetry: "3.8%",
    peakVelocity: "32.4 km/h",
    mechanicalEfficiencyDropoff: "12%",
  };
  const feedback = latest?.feedback ?? {
    title: "Body leaning too far back & High Ground Contact Time",
    description:
      "Your torso is angled behind your hips at contact during free kicks, and your plyometric ground contact is sluggish. Keep your chest forward and drive dynamically off the balls of your feet.",
    drillName: "Dead-Ball Linear Drives & Box Jumps",
    drillDetails:
      "Execute 5 free kicks focusing entirely on keeping forward shoulder alignment, followed immediately by 3 sets of 8 explosive box jumps.",
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6 sm:space-y-8">
      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">Analysis</h1>
          <p className="mt-1 text-sm text-muted">
            {fileName
              ? `Results for "${fileName}"`
              : "Compare your form side-by-side with the reference technique."}
          </p>
        </div>
        {fileName && (
          <button
            onClick={onUploadMore}
            className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2 text-sm font-medium text-muted hover:text-foreground hover:bg-surface-hover transition-all duration-200 active:scale-[0.97] cursor-pointer"
          >
            <RefreshCw size={14} /> Upload New
          </button>
        )}
      </div>

      {/* ── Warnings banner ── */}
      {hasWarnings && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/5 p-4"
        >
          <AlertTriangle size={20} className="mt-0.5 flex-shrink-0 text-destructive" strokeWidth={2} />
          <div>
            <p className="text-sm font-semibold text-foreground">
              {warnings.length} issue{warnings.length > 1 ? "s" : ""} detected
            </p>
            <p className="mt-0.5 text-xs text-muted">
              Review the feedback below and try the suggested corrective drills.
            </p>
          </div>
        </motion.div>
      )}

      {/* ── Split-Screen Video Telemetry ── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 sm:gap-6">
        <div>
          <VideoPlayer
            label="Subject Vector Telemetry"
            subtitle="Live skeleton tracking active"
            gradient="bg-gradient-to-br from-emerald-950/70 via-emerald-900/30 to-black"
            skeleton
            skeletonOptions={overlayOptions}
            videoUrl={videoUrl}
          />
          <OverlayToggles options={overlayOptions} onChange={toggleOverlay} />
        </div>
        <div>
          <VideoPlayer
            label="Pro Baseline Standard"
            subtitle="Reference technique"
            gradient="bg-gradient-to-br from-blue-950/70 via-slate-900/30 to-black"
          />
        </div>
      </div>

      {/* ── Omni-Metric Sub-navigation ── */}
      <div className="flex overflow-x-auto gap-1 rounded-2xl border border-border bg-surface p-1.5 scrollbar-none">
        {PILLAR_ITEMS.map((p) => (
          <button
            key={p.id}
            onClick={() => setActivePillar(p.id)}
            className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs font-semibold transition-all duration-200 whitespace-nowrap cursor-pointer ${
              activePillar === p.id
                ? "bg-primary/15 text-primary shadow-sm"
                : "text-muted hover:text-foreground hover:bg-surface-hover"
            }`}
          >
            <span className={activePillar === p.id ? "text-primary" : "text-muted"}>{p.icon}</span>
            <span className="hidden sm:inline">{p.label}</span>
            <span className="sm:hidden">{p.shortLabel}</span>
          </button>
        ))}
      </div>

      {/* ── Pillar content + Heatmap ── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-2xl border border-border bg-surface p-5 sm:p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activePillar}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25 }}
            >
              {activePillar === "technical" && <PillarTechnical metrics={technical} />}
              {activePillar === "setpiece" && <PillarSetPiece metrics={setPieces} />}
              {activePillar === "gymplyo" && <PillarGymPlyo metrics={gymPlyo} />}
              {activePillar === "stamina" && <PillarStamina metrics={stamina} />}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* ── Heatmap ── */}
        <div className="rounded-2xl border border-border bg-surface p-5 sm:p-6">
          <div className="flex items-center gap-2 mb-3">
            <Target size={16} className="text-primary" />
            <h3 className="font-heading text-sm font-semibold text-foreground">Impact Distribution</h3>
          </div>
          <p className="mb-3 text-[11px] text-muted">
            Kick &amp; throw-in landing positions — goal zone matrix
          </p>
          <TargetHeatmap />
          <div className="mt-3 flex items-center justify-between border-t border-border/50 pt-3 text-[11px] text-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full bg-primary/70" /> On-target
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full bg-accent/60" /> Off-target
            </span>
          </div>
        </div>
      </div>

      {/* ── Correction Hub ── */}
      <div className="rounded-2xl border border-border bg-surface p-5 sm:p-6">
        <div className="mb-4 flex items-center gap-2">
          <AlertTriangle size={18} className="text-destructive" strokeWidth={2.2} />
          <h2 className="font-heading text-base font-semibold text-foreground">Diagnostic Feedback</h2>
        </div>

        <div className="space-y-3">
          {feedbackItems.map((item) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: item.id * 0.1 }}
              className="flex items-start gap-3 rounded-xl border border-border bg-background/60 p-4"
            >
              <div className="mt-0.5 flex-shrink-0">
                <div className={`inline-flex h-8 w-8 items-center justify-center rounded-lg ${severityColor(item.severity)}`}>
                  <AlertTriangle size={16} strokeWidth={2.2} />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground">{item.title}</p>
                <span className={`mt-1 inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold ${severityColor(item.severity)}`}>
                  {item.severity === "high" ? "High priority" : item.severity === "medium" ? "Medium" : "Suggestion"}
                </span>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-5 flex flex-col gap-4 rounded-xl border border-accent/30 bg-accent/[0.04] p-5 sm:flex-row sm:items-center"
        >
          <div className="flex-shrink-0">
            <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-accent/10">
              <Lightbulb size={22} className="text-accent" strokeWidth={2} />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-foreground">{feedback.title}</p>
            <p className="text-xs text-accent/80 font-medium mt-0.5">{feedback.drillName}</p>
            <p className="mt-1.5 text-sm leading-relaxed text-muted">{feedback.description}</p>
            <p className="mt-2 text-xs text-muted/70">{feedback.drillDetails}</p>
          </div>
          <div className="flex flex-shrink-0 items-center gap-3 sm:flex-col sm:items-end">
            <span className="inline-flex items-center gap-1 rounded-full bg-surface-hover px-2.5 py-1 text-xs font-medium text-muted">
              <RefreshCw size={12} /> 12 min
            </span>
            <button className="inline-flex items-center gap-1.5 rounded-xl bg-accent px-4 py-2 text-xs font-semibold text-black transition-all duration-200 hover:brightness-110 active:scale-[0.97] cursor-pointer">
              Start drill <ArrowRight size={14} />
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
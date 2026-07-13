import type { StaminaMetrics } from "../types";
import { motion } from "framer-motion";
import {
  Zap,
  Activity,
  Thermometer,
  Timer,
  Ruler,
  Gauge,
  Shield,
  Wind,
  Heart,
  TrendingDown,
  ArrowUp,
  RefreshCw,
  Footprints,
  Scale,
  Move,
  Target,
} from "lucide-react";

interface Props {
  metrics: StaminaMetrics;
}

/* ── Reusable metric card ────────────────────────────── */
function MetricCard({
  label,
  value,
  icon,
  color = "primary",
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  color?: "primary" | "accent" | "info" | "warning" | "rose";
}) {
  const bgMap = {
    primary: "bg-primary/8",
    accent: "bg-accent/8",
    info: "bg-sky-500/8",
    warning: "bg-amber-500/8",
    rose: "bg-rose-500/8",
  };
  const textMap = {
    primary: "text-primary",
    accent: "text-accent",
    info: "text-sky-400",
    warning: "text-amber-400",
    rose: "text-rose-400",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-surface p-4 transition-all duration-200 hover:border-primary/40"
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-muted">
          {label}
        </span>
        <span
          className={`flex h-8 w-8 items-center justify-center rounded-lg ${bgMap[color]}`}
        >
          <span className={textMap[color]}>{icon}</span>
        </span>
      </div>
      <p className="mt-2 font-heading text-2xl font-bold tracking-tight text-foreground">
        {value}
      </p>
    </motion.div>
  );
}

function SectionHeading({ title }: { title: string }) {
  return (
    <h3 className="mb-3 font-heading text-sm font-semibold uppercase tracking-widest text-muted">
      {title}
    </h3>
  );
}

/* ── Fatigue Degradation Mini-Chart ──────────────────── */
function FatigueTrendBar() {
  const segments = [
    { label: "Min 1", value: 100, color: "bg-primary" },
    { label: "Min 2", value: 94, color: "bg-primary" },
    { label: "Min 3", value: 88, color: "bg-amber-500" },
    { label: "Min 4", value: 78, color: "bg-destructive" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-surface p-5"
    >
      <div className="mb-4 flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-muted">
          Fatigue &amp; Degradation Trend
        </span>
        <span className="text-xs font-semibold text-destructive">-22% drop-off</span>
      </div>
      <div className="flex items-end gap-3">
        {segments.map((seg) => (
          <div key={seg.label} className="flex flex-1 flex-col items-center gap-1.5">
            <div className="flex h-32 w-full items-end rounded-lg bg-border/40">
              <div
                className={`w-full rounded-t-lg transition-all duration-500 ${seg.color}`}
                style={{ height: `${seg.value}%` }}
              />
            </div>
            <span className="text-[10px] font-medium text-muted">{seg.label}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ── Aerobic Power ───────────────────────────────────── */
function AerobicSection(metrics: StaminaMetrics) {
  return (
    <div>
      <SectionHeading title="Aerobic Power &amp; Recovery" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <MetricCard
          label="VO₂ Max Estimate"
          value={metrics.vo2MaxEstimate}
          color="info"
          icon={<Wind size={16} />}
        />
        <MetricCard
          label="Heart Rate Recovery"
          value={metrics.heartRateRecovery}
          color="success"
          icon={<Heart size={16} />}
        />
        <MetricCard
          label="Lactate Threshold Pace"
          value={metrics.lactateThresholdPace}
          color="warning"
          icon={<Thermometer size={16} />}
        />
      </div>
    </div>
  );
}

/* ── Running Gait & Mechanics ────────────────────────── */
function GaitSection(metrics: StaminaMetrics) {
  return (
    <div>
      <SectionHeading title="Running Gait &amp; Mechanics" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        <MetricCard
          label="Stride Frequency"
          value={metrics.strideFrequency}
          color="primary"
          icon={<Activity size={16} />}
        />
        <MetricCard
          label="Stride Length"
          value={metrics.strideLength}
          color="info"
          icon={<Ruler size={16} />}
        />
        <MetricCard
          label="Vertical Oscillation"
          value={metrics.verticalOscillation}
          color="warning"
          icon={<Move size={16} />}
        />
        <MetricCard
          label="Ground Contact Balance"
          value={metrics.groundContactBalance}
          color="accent"
          icon={<Scale size={16} />}
        />
      </div>
    </div>
  );
}

/* ── Sprint & Explosiveness ──────────────────────────── */
function SprintSection(metrics: StaminaMetrics) {
  return (
    <div>
      <SectionHeading title="Sprint &amp; Explosiveness" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        <MetricCard
          label="Max Sprint Velocity"
          value={metrics.maxSprintVelocity}
          color="primary"
          icon={<Zap size={16} />}
        />
        <MetricCard
          label="Acceleration Burst Time"
          value={metrics.accelerationBurstTime}
          color="info"
          icon={<Timer size={16} />}
        />
        <MetricCard
          label="Peak Velocity"
          value={metrics.peakVelocity}
          color="accent"
          icon={<ArrowUp size={16} />}
        />
        <MetricCard
          label="Braking Force Symmetry"
          value={metrics.brakingForceSymmetry}
          color="success"
          icon={<Shield size={16} />}
        />
      </div>
    </div>
  );
}

/* ── Asymmetry & Efficiency ──────────────────────────── */
function AsymmetrySection(metrics: StaminaMetrics) {
  return (
    <div>
      <SectionHeading title="Asymmetry &amp; Efficiency" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        <MetricCard
          label="Stride Length Consistency"
          value={metrics.strideLengthConsistency}
          color="primary"
          icon={<Target size={16} />}
        />
        <MetricCard
          label="Stride Asymmetry"
          value={metrics.strideAsymmetry}
          color="warning"
          icon={<Scale size={16} />}
        />
        <MetricCard
          label="Stride Extension Asymmetry"
          value={metrics.strideExtensionAsymmetry}
          color="rose"
          icon={<Footprints size={16} />}
        />
        <MetricCard
          label="Mechanical Efficiency Drop-off"
          value={metrics.mechanicalEfficiencyDropoff}
          color="destructive"
          icon={<TrendingDown size={16} />}
        />
      </div>
    </div>
  );
}

/* ── Work-to-Rest & Fatigue ──────────────────────────── */
function WorkRestSection(metrics: StaminaMetrics) {
  return (
    <div>
      <SectionHeading title="Work-to-Rest &amp; Fatigue Index" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        <div className="sm:col-span-2">
          <FatigueTrendBar />
        </div>
        <MetricCard
          label="Work / Rest Ratio"
          value={metrics.workToRestRatio}
          color="primary"
          icon={<RefreshCw size={16} />}
        />
        <MetricCard
          label="Fatigue Drop-Off"
          value={metrics.fatigueDropoff}
          color="warning"
          icon={<TrendingDown size={16} />}
        />
      </div>
    </div>
  );
}

export default function PillarStamina({ metrics }: Props) {
  return (
    <div className="space-y-8">
      <AerobicSection {...metrics} />
      <GaitSection {...metrics} />
      <SprintSection {...metrics} />
      <AsymmetrySection {...metrics} />
      <WorkRestSection {...metrics} />
    </div>
  );
}
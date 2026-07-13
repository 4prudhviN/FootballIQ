import type { TechnicalMetrics } from "../types";
import { motion } from "framer-motion";

interface Props {
  metrics: TechnicalMetrics;
}

/* ── Metric card ────────────────────────────────────── */
function MetricCard({
  label,
  value,
  icon,
  color = "primary",
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  color?: "primary" | "accent" | "info" | "warning";
}) {
  const borderMap = {
    primary: "border-primary/30",
    accent: "border-accent/30",
    info: "border-sky-500/30",
    warning: "border-amber-500/30",
  };
  const bgMap = {
    primary: "bg-primary/8",
    accent: "bg-accent/8",
    info: "bg-sky-500/8",
    warning: "bg-amber-500/8",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border ${borderMap[color]} bg-surface p-4 transition-all duration-200 hover:border-${color}/50`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-muted">
          {label}
        </span>
        <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${bgMap[color]}`}>
          {icon}
        </span>
      </div>
      <p className="mt-2 font-heading text-2xl font-bold tracking-tight text-foreground">
        {value}
      </p>
    </motion.div>
  );
}

/* ── Section heading ─────────────────────────────────── */
function SectionHeading({ title }: { title: string }) {
  return (
    <h3 className="mb-3 font-heading text-sm font-semibold uppercase tracking-widest text-muted">
      {title}
    </h3>
  );
}

/* ── Component ───────────────────────────────────────── */
export default function PillarTechnical({ metrics }: Props) {
  return (
    <div className="space-y-8">
      {/* Dribbling Control */}
      <div>
        <SectionHeading title="Dribbling Control" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Cone Slalom Agility"
            value={metrics.coneAgility}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="m22 8-6 4 6 4V8Z" />
                <path d="m2 8 6 4-6 4V8Z" />
                <path d="M8 8v8" />
                <path d="M16 8v8" />
              </svg>
            }
          />
          <MetricCard
            label="Change of Direction Speed"
            value={metrics.changeOfDirectionSpeed}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <path d="M18 8L22 12L18 16" />
                <path d="M2 12H22" />
                <path d="M2 12L6 8L2 12Z" />
              </svg>
            }
          />
          <MetricCard
            label="Touch Tightness Index"
            value={metrics.touchTightness}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="4" />
                <circle cx="12" cy="12" r="1" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Passing & First Touch */}
      <div>
        <SectionHeading title="Passing &amp; First Touch" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Wall Rebound Accuracy"
            value={metrics.wallReboundAccuracy}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            }
          />
          <MetricCard
            label="First Touch Control"
            value={metrics.firstTouchControlCushion}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <path d="M2 6c6 5 8 11 10 18" />
                <path d="M22 6c-6 5-8 11-10 18" />
              </svg>
            }
          />
          <MetricCard
            label="Weak Foot Ratio"
            value={metrics.weakFootRatio}
            color="warning"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-400">
                <path d="M3 7v6a4 4 0 0 0 4 4h10a4 4 0 0 0 4-4V7" />
                <path d="M3 7V5" />
                <path d="M3 7h18" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Open-Play Shooting */}
      <div>
        <SectionHeading title="Open-Play Shooting" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Shot Velocity"
            value={metrics.shotVelocity}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
              </svg>
            }
          />
          <MetricCard
            label="Launch Angle"
            value={metrics.launchAngleElevation}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <line x1="12" y1="20" x2="12" y2="4" />
                <polyline points="6 10 12 4 18 10" />
              </svg>
            }
          />
          <MetricCard
            label="Apex Target Accuracy"
            value={metrics.apexTargetAccuracy}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="4" />
              </svg>
            }
          />
        </div>
      </div>
    </div>
  );
}
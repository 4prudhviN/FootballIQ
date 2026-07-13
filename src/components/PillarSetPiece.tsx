import type { SetPieceMetrics } from "../types";
import { motion } from "framer-motion";

interface Props {
  metrics: SetPieceMetrics;
}

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
      className="rounded-xl border border-border bg-surface p-4 transition-all duration-200 hover:border-primary/40"
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

function SectionHeading({ title }: { title: string }) {
  return (
    <h3 className="mb-3 font-heading text-sm font-semibold uppercase tracking-widest text-muted">
      {title}
    </h3>
  );
}

export default function PillarSetPiece({ metrics }: Props) {
  return (
    <div className="space-y-8">
      {/* Free Kick Analytics */}
      <div>
        <SectionHeading title="Free Kick Analytics" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Ball Dip &amp; Curve Rate"
            value={metrics.curveRate}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M3 17c4-2 8-4 14-4" />
                <path d="M3 7c4 2 8 4 14 4" />
                <path d="M17 3v14" />
                <path d="M21 3v14" />
              </svg>
            }
          />
          <MetricCard
            label="Wall Clearance Margin"
            value={metrics.wallClearance}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
            }
          />
          <MetricCard
            label="Target Corner Accuracy"
            value={metrics.targetCornerAccuracy}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <rect x="2" y="2" width="20" height="20" rx="2" />
                <circle cx="12" cy="12" r="3" />
                <circle cx="12" cy="12" r="1" fill="currentColor" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Penalty Kick Telemetry */}
      <div>
        <SectionHeading title="Penalty Kick Telemetry" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Penalty Striking Force"
            value={metrics.penaltyForce}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            }
          />
          <MetricCard
            label="Goalkeeper Deception"
            value={metrics.goalkeeperDeceptionIndex}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
              </svg>
            }
          />
          <MetricCard
            label="Gait Vector Consistency"
            value={metrics.gaitVectorConsistency}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Throw-In Mechanics */}
      <div>
        <SectionHeading title="Throw-In Mechanics" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Spine Extension Angle"
            value={metrics.throwInSpineFlexion}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M12 2a10 10 0 0 1 10 10" />
                <path d="M12 2a10 10 0 0 0-10 10" />
                <path d="M2 12h20" />
              </svg>
            }
          />
          <MetricCard
            label="Elbow Flexion Acceleration"
            value={metrics.elbowFlexionAcceleration}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
              </svg>
            }
          />
          <MetricCard
            label="Release Trajectory"
            value={metrics.releasePointTrajectory}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <path d="M12 19V5" />
                <path d="M5 12l7-7 7 7" />
              </svg>
            }
          />
        </div>
      </div>
    </div>
  );
}
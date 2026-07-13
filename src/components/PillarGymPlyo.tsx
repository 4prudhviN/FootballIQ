import type { GymPlyoMetrics } from "../types";
import { motion } from "framer-motion";

interface Props {
  metrics: GymPlyoMetrics;
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

export default function PillarGymPlyo({ metrics }: Props) {
  return (
    <div className="space-y-8">
      {/* Skeletal Mechanical Biometrics */}
      <div>
        <SectionHeading title="Skeletal Mechanical Biometrics" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Hip Extension Angle"
            value={metrics.hipExtensionAngle}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M3 6h18" />
                <path d="M3 12h18" />
                <path d="M3 18h12" />
              </svg>
            }
          />
          <MetricCard
            label="Knee Flexion Angle"
            value={metrics.kneeFlexionAngle}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <path d="M4 20L12 4" />
                <path d="M20 20L12 4" />
                <path d="M12 4v16" />
              </svg>
            }
          />
          <MetricCard
            label="L/R Force Balancing"
            value={metrics.leftRightForceBalancing}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <line x1="12" y1="2" x2="12" y2="22" />
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Gym Strength Indicators */}
      <div>
        <SectionHeading title="Gym Strength Indicators" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Bar Path Velocity"
            value={metrics.barVelocity}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M6 4h12" />
                <path d="M6 20h12" />
                <path d="M10 4v16" />
                <path d="M14 4v16" />
              </svg>
            }
          />
          <MetricCard
            label="Eccentric Control Ratio"
            value={metrics.eccentricControlRatio}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
                <polyline points="16 7 22 7 22 13" />
              </svg>
            }
          />
          <MetricCard
            label="Asymmetric Balancing"
            value={metrics.leftRightForceBalancing}
            color="warning"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-400">
                <line x1="12" y1="2" x2="12" y2="22" />
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Plyometrics & Explosiveness */}
      <div>
        <SectionHeading title="Plyometrics &amp; Explosiveness" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Vertical Jump Height"
            value={metrics.verticalJumpHeight}
            color="primary"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <line x1="12" y1="20" x2="12" y2="4" />
                <polyline points="6 10 12 4 18 10" />
              </svg>
            }
          />
          <MetricCard
            label="Ground Contact Time"
            value={metrics.groundContactTime}
            color="accent"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            }
          />
          <MetricCard
            label="Reactive Strength Index"
            value={metrics.reactiveStrengthIndex}
            color="info"
            icon={
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-sky-400">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            }
          />
        </div>
      </div>
    </div>
  );
}
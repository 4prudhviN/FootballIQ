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
  CircleDot,
  Shield,
  Zap,
  PersonStanding,
  Eye,
  Brain,
  BarChart2,
  MessageSquare,
  GraduationCap,
  CheckCircle2,
  ChevronRight,
} from "lucide-react";

import type { AnalysisResult, FootballAction } from "../types";

interface AnalysisTabProps {
  results: AnalysisResult[];
  onUploadMore: () => void;
}

// ---------------------------------------------------------------------------
// Phase definitions — the 5-step intelligence pipeline
// ---------------------------------------------------------------------------

interface Phase {
  id: string;
  step: number;
  label: string;
  icon: React.ReactNode;
  tagline: string;
  description: string;
}

const PHASES: Phase[] = [
  {
    id: "detect",
    step: 1,
    label: "Detect",
    icon: <Eye size={18} />,
    tagline: "What happened?",
    description: "Computer vision identifies the player, the ball, and body landmarks in every frame.",
  },
  {
    id: "understand",
    step: 2,
    label: "Understand",
    icon: <Brain size={18} />,
    tagline: "What was the player doing?",
    description: "Activity detection classifies the football action — passing, shooting, dribbling, or defending.",
  },
  {
    id: "analyze",
    step: 3,
    label: "Analyze",
    icon: <BarChart2 size={18} />,
    tagline: "How well was it done?",
    description: "The activity-specific analyzer measures precision, posture, speed, and biomechanical efficiency.",
  },
  {
    id: "explain",
    step: 4,
    label: "Explain",
    icon: <MessageSquare size={18} />,
    tagline: "What does it mean?",
    description: "The feedback engine translates raw metrics into plain-English observations tied to the player's skill level.",
  },
  {
    id: "teach",
    step: 5,
    label: "Teach",
    icon: <GraduationCap size={18} />,
    tagline: "What to do next?",
    description: "Targeted drills and coach tips are selected from the knowledge base to fix each weak area.",
  },
];

// ---------------------------------------------------------------------------
// Action config map
// ---------------------------------------------------------------------------

const ACTION_CONFIG: Record<FootballAction, { label: string; icon: React.ReactNode; subMetrics: string[] }> = {
  passing:     { label: "Passing",     icon: <Crosshair size={15} />,      subMetrics: ["Ball Control", "First Touch", "Pass Accuracy", "Weight of Pass"] },
  dribbling:   { label: "Dribbling",   icon: <CircleDot size={15} />,      subMetrics: ["Close Control", "Change of Direction", "Touch Tightness", "Speed with Ball"] },
  shooting:    { label: "Shooting",    icon: <Zap size={15} />,            subMetrics: ["Shot Velocity", "Launch Angle", "Target Accuracy", "Torso Alignment"] },
  goalkeeping: { label: "Goalkeeping", icon: <Shield size={15} />,         subMetrics: ["Reaction Time", "Diving Range", "Distribution", "Positioning"] },
  defending:   { label: "Defending",   icon: <Shield size={15} />,         subMetrics: ["Tackle Timing", "Positioning", "Interception", "Aerial Duels"] },
  movement:    { label: "Movement",    icon: <PersonStanding size={15} />, subMetrics: ["Gait Symmetry", "Stride Length", "Sprint Speed", "Agility"] },
};

const MOCK_METRICS: Record<FootballAction, Record<string, string>> = {
  passing:     { "Ball Control": "92%", "First Touch": "0.36 m/s²", "Pass Accuracy": "87%", "Weight of Pass": "Medium" },
  dribbling:   { "Close Control": "88%", "Change of Direction": "5.8 m/s", "Touch Tightness": "±2.4 cm", "Speed with Ball": "24 km/h" },
  shooting:    { "Shot Velocity": "88 km/h", "Launch Angle": "14°", "Target Accuracy": "81%", "Torso Alignment": "12°" },
  goalkeeping: { "Reaction Time": "0.28s", "Diving Range": "2.4m", "Distribution": "74%", "Positioning": "Good" },
  defending:   { "Tackle Timing": "Good", "Positioning": "88%", "Interception": "3", "Aerial Duels": "67%" },
  movement:    { "Gait Symmetry": "92%", "Stride Length": "1.24m", "Sprint Speed": "31.2 km/h", "Agility": "4.2s" },
};

// ---------------------------------------------------------------------------
// Phase 1 — DETECT
// ---------------------------------------------------------------------------

function PhaseDetect({ videoUrl, warnings }: { videoUrl: string | null; warnings: string[] }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);

  return (
    <div className="space-y-5">
      {/* Video player */}
      <div className="group relative overflow-hidden rounded-2xl border border-border bg-black/60">
        <div className="aspect-video w-full bg-gradient-to-br from-emerald-950/70 via-emerald-900/30 to-black relative flex items-center justify-center">
          {videoUrl ? (
            <>
              <video
                ref={videoRef}
                src={videoUrl}
                className="absolute inset-0 h-full w-full object-contain"
                playsInline
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
              />
              {!playing && (
                <button
                  onClick={() => videoRef.current?.play()}
                  className="absolute inset-0 z-10 flex items-center justify-center bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 cursor-pointer"
                >
                  <span className="flex h-14 w-14 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm">
                    <Play size={24} className="ml-0.5 text-white" fill="white" />
                  </span>
                </button>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 text-white/30">
              <Play size={32} strokeWidth={1.5} />
              <p className="text-xs font-medium">No video recorded</p>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between border-t border-border bg-surface/90 px-4 py-3">
          <div>
            <p className="text-sm font-semibold text-foreground">Pose Estimation Feed</p>
            <p className="text-xs text-muted">MediaPipe skeleton overlay active</p>
          </div>
          <span className="flex h-2 w-2 rounded-full bg-success animate-pulse" />
        </div>
      </div>

      {/* Detection summary */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Player", value: "Detected", ok: true },
          { label: "Ball", value: "Detected", ok: true },
          { label: "Pose", value: `${33 - warnings.length * 3} landmarks`, ok: true },
        ].map((item) => (
          <div key={item.label} className="rounded-xl border border-border bg-surface p-4 text-center">
            <CheckCircle2 size={16} className="mx-auto mb-2 text-success" />
            <p className="text-xs text-muted">{item.label}</p>
            <p className="mt-0.5 text-sm font-semibold text-foreground">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Phase 2 — UNDERSTAND
// ---------------------------------------------------------------------------

function PhaseUnderstand({ detectedActions }: { detectedActions: FootballAction[] }) {
  if (detectedActions.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 py-10 text-center text-muted">
        <Brain size={32} strokeWidth={1.5} />
        <p className="text-sm">No activities detected yet. Upload a video to begin.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        The activity detector identified the following football actions from your video:
      </p>
      <div className="space-y-3">
        {detectedActions.map((action, i) => {
          const cfg = ACTION_CONFIG[action];
          return (
            <motion.div
              key={action}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className="flex items-center gap-4 rounded-xl border border-border bg-surface p-4"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary flex-shrink-0">
                {cfg.icon}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-foreground">{cfg.label}</p>
                <p className="text-xs text-muted mt-0.5">
                  {cfg.subMetrics.slice(0, 2).join(" · ")} will be measured
                </p>
              </div>
              <span className="text-[11px] font-semibold rounded-full bg-success/10 text-success px-2.5 py-0.5">
                Detected
              </span>
            </motion.div>
          );
        })}
      </div>
      <p className="text-xs text-muted/60 italic">
        Only detected actions are analysed — nothing is assumed.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Phase 3 — ANALYZE
// ---------------------------------------------------------------------------

function PhaseAnalyze({ detectedActions }: { detectedActions: FootballAction[] }) {
  const [activeAction, setActiveAction] = useState<FootballAction>(
    detectedActions[0] ?? "passing",
  );

  const actions = detectedActions.length > 0 ? detectedActions : (["passing"] as FootballAction[]);

  return (
    <div className="space-y-4">
      {/* Action tabs */}
      {actions.length > 1 && (
        <div className="flex gap-1 overflow-x-auto rounded-xl border border-border bg-background/60 p-1 scrollbar-none">
          {actions.map((a) => {
            const cfg = ACTION_CONFIG[a];
            const isActive = a === activeAction;
            return (
              <button
                key={a}
                onClick={() => setActiveAction(a)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-semibold whitespace-nowrap transition-all duration-150 cursor-pointer ${
                  isActive ? "bg-primary/15 text-primary" : "text-muted hover:text-foreground"
                }`}
              >
                <span className={isActive ? "text-primary" : "text-muted"}>{cfg.icon}</span>
                {cfg.label}
              </button>
            );
          })}
        </div>
      )}

      {/* Metric grid */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeAction}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
        >
          <p className="mb-3 text-xs text-muted">
            Measured by <span className="text-foreground font-medium">{activeAction}_analyzer.py</span>
          </p>
          <div className="grid grid-cols-2 gap-3">
            {ACTION_CONFIG[activeAction].subMetrics.map((name) => (
              <div key={name} className="rounded-xl border border-border bg-background/60 p-4">
                <p className="text-[11px] font-medium uppercase tracking-wider text-muted">{name}</p>
                <p className="mt-1.5 text-xl font-bold text-foreground">
                  {MOCK_METRICS[activeAction][name] ?? "—"}
                </p>
              </div>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Phase 4 — EXPLAIN
// ---------------------------------------------------------------------------

function PhaseExplain({ warnings, playerLevel }: { warnings: string[]; playerLevel: string }) {
  const explanations: { warning: string; plain: string; severity: "high" | "medium" | "low" }[] = [
    {
      warning: "POOR POSTURE / LEANING BACK",
      plain: "Your upper body was leaning behind your hips at the moment of contact. This sends the ball upward instead of forward and reduces accuracy.",
      severity: "high",
    },
    {
      warning: "KNEE ALIGNMENT RISK",
      plain: "Your knee was collapsing inward on your plant foot. This reduces power transfer and increases injury risk over time.",
      severity: "medium",
    },
    {
      warning: "ASYMMETRIC GAIT DETECTED",
      plain: "Your left and right strides are uneven. One leg is working harder than the other, which causes faster fatigue.",
      severity: "low",
    },
  ];

  const active = warnings.length > 0
    ? explanations.filter((e) => warnings.includes(e.warning))
    : explanations.slice(0, 2); // demo fallback

  const colorMap = {
    high:   "border-destructive/30 bg-destructive/5",
    medium: "border-accent/30 bg-accent/5",
    low:    "border-success/30 bg-success/5",
  };
  const badgeMap = {
    high:   "bg-destructive/15 text-destructive",
    medium: "bg-accent/15 text-accent-foreground",
    low:    "bg-success/15 text-success",
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted">
          Feedback for <span className="font-semibold text-foreground">{playerLevel}</span> player
        </p>
        <span className="text-xs text-muted">{active.length} issue{active.length !== 1 ? "s" : ""}</span>
      </div>

      {active.map((item, i) => (
        <motion.div
          key={item.warning}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1 }}
          className={`rounded-xl border p-4 ${colorMap[item.severity]}`}
        >
          <div className="flex items-start gap-3">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0 text-muted" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1.5">
                <span className={`text-[11px] font-semibold rounded-full px-2 py-0.5 ${badgeMap[item.severity]}`}>
                  {item.severity === "high" ? "High priority" : item.severity === "medium" ? "Medium" : "Note"}
                </span>
              </div>
              <p className="text-sm text-foreground leading-relaxed">{item.plain}</p>
            </div>
          </div>
        </motion.div>
      ))}

      {active.length === 0 && (
        <div className="flex flex-col items-center gap-3 py-8 text-center">
          <CheckCircle2 size={28} className="text-success" />
          <p className="text-sm text-muted">No issues detected — great session!</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Phase 5 — TEACH
// ---------------------------------------------------------------------------

interface Drill {
  name: string;
  metric: string;
  instructions: string;
  coachTip: string;
  duration: string;
}

function PhaseTeach({ warnings, playerLevel }: { warnings: string[]; playerLevel: string }) {
  const allDrills: (Drill & { forWarning: string })[] = [
    {
      forWarning: "POOR POSTURE / LEANING BACK",
      name: "Wall Lean Drill",
      metric: "Torso Alignment",
      instructions: "Stand 30 cm from a wall and practise driving your knee up without your back touching the wall. 3 × 15 reps each leg.",
      coachTip: "At contact, your chin should be down and chest forward over the ball.",
      duration: "10 min",
    },
    {
      forWarning: "KNEE ALIGNMENT RISK",
      name: "Lateral Band Walk",
      metric: "Knee Stability",
      instructions: "Place a resistance band above your knees and take 20 side steps each direction. 3 sets daily.",
      coachTip: "Push your knee outward over your second toe on every step.",
      duration: "8 min",
    },
    {
      forWarning: "ASYMMETRIC GAIT DETECTED",
      name: "Unilateral Bounds",
      metric: "Gait Symmetry",
      instructions: "Bound 20 m on one leg, then switch. 5 sets each side to equalise strength and stride length.",
      coachTip: "Count your steps on each side — aim for equal rhythm and equal power.",
      duration: "12 min",
    },
  ];

  const drills = warnings.length > 0
    ? allDrills.filter((d) => warnings.includes(d.forWarning))
    : allDrills.slice(0, 2);

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        Training drills selected for a{" "}
        <span className="font-semibold text-foreground">{playerLevel}</span> player:
      </p>

      {drills.map((drill, i) => (
        <motion.div
          key={drill.name}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1 }}
          className="rounded-xl border border-border bg-surface p-5"
        >
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-accent/10">
              <Lightbulb size={18} className="text-accent" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <p className="text-sm font-semibold text-foreground">{drill.name}</p>
                <span className="text-[11px] font-medium text-muted bg-surface-hover rounded-full px-2.5 py-0.5">
                  {drill.duration}
                </span>
              </div>
              <p className="text-[11px] font-medium text-primary mt-0.5">{drill.metric}</p>
              <p className="mt-2 text-sm text-muted leading-relaxed">{drill.instructions}</p>
              <div className="mt-3 flex items-start gap-2 rounded-lg bg-primary/5 border border-primary/15 p-3">
                <Target size={13} className="mt-0.5 flex-shrink-0 text-primary" />
                <p className="text-xs text-primary/80 leading-relaxed">
                  <span className="font-semibold">Coach tip:</span> {drill.coachTip}
                </p>
              </div>
            </div>
          </div>
          <div className="mt-4 flex justify-end">
            <button className="inline-flex items-center gap-1.5 rounded-xl bg-accent px-4 py-2 text-xs font-semibold text-black transition-all hover:brightness-110 active:scale-[0.97] cursor-pointer">
              Start drill <ArrowRight size={13} />
            </button>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Phase stepper header
// ---------------------------------------------------------------------------

function PhaseStepper({ active, onSelect }: { active: number; onSelect: (i: number) => void }) {
  return (
    <div className="flex items-center overflow-x-auto gap-0 rounded-2xl border border-border bg-surface p-1.5 scrollbar-none">
      {PHASES.map((phase, i) => {
        const isActive = i === active;
        const isDone   = i < active;
        return (
          <button
            key={phase.id}
            onClick={() => onSelect(i)}
            className={`flex items-center gap-2 rounded-xl px-3 py-2.5 text-xs font-semibold whitespace-nowrap transition-all duration-200 cursor-pointer flex-shrink-0 ${
              isActive
                ? "bg-primary/15 text-primary shadow-sm"
                : isDone
                  ? "text-success hover:bg-surface-hover"
                  : "text-muted hover:text-foreground hover:bg-surface-hover"
            }`}
          >
            <span className={isActive ? "text-primary" : isDone ? "text-success" : "text-muted"}>
              {isDone ? <CheckCircle2 size={15} /> : phase.icon}
            </span>
            <span className="hidden sm:inline">{phase.label}</span>
            <span className="sm:hidden">{phase.step}</span>
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export default function AnalysisTab({ results, onUploadMore }: AnalysisTabProps) {
  const latest      = results.find((r) => r.status === "completed") ?? null;
  const videoUrl    = latest?.videoUrl ?? null;
  const fileName    = latest?.fileName ?? null;
  const warnings    = latest?.warnings ?? [];
  const detectedActions: FootballAction[] = latest?.detectedActions ?? [];
  const playerLevel = "Intermediate"; // from skill_classifier — wire up when backend is live

  const [activePhase, setActivePhase] = useState(0);
  const phase = PHASES[activePhase];

  const goNext = () => setActivePhase((p) => Math.min(p + 1, PHASES.length - 1));
  const goPrev = () => setActivePhase((p) => Math.max(p - 1, 0));

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6 sm:space-y-8">

      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">Analysis</h1>
          <p className="mt-1 text-sm text-muted">
            {fileName ? `Results for "${fileName}"` : "Upload a video to begin your analysis."}
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

      {/* ── Phase stepper ── */}
      <PhaseStepper active={activePhase} onSelect={setActivePhase} />

      {/* ── Active phase card ── */}
      <div className="rounded-2xl border border-border bg-surface p-5 sm:p-6">
        {/* Phase header */}
        <div className="mb-5 flex items-start gap-4">
          <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
            {phase.icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                Step {phase.step} of {PHASES.length}
              </span>
            </div>
            <h2 className="font-heading text-lg font-bold text-foreground">{phase.label}</h2>
            <p className="text-sm text-primary font-medium">{phase.tagline}</p>
            <p className="mt-0.5 text-xs text-muted">{phase.description}</p>
          </div>
        </div>

        {/* Phase content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={phase.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22 }}
          >
            {activePhase === 0 && <PhaseDetect videoUrl={videoUrl} warnings={warnings} />}
            {activePhase === 1 && <PhaseUnderstand detectedActions={detectedActions} />}
            {activePhase === 2 && <PhaseAnalyze detectedActions={detectedActions} />}
            {activePhase === 3 && <PhaseExplain warnings={warnings} playerLevel={playerLevel} />}
            {activePhase === 4 && <PhaseTeach warnings={warnings} playerLevel={playerLevel} />}
          </motion.div>
        </AnimatePresence>

        {/* Phase navigation */}
        <div className="mt-6 flex items-center justify-between border-t border-border/50 pt-5">
          <button
            onClick={goPrev}
            disabled={activePhase === 0}
            className="inline-flex items-center gap-1.5 rounded-xl border border-border px-4 py-2 text-sm font-medium text-muted hover:text-foreground hover:bg-surface-hover transition-all disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
          >
            ← Previous
          </button>

          <div className="flex gap-1.5">
            {PHASES.map((_, i) => (
              <button
                key={i}
                onClick={() => setActivePhase(i)}
                className={`h-1.5 rounded-full transition-all duration-200 cursor-pointer ${
                  i === activePhase ? "w-6 bg-primary" : "w-1.5 bg-border hover:bg-muted"
                }`}
              />
            ))}
          </div>

          <button
            onClick={goNext}
            disabled={activePhase === PHASES.length - 1}
            className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-on-primary transition-all hover:bg-primary-dark active:scale-[0.97] disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}

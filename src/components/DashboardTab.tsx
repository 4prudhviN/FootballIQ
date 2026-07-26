import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Video, Target, TrendingUp, Upload, Clock, CheckCircle,
  ChevronRight, Trophy, Sparkles, Lightbulb, Dumbbell,
  AlertCircle, Star,
} from "lucide-react";
import type { AnalysisResult } from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function formatDate(d: Date) {
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

function levelColor(level: string) {
  switch (level) {
    case "Elite":        return "bg-yellow-500/15 text-yellow-400";
    case "Advanced":     return "bg-success/15 text-success";
    case "Intermediate": return "bg-accent/15 text-accent";
    case "Developing":   return "bg-orange-500/15 text-orange-400";
    default:             return "bg-muted/20 text-muted";
  }
}

function activityIcon(activity: string): string {
  const icons: Record<string, string> = {
    passing:     "⚽", dribbling: "🏃", shooting:    "🎯",
    goalkeeping: "🧤", defending: "🛡️", movement:    "💨",
    free_kick:   "🦵", penalty:   "⚽", header:      "🤸",
  };
  return icons[activity] ?? "⚽";
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DashboardTabProps {
  analysisResults: AnalysisResult[];
  onUploadClick:   () => void;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Renders one detected activity card with its metrics — fully dynamic */
function ActivityCard({ activity, metrics }: {
  activity: string;
  metrics:  Record<string, string>;
}) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) return null;

  return (
    <div className="rounded-xl border border-border bg-background/60 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-base">{activityIcon(activity)}</span>
        <h4 className="text-sm font-semibold text-foreground capitalize">
          {activity.replace("_", " ")}
        </h4>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {entries.map(([label, value]) => (
          <div key={label} className="space-y-0.5">
            <p className="text-[10px] font-medium uppercase tracking-wider text-muted">
              {label}
            </p>
            <p className="text-sm font-bold text-foreground">{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/** "Focus this week" section — renders whatever the pipeline returns */
function FocusThisWeek({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-3">
        <Target size={16} className="text-primary" />
        <h3 className="font-heading text-sm font-semibold text-foreground">
          Focus this week
        </h3>
      </div>
      <div className="space-y-2">
        {items.map((item, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-primary/15 text-[10px] font-bold text-primary">
              {i + 1}
            </span>
            <span className="text-sm text-foreground">{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Coach tips section — renders whatever the pipeline returns */
function CoachTips({ tips }: { tips: string[] }) {
  if (!tips.length) return null;
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb size={16} className="text-accent" />
        <h3 className="font-heading text-sm font-semibold text-foreground">
          Coach tips
        </h3>
      </div>
      <div className="space-y-2.5">
        {tips.map((tip, i) => (
          <p key={i} className="text-sm text-muted leading-relaxed">
            <span className="font-semibold text-accent mr-1">•</span>
            {tip}
          </p>
        ))}
      </div>
    </div>
  );
}

/** Recommendations section — renders whatever the pipeline returns */
function Recommendations({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-3">
        <Dumbbell size={16} className="text-success" />
        <h3 className="font-heading text-sm font-semibold text-foreground">
          Recommendations
        </h3>
      </div>
      <div className="space-y-2">
        {items.map((rec, i) => (
          <p key={i} className="text-sm text-muted">{rec}</p>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------

export default function DashboardTab({ analysisResults, onUploadClick }: DashboardTabProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const completed = analysisResults.filter((r) => r.status === "completed");
  const lastWeek  = analysisResults.filter(
    (r) => r.date.getTime() > Date.now() - 7 * 24 * 60 * 60 * 1000
  ).length;

  // Latest completed session for the detail panel.
  const latest = completed[0] ?? null;
  const selected = analysisResults.find((r) => r.id === selectedId) ?? latest;

  // Stats — computed from actual session data, no hardcoding.
  const stats = [
    {
      label: "Sessions",
      value: String(completed.length),
      icon:  Video,
      color: "text-primary",
      bg:    "bg-primary/10",
    },
    {
      label: "This Week",
      value: String(lastWeek),
      icon:  TrendingUp,
      color: "text-success",
      bg:    "bg-success/10",
    },
    {
      label: "Level",
      value: latest?.playerLevel ?? "—",
      icon:  Star,
      color: "text-accent",
      bg:    "bg-accent/10",
    },
    {
      label: "Activities",
      value: latest?.detectedActions?.length
        ? String(latest.detectedActions.length)
        : "—",
      icon:  Trophy,
      color: "text-accent",
      bg:    "bg-accent/10",
    },
  ] as const;

  const hasResults = analysisResults.length > 0;

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">

      {/* Welcome */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/10">
          <Sparkles size={20} className="text-accent" strokeWidth={2} />
        </div>
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">
            Welcome back
          </h1>
          <p className="mt-0.5 text-sm text-muted">
            Your training intelligence overview.
          </p>
        </div>
      </div>

      {/* Stats grid — dynamic */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.label}
              className="rounded-2xl border border-border bg-surface p-4 sm:p-5 transition-all duration-200 hover:border-muted hover:bg-surface-hover">
              <div className={`mb-3 inline-flex rounded-xl p-2.5 ${stat.bg}`}>
                <Icon size={20} className={stat.color} strokeWidth={2.2} />
              </div>
              <p className="text-2xl font-bold tracking-tight text-foreground">{stat.value}</p>
              <p className="mt-0.5 text-xs text-muted">{stat.label}</p>
            </div>
          );
        })}
      </div>

      {hasResults ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">

          {/* ── Session list ── */}
          <div className="lg:col-span-1 space-y-2.5">
            <h2 className="font-heading text-lg font-semibold text-foreground mb-4">
              Recent Sessions
            </h2>
            {analysisResults.map((result) => {
              const isComplete = result.status === "completed";
              const isFailed   = result.status === "failed";
              const isSelected = result.id === (selectedId ?? latest?.id);

              return (
                <button
                  key={result.id}
                  onClick={() => setSelectedId(result.id)}
                  className={`w-full flex items-center gap-3 rounded-2xl border p-4 text-left transition-all duration-200 cursor-pointer group ${
                    isSelected
                      ? "border-primary/40 bg-primary/5"
                      : "border-border bg-surface hover:border-muted hover:bg-surface-hover"
                  }`}
                >
                  <div className={`flex-shrink-0 rounded-xl p-2 ${
                    isComplete ? "bg-success/10 text-success"
                    : isFailed  ? "bg-destructive/10 text-destructive"
                    : "bg-accent/10 text-accent"
                  }`}>
                    {isComplete ? <CheckCircle size={18} strokeWidth={2} />
                     : isFailed  ? <AlertCircle size={18} strokeWidth={2} />
                     : <Clock     size={18} strokeWidth={2} />}
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="truncate text-sm font-medium text-foreground">
                      {result.fileName}
                    </p>
                    <p className="text-xs text-muted mt-0.5">{formatDate(result.date)}</p>
                    {/* Detected activities — dynamic pills */}
                    {result.detectedActions && result.detectedActions.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {result.detectedActions.slice(0, 3).map((act) => (
                          <span key={act}
                            className="inline-flex items-center gap-0.5 rounded-full bg-primary/10 px-1.5 py-0.5 text-[9px] font-semibold text-primary">
                            {activityIcon(act)} {act}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {result.playerLevel && (
                    <span className={`flex-shrink-0 text-[10px] font-bold rounded-full px-2 py-0.5 ${levelColor(result.playerLevel)}`}>
                      {result.playerLevel}
                    </span>
                  )}

                  <ChevronRight size={14} className="flex-shrink-0 text-muted group-hover:translate-x-0.5 transition-transform" />
                </button>
              );
            })}
          </div>

          {/* ── Session detail — fully dynamic, renders whatever pipeline returns ── */}
          <AnimatePresence mode="wait">
            {selected && (
              <motion.div
                key={selected.id}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.2 }}
                className="lg:col-span-2 space-y-4"
              >
                {/* Session header */}
                <div className="rounded-2xl border border-border bg-surface p-5">
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div>
                      <h3 className="font-heading text-base font-semibold text-foreground">
                        {selected.fileName}
                      </h3>
                      <p className="text-xs text-muted mt-0.5">{formatDate(selected.date)}</p>
                    </div>
                    {selected.playerLevel && (
                      <span className={`text-xs font-bold rounded-full px-3 py-1 ${levelColor(selected.playerLevel)}`}>
                        {selected.playerLevel}
                      </span>
                    )}
                  </div>

                  {/* AI summary — LLM natural language rewrite */}
                  {selected.aiFeedback?.summary && (
                    <p className="mt-3 text-sm text-muted leading-relaxed">
                      {selected.aiFeedback.summary}
                    </p>
                  )}

                  {/* Activity analysis */}
                  {selected.aiFeedback?.activityAnalysis && (
                    <p className="mt-2 text-sm text-muted/80 leading-relaxed italic">
                      {selected.aiFeedback.activityAnalysis}
                    </p>
                  )}

                  {/* Detected activities — rendered dynamically */}
                  {selected.detectedActions && selected.detectedActions.length > 0 && (
                    <div className="mt-4">
                      <p className="text-[11px] font-medium uppercase tracking-wider text-muted mb-2">
                        Detected Activities
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {selected.detectedActions.map((act) => (
                          <span key={act}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-background/60 px-3 py-1.5 text-xs font-semibold text-foreground">
                            {activityIcon(act)}
                            <span className="capitalize">{act.replace("_", " ")}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Per-activity metric cards — dynamic: only renders what pipeline returned */}
                {selected.metrics?.byAction &&
                  Object.keys(selected.metrics.byAction).length > 0 && (
                  <div>
                    <p className="text-[11px] font-medium uppercase tracking-wider text-muted mb-2 px-1">
                      Metrics
                    </p>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      {Object.entries(selected.metrics.byAction).map(([activity, metrics]) => (
                        <ActivityCard
                          key={activity}
                          activity={activity}
                          metrics={metrics as Record<string, string>}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Coach explanation */}
                {selected.aiFeedback?.coachExplanation && (
                  <div className="rounded-2xl border border-border bg-surface p-5">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-base">🗣️</span>
                      <h3 className="font-heading text-sm font-semibold text-foreground">Coach Explanation</h3>
                    </div>
                    <p className="text-sm text-muted leading-relaxed">{selected.aiFeedback.coachExplanation}</p>
                  </div>
                )}

                {/* Focus this week — dynamic */}
                {selected.focusThisWeek && selected.focusThisWeek.length > 0 && (
                  <FocusThisWeek items={selected.focusThisWeek} />
                )}

                {/* Recommendations — dynamic */}
                {selected.aiFeedback?.weaknesses && selected.aiFeedback.weaknesses.length > 0 && (
                  <Recommendations items={selected.aiFeedback.weaknesses} />
                )}

                {/* Coach tips — dynamic */}
                {selected.aiFeedback?.coachingTips && selected.aiFeedback.coachingTips.length > 0 && (
                  <CoachTips tips={selected.aiFeedback.coachingTips} />
                )}

                {/* Motivational tip */}
                {selected.aiFeedback?.motivationalTip && (
                  <div className="rounded-2xl border border-accent/30 bg-accent/5 p-4">
                    <p className="text-sm text-accent/90 italic">
                      💬 {selected.aiFeedback.motivationalTip}
                    </p>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ) : (
        /* Empty state */
        <div className="flex flex-col items-center rounded-2xl border border-dashed border-border bg-surface/50 px-6 py-14 text-center">
          <Video size={40} className="mb-4 text-muted" strokeWidth={1.5} />
          <h3 className="font-heading text-lg font-semibold text-foreground">No sessions yet</h3>
          <p className="mt-1 max-w-xs text-sm text-muted">
            Upload your first training video to start getting AI-powered feedback.
          </p>
          <button
            onClick={onUploadClick}
            className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-on-primary transition-all duration-200 hover:bg-primary-dark active:scale-[0.97] cursor-pointer"
          >
            <Upload size={16} />
            Upload a video
          </button>
        </div>
      )}
    </div>
  );
}

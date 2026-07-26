import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RefreshCw, Play, Target, Lightbulb, ArrowRight,
  CheckCircle2, AlertTriangle, Crosshair, CircleDot,
  Zap, Shield, PersonStanding, CalendarDays, Star,
  MessageSquare, Dumbbell,
} from "lucide-react";
import type { AnalysisResult, FootballAction } from "../types";

interface AnalysisTabProps {
  results:       AnalysisResult[];
  onUploadMore:  () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function activityIcon(a: string) {
  const map: Record<string, React.ReactNode> = {
    passing:     <Crosshair    size={15} />,
    dribbling:   <CircleDot    size={15} />,
    shooting:    <Zap          size={15} />,
    goalkeeping: <Shield       size={15} />,
    defending:   <Shield       size={15} />,
    movement:    <PersonStanding size={15} />,
  };
  return map[a] ?? <Crosshair size={15} />;
}

function levelColor(l: string) {
  const m: Record<string, string> = {
    Elite:        "text-yellow-400 bg-yellow-400/10",
    Advanced:     "text-success   bg-success/10",
    Intermediate: "text-accent    bg-accent/10",
    Developing:   "text-orange-400 bg-orange-400/10",
    Beginner:     "text-muted     bg-muted/10",
  };
  return m[l] ?? "text-muted bg-muted/10";
}

const LEVEL_STARS: Record<string, string> = {
  Elite: "⭐⭐⭐⭐⭐", Advanced: "⭐⭐⭐⭐",
  Intermediate: "⭐⭐⭐", Developing: "⭐⭐", Beginner: "⭐",
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** One activity metric card — renders whatever metrics exist */
function ActivityCard({ activity, metrics }: {
  activity: string;
  metrics:  Record<string, string>;
}) {
  const entries = Object.entries(metrics);
  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-primary">{activityIcon(activity)}</span>
        <h3 className="font-heading text-sm font-semibold text-foreground capitalize">
          {activity.replace("_", " ")} Analysis
        </h3>
      </div>
      {entries.length > 0 ? (
        <div className="grid grid-cols-2 gap-3">
          {entries.map(([label, value]) => (
            <div key={label} className="rounded-xl border border-border bg-background/60 p-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-muted">{label.replace(/_/g, " ")}</p>
              <p className="mt-1 text-lg font-bold text-foreground">{value}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted">No metrics available for this activity.</p>
      )}
    </div>
  );
}

/** Coach Report — 7 LLM sections */
function CoachReport({ feedback }: { feedback: NonNullable<AnalysisResult["aiFeedback"]> }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-5 space-y-5">
      <div className="flex items-center gap-2">
        <MessageSquare size={16} className="text-accent" />
        <h3 className="font-heading text-sm font-semibold text-foreground">Coach Report</h3>
      </div>

      {/* Session Summary */}
      {feedback.summary && (
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted mb-1">Session Summary</p>
          <p className="text-sm text-foreground leading-relaxed">{feedback.summary}</p>
        </div>
      )}

      {/* Activity Analysis */}
      {feedback.activityAnalysis && (
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-muted mb-1">Activity Analysis</p>
          <p className="text-sm text-muted leading-relaxed">{feedback.activityAnalysis}</p>
        </div>
      )}

      {/* Strengths */}
      {feedback.strengths && feedback.strengths.length > 0 && (
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-success mb-2">Strengths</p>
          <div className="space-y-1.5">
            {feedback.strengths.map((s, i) => (
              <div key={i} className="flex items-start gap-2">
                <CheckCircle2 size={13} className="mt-0.5 flex-shrink-0 text-success" />
                <p className="text-sm text-foreground">{s}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Areas to Improve */}
      {feedback.weaknesses && feedback.weaknesses.length > 0 && (
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-destructive mb-2">Areas to Improve</p>
          <div className="space-y-1.5">
            {feedback.weaknesses.map((w, i) => (
              <div key={i} className="flex items-start gap-2">
                <AlertTriangle size={13} className="mt-0.5 flex-shrink-0 text-destructive" />
                <p className="text-sm text-foreground">{w}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Coach Explanation */}
      {feedback.coachExplanation && (
        <div className="rounded-xl border border-accent/20 bg-accent/5 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-accent mb-1">Coach Explanation</p>
          <p className="text-sm text-accent/90 leading-relaxed">{feedback.coachExplanation}</p>
        </div>
      )}
    </div>
  );
}

/** Training Plan — drills from the recommendation engine */
function TrainingPlan({ drills, tips }: {
  drills: NonNullable<AnalysisResult["aiFeedback"]>["coachingTips"];
  tips:   NonNullable<AnalysisResult["aiFeedback"]>["coachingTips"];
}) {
  const items = drills && drills.length > 0 ? drills : tips;
  if (!items || items.length === 0) return null;

  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-4">
        <Dumbbell size={16} className="text-primary" />
        <h3 className="font-heading text-sm font-semibold text-foreground">Training Recommendations</h3>
      </div>
      <div className="space-y-3">
        {items.map((item, i) => (
          <div key={i} className="flex items-start gap-3 rounded-xl border border-border bg-background/60 p-4">
            <div className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-bold text-primary">
              {i + 1}
            </div>
            <p className="text-sm text-foreground leading-relaxed">{item}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Next Session Goals — next_focus + motivationalTip */
function NextSessionGoals({ nextFocus, motivationalTip, focusAreas }: {
  nextFocus:       string | undefined;
  motivationalTip: string | undefined;
  focusAreas:      string[] | undefined;
}) {
  const hasGoals = nextFocus || (focusAreas && focusAreas.length > 0);
  if (!hasGoals && !motivationalTip) return null;

  return (
    <div className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-4">
        <CalendarDays size={16} className="text-accent" />
        <h3 className="font-heading text-sm font-semibold text-foreground">Next Session Goals</h3>
      </div>

      {focusAreas && focusAreas.length > 0 && (
        <div className="mb-4 space-y-2">
          {focusAreas.map((item, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-primary/15 text-[10px] font-bold text-primary">
                {i + 1}
              </span>
              <span className="text-sm text-foreground">{item}</span>
            </div>
          ))}
        </div>
      )}

      {nextFocus && (
        <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 mb-3">
          <div className="flex items-center gap-2 mb-1">
            <Target size={13} className="text-primary" />
            <p className="text-[11px] font-semibold uppercase tracking-wider text-primary">Priority Focus</p>
          </div>
          <p className="text-sm text-foreground">{nextFocus}</p>
        </div>
      )}

      {motivationalTip && (
        <p className="text-sm text-accent/90 italic">💬 {motivationalTip}</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export default function AnalysisTab({ results, onUploadMore }: AnalysisTabProps) {
  const latest = results.find((r) => r.status === "completed") ?? null;

  if (!latest) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Play size={40} className="mb-4 text-muted" strokeWidth={1.5} />
        <h2 className="font-heading text-lg font-semibold text-foreground">No analysis yet</h2>
        <p className="mt-1 text-sm text-muted max-w-xs">
          Upload a training video to get your personalised coaching report.
        </p>
        <button
          onClick={onUploadMore}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-on-primary transition-all hover:bg-primary-dark active:scale-[0.97] cursor-pointer"
        >
          Upload Video
        </button>
      </div>
    );
  }

  const activities  = latest.detectedActions   ?? [];
  const byAction    = latest.metrics?.byAction ?? {};
  const feedback    = latest.aiFeedback;
  const level       = latest.playerLevel ?? "Beginner";
  const stars       = LEVEL_STARS[level] ?? "⭐";
  const focusAreas  = latest.focusThisWeek ?? [];

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">Analysis</h1>
          <p className="mt-1 text-sm text-muted">
            Results for &ldquo;{latest.fileName}&rdquo;
          </p>
        </div>
        <div className="flex items-center gap-3">
          {level && (
            <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-bold ${levelColor(level)}`}>
              {stars} {level}
            </span>
          )}
          <button
            onClick={onUploadMore}
            className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2 text-sm font-medium text-muted hover:text-foreground hover:bg-surface-hover transition-all duration-200 cursor-pointer"
          >
            <RefreshCw size={14} /> Upload New
          </button>
        </div>
      </div>

      {/* Detected Activities — dynamic pills */}
      {activities.length > 0 && (
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted mb-2">
            Detected Activities
          </p>
          <div className="flex flex-wrap gap-2">
            {activities.map((act) => (
              <span key={act}
                className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-foreground">
                {activityIcon(act)}
                <span className="capitalize">{act.replace("_", " ")}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Activity Cards — one per detected activity, all dynamic */}
      {Object.keys(byAction).length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {Object.entries(byAction).map(([activity, metrics]) => (
            <ActivityCard
              key={activity}
              activity={activity}
              metrics={metrics as Record<string, string>}
            />
          ))}
        </div>
      )}

      {/* Coach Report — 7 LLM sections */}
      {feedback && (
        <CoachReport feedback={feedback} />
      )}

      {/* Training Plan */}
      {feedback?.coachingTips && (
        <TrainingPlan
          drills={feedback.coachingTips}
          tips={feedback.coachingTips}
        />
      )}

      {/* Next Session Goals */}
      <NextSessionGoals
        nextFocus       = {feedback?.nextFocus}
        motivationalTip = {feedback?.motivationalTip}
        focusAreas      = {focusAreas}
      />

    </div>
  );
}

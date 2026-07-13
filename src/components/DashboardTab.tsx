import {
  Video,
  Target,
  TrendingUp,
  Upload,
  Clock,
  CheckCircle,
  ChevronRight,
  Trophy,
  Sparkles,
} from "lucide-react";
import type { AnalysisResult } from "../types";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function formatDate(d: Date): string {
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

/* ── Progress badge color ──────────────────────────────── */
function badgeColor(progress: number, status: string) {
  if (status === "failed") return "bg-destructive/20 text-destructive";
  if (status === "completed" || progress === 100) return "bg-success/20 text-success";
  return "bg-accent/20 text-accent";
}

/* ── Props ──────────────────────────────────────────────── */
interface DashboardTabProps {
  analysisResults: AnalysisResult[];
  onUploadClick: () => void;
}

/* ── Component ─────────────────────────────────────────── */
export default function DashboardTab({
  analysisResults,
  onUploadClick,
}: DashboardTabProps) {
  const completed = analysisResults.filter((r) => r.status === "completed");
  const avgAccuracy = completed.length
    ? Math.round(
        completed.reduce((sum, r) => {
          const m = r.metrics;
          if (!m) return sum + 70;
          return sum + Math.round((m.torsoLean + m.kneeStability + m.gaitSymmetry) / 3);
        }, 0) / completed.length
      )
    : null;

  const bestRating = completed.length
    ? Math.max(
        ...completed.map((r) => {
          const m = r.metrics;
          if (!m) return 0;
          return Math.round(((m.torsoLean + m.kneeStability + m.gaitSymmetry) / 3) * 10) / 10;
        })
      )
    : null;

  const lastWeek = analysisResults.filter(
    (r) => r.date.getTime() > Date.now() - 7 * 24 * 60 * 60 * 1000
  ).length;

  const stats = [
    {
      label: "Videos Analyzed",
      value: String(completed.length),
      icon: Video,
      color: "text-primary",
      bg: "bg-primary/10",
    },
    {
      label: "Avg. Accuracy",
      value: avgAccuracy !== null ? `${avgAccuracy}%` : "—",
      icon: Target,
      color: "text-accent",
      bg: "bg-accent/10",
    },
    {
      label: "Best Rating",
      value: bestRating !== null ? String(bestRating) : "—",
      icon: Trophy,
      color: "text-accent",
      bg: "bg-accent/10",
    },
    {
      label: "This Week",
      value: String(lastWeek),
      icon: TrendingUp,
      color: "text-success",
      bg: "bg-success/10",
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
            Here&apos;s your training intelligence overview.
          </p>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="rounded-2xl border border-border bg-surface p-4 sm:p-5 transition-all duration-200 hover:border-muted hover:bg-surface-hover"
            >
              <div className={`mb-3 inline-flex rounded-xl p-2.5 ${stat.bg}`}>
                <Icon size={20} className={stat.color} strokeWidth={2.2} />
              </div>
              <p className="text-2xl font-bold tracking-tight text-foreground">
                {stat.value}
              </p>
              <p className="mt-0.5 text-xs text-muted">{stat.label}</p>
            </div>
          );
        })}
      </div>

      {/* Session history */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-heading text-lg font-semibold text-foreground">
            Recent Sessions
          </h2>
          {hasResults && (
            <button className="flex items-center gap-1 text-xs font-medium text-muted hover:text-foreground transition-colors duration-200 cursor-pointer">
              View all
              <ChevronRight size={14} />
            </button>
          )}
        </div>

        {hasResults ? (
          <div className="space-y-2.5">
            {analysisResults.map((result) => {
              const isComplete = result.status === "completed";
              const isProcessing = result.status === "processing";
              const isFailed = result.status === "failed";
              const avg = result.metrics
                ? Math.round(
                    (result.metrics.torsoLean +
                      result.metrics.kneeStability +
                      result.metrics.gaitSymmetry) /
                      3
                  )
                : null;

              return (
                <button
                  key={result.id}
                  className="w-full flex items-center gap-4 rounded-2xl border border-border bg-surface p-4 text-left transition-all duration-200 hover:border-muted hover:bg-surface-hover cursor-pointer group"
                >
                  {/* Status icon */}
                  <div
                    className={`flex-shrink-0 rounded-xl p-2.5 ${
                      isComplete
                        ? "bg-success/10 text-success"
                        : isFailed
                          ? "bg-destructive/10 text-destructive"
                          : "bg-accent/10 text-accent"
                    }`}
                  >
                    {isComplete ? (
                      <CheckCircle size={20} strokeWidth={2} />
                    ) : (
                      <Clock size={20} strokeWidth={2} />
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="truncate font-medium text-foreground group-hover:text-primary transition-colors duration-200">
                      {result.fileName}
                    </p>
                    <p className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
                      <Clock size={12} strokeWidth={2} />
                      {formatDate(result.date)}
                    </p>
                  </div>

                  {/* Accuracy / Progress badge */}
                  <div className="flex flex-col items-end gap-1">
                    <span
                      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${badgeColor(isComplete ? 100 : 0, result.status)}`}
                    >
                      {isComplete
                        ? avg !== null
                          ? `${avg}%`
                          : "Done"
                        : isFailed
                          ? "Failed"
                          : "Processing"}
                    </span>
                    <span className="text-[10px] uppercase tracking-wider text-muted">
                      {isComplete ? "Accuracy" : isFailed ? "Error" : "Status"}
                    </span>
                  </div>

                  {/* Chevron */}
                  <ChevronRight
                    size={16}
                    className="flex-shrink-0 text-muted transition-transform duration-200 group-hover:translate-x-0.5"
                  />
                </button>
              );
            })}
          </div>
        ) : (
          /* ── Empty state ─────────────────────────── */
          <div className="flex flex-col items-center rounded-2xl border border-dashed border-border bg-surface/50 px-6 py-14 text-center">
            <Video size={40} className="mb-4 text-muted" strokeWidth={1.5} />
            <h3 className="font-heading text-lg font-semibold text-foreground">
              No sessions yet
            </h3>
            <p className="mt-1 max-w-xs text-sm text-muted">
              Upload your first training video to start getting AI-powered
              feedback on your technique.
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
      </section>
    </div>
  );
}


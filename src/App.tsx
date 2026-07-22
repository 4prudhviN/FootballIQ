import { useState, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import type { NavTab, AnalysisResult, FootballAction } from "./types";
import Sidebar from "./components/Sidebar";
import DashboardTab from "./components/DashboardTab";
import UploadTab from "./components/UploadTab";
import AnalysisTab from "./components/AnalysisTab";
import SettingsTab from "./components/SettingsTab";
import AuthScreen from "./components/AuthScreen";

let nextId = 1;

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [playerName, setPlayerName] = useState("");
  const [activeTab, setActiveTab] = useState<NavTab>("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);

  /** Called by AuthScreen when a user signs in or creates a profile */
  const handleLogin = useCallback((name: string) => {
    setPlayerName(name);
    setIsAuthenticated(true);
  }, []);

  /** Called by UploadTab when a video finishes analysis */
  const handleAnalysisComplete = useCallback(
    (videoUrl: string, warnings: string[], fileName: string) => {
      // Derive detected actions from warning flags — in production this
      // would come from activity_detector.py via the backend response.
      const detectedActions: FootballAction[] = [];
      if (warnings.includes("POOR POSTURE / LEANING BACK")) detectedActions.push("shooting");
      if (warnings.includes("KNEE ALIGNMENT RISK")) detectedActions.push("dribbling");
      if (warnings.includes("ASYMMETRIC GAIT DETECTED")) detectedActions.push("movement");
      // Always include at least one action so the dashboard is never empty.
      if (detectedActions.length === 0) detectedActions.push("passing");

      const result: AnalysisResult = {
        id: String(nextId++),
        fileName,
        date: new Date(),
        status: "completed",
        videoUrl,
        warnings,
        metrics: {
          torsoLean: warnings.includes("POOR POSTURE / LEANING BACK") ? 22 : 12,
          kneeStability: warnings.includes("KNEE ALIGNMENT RISK") ? 72 : 87,
          gaitSymmetry: warnings.includes("ASYMMETRIC GAIT DETECTED") ? 78 : 92,
        },
        detectedActions,
      };
      setAnalysisResults((prev) => [result, ...prev]);
      setActiveTab("analysis");
    },
    [],
  );

  /* ── Not authenticated → show auth screen ────────────── */
  if (!isAuthenticated) {
    return <AuthScreen onLogin={handleLogin} />;
  }

  /* ── Authenticated → show main application ───────────── */
  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
      />

      {/* Main content area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Top header bar */}
        <header className="glass-panel-light flex h-16 shrink-0 items-center justify-between border-b border-border px-4 sm:px-6">
          <div className="flex items-center gap-3">
            {sidebarCollapsed && (
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                  <span className="text-base font-bold text-primary">IQ</span>
                </div>
                <span className="text-lg font-bold tracking-tight text-foreground">
                  Football<span className="text-primary">IQ</span>
                </span>
              </div>
            )}
            <div className="hidden sm:flex items-center gap-2 ml-2">
              <span className="text-xs font-medium text-muted uppercase tracking-widest">
                {activeTab === "dashboard"
                  ? "Command Center"
                  : activeTab === "upload"
                    ? "AI Analyzer"
                    : activeTab === "analysis"
                      ? "Biomechanical Analysis"
                      : "Settings & Preferences"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Notification bell */}
            <button
              type="button"
              className="relative flex h-9 w-9 items-center justify-center rounded-xl border border-border text-muted transition-all duration-200 hover:border-primary/30 hover:text-foreground hover:bg-surface-hover cursor-pointer"
              aria-label="Notifications"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
                <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
              </svg>
              <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-primary" />
            </button>

            {/* Profile */}
            <button
              type="button"
              className="flex h-9 items-center gap-2.5 rounded-xl border border-border px-3 transition-all duration-200 hover:border-primary/30 hover:bg-surface-hover cursor-pointer"
              aria-label="User profile"
            >
              <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-primary/15">
                <span className="text-xs font-bold text-primary">
                  {playerName.charAt(0).toUpperCase()}
                </span>
              </div>
              <span className="hidden sm:inline text-sm font-medium text-foreground">
                {playerName}
              </span>
              <span className="hidden md:inline-flex items-center rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-primary">
                Pro
              </span>
            </button>
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          <AnimatePresence mode="wait">
            {activeTab === "dashboard" && (
              <DashboardTab
                key="dashboard"
                analysisResults={analysisResults}
                onUploadClick={() => setActiveTab("upload")}
              />
            )}
            {activeTab === "upload" && (
              <UploadTab
                key="upload"
                onAnalysisComplete={handleAnalysisComplete}
              />
            )}
            {activeTab === "analysis" && (
              <AnalysisTab
                key="analysis"
                results={analysisResults}
                onUploadMore={() => setActiveTab("upload")}
              />
            )}
            {activeTab === "settings" && <SettingsTab key="settings" />}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
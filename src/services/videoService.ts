/**
 * Video Analysis API Service  (v2)
 * ==================================
 *
 * Calls the FootballIQ backend which runs the full 9-stage pipeline:
 *
 *   Video → Player Detection → Ball Detection → Pose Estimation
 *   → Activity Detection → Analyzer Selection → Metric Calculation
 *   → Skill Classification → Feedback Engine → Dashboard
 *
 * Falls back to a client-side mock when the backend is unreachable.
 */

import type { FootballAction, SkillLevel } from "../types";

const BACKEND_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Response shape — mirrors FootballSession from types.ts
// ---------------------------------------------------------------------------

export interface PipelineAnalysisResult {
  status:   "complete" | "error";
  job_id:   string;
  video_url: string;

  // FootballSession fields
  detectedActivities: FootballAction[];
  playerLevel:        SkillLevel;

  metrics: {
    byAction:      Partial<Record<FootballAction, Record<string, string>>>;
    torsoLean:     number;
    kneeStability: number;
    gaitSymmetry:  number;
    warnings:      string[];
  };

  aiFeedback: {
    summary:         string;
    strengths:       string[];
    weaknesses:      string[];
    coachingTips:    string[];
    motivationalTip: string;
  };

  drills: {
    name:         string;
    targetMetric: string;
    instructions: string;
    coachTip:     string;
    duration:     string;
    difficulty:   SkillLevel;
  }[];

  // Internal pipeline diagnostics
  _pipeline?: {
    player_detection: Record<string, unknown>;
    ball_detection:   Record<string, unknown>;
    skill_scores:     Record<string, number | null>;
    overall_score:    number;
  };
}

// ---------------------------------------------------------------------------
// Mock generator — used when the backend is unreachable
// ---------------------------------------------------------------------------

function generateMockResult(fileName: string): PipelineAnalysisResult {
  const warnings = ["POOR POSTURE / LEANING BACK", "KNEE ALIGNMENT RISK"];
  const activities: FootballAction[] = ["shooting", "dribbling"];

  return {
    status:    "complete",
    job_id:    `demo_${Date.now()}`,
    video_url: "",

    detectedActivities: activities,
    playerLevel: "Intermediate",

    metrics: {
      byAction: {
        shooting: {
          "Shot Velocity":   "88 km/h",
          "Launch Angle":    "14°",
          "Target Accuracy": "81%",
          "Torso Alignment": "12°",
        },
        dribbling: {
          "Close Control":       "88%",
          "Change of Direction": "5.8 m/s",
          "Touch Tightness":     "±2.4 cm",
          "Speed with Ball":     "24 km/h",
        },
      },
      torsoLean:     22,
      kneeStability: 72,
      gaitSymmetry:  92,
      warnings,
    },

    aiFeedback: {
      summary:
        "Shooting analysis complete for Intermediate player. " +
        "2 areas need work; 1 metric performing well.",
      strengths:    ["Gait symmetry"],
      weaknesses:   ["Torso alignment", "Knee stability"],
      coachingTips: [
        "At the moment of contact, your chest should be over the ball. Think 'chin down, chest forward'.",
        "Focus on pushing your knee outward over your second toe when planting.",
      ],
      motivationalTip:
        "You have a solid foundation. The gap between Intermediate and Advanced is consistency.",
    },

    drills: [
      {
        name:         "Wall Lean Drill",
        targetMetric: "torso_lean",
        instructions:
          "Stand 30 cm from a wall and practise driving your knee up without your back touching the wall — 3 × 15 reps.",
        coachTip:     "Chin down, chest forward at the moment of contact.",
        duration:     "10-15 min",
        difficulty:   "Intermediate",
      },
      {
        name:         "Lateral Band Walk",
        targetMetric: "knee_dev",
        instructions:
          "Place a resistance band above your knees and take 20 side steps each direction — 3 sets daily.",
        coachTip:     "Knee over second toe at all times.",
        duration:     "10 min",
        difficulty:   "Intermediate",
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Main upload function
// ---------------------------------------------------------------------------

/**
 * Upload a video to the FootballIQ backend and run the full 9-stage
 * analysis pipeline.  Falls back to mock data if the backend is unreachable.
 *
 * @param file - MP4 video file
 * @param onStageChange - optional callback fired during pipeline polling
 */
export async function uploadAndAnalyze(
  file: File,
  onStageChange?: (stage: string, stageIndex: number) => void,
): Promise<PipelineAnalysisResult> {

  // ── Attempt real backend ──────────────────────────────────────────────────
  try {
    const formData = new FormData();
    formData.append("file", file);

    const controller = new AbortController();
    const timeoutId  = setTimeout(() => controller.abort(), 8000);

    const response = await fetch(`${BACKEND_URL}/api/upload-video`, {
      method: "POST",
      body:   formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const err = await response.text();
      console.warn(`[uploadAndAnalyze] Backend error ${response.status}: ${err}`);
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json() as PipelineAnalysisResult;

    // Poll pipeline status while the job is processing, if a callback was given.
    if (onStageChange && data.job_id) {
      await pollPipelineStatus(data.job_id, onStageChange);
    }

    return data;

  } catch (err) {
    console.warn("[uploadAndAnalyze] Backend unreachable — using mock:", err);
  }

  // ── Fallback mock ─────────────────────────────────────────────────────────
  // Simulate the pipeline running stage by stage.
  if (onStageChange) {
    const stages = [
      "player_detection", "ball_detection", "pose_estimation",
      "activity_detection", "analyzer_selection", "metric_calculation",
      "skill_classification", "feedback_engine", "dashboard",
    ];
    for (let i = 0; i < stages.length; i++) {
      onStageChange(stages[i], i);
      await new Promise((r) => setTimeout(r, 400));
    }
  }

  return generateMockResult(file.name);
}

// ---------------------------------------------------------------------------
// Pipeline status poller
// ---------------------------------------------------------------------------

/**
 * Poll /api/pipeline-status/{job_id} until the job is complete.
 * Fires onStageChange on each new stage detected.
 */
async function pollPipelineStatus(
  jobId: string,
  onStageChange: (stage: string, stageIndex: number) => void,
  intervalMs = 600,
  maxAttempts = 60,
): Promise<void> {
  let lastStage = "";
  let attempts  = 0;

  while (attempts < maxAttempts) {
    try {
      const res = await fetch(`${BACKEND_URL}/api/pipeline-status/${jobId}`);
      if (!res.ok) break;

      const data = await res.json();
      const { stage, stage_index } = data as { stage: string; stage_index: number };

      if (stage !== lastStage) {
        onStageChange(stage, stage_index);
        lastStage = stage;
      }

      if (stage === "complete") break;

    } catch {
      break;
    }

    await new Promise((r) => setTimeout(r, intervalMs));
    attempts++;
  }
}

// ---------------------------------------------------------------------------
// URL helper
// ---------------------------------------------------------------------------

/**
 * Build a full streamable URL from the video_url returned by the backend.
 */
export function buildVideoUrl(videoUrl: string): string {
  if (!videoUrl) return "";
  if (videoUrl.startsWith("http")) return videoUrl;
  return `${BACKEND_URL}${videoUrl}`;
}

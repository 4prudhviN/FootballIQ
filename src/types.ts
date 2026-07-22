/* ── Navigation ──────────────────────────────────────── */
export type NavTab = "dashboard" | "upload" | "analysis" | "settings";

/** Detected football actions from video analysis */
export type FootballAction =
  | "passing"
  | "dribbling"
  | "shooting"
  | "goalkeeping"
  | "defending"
  | "movement";

/** Sub-navigation pillar tabs inside the Analysis view */
export type PillarTab = "technical" | "setpiece" | "gymplyo" | "stamina";

/* ── Pillar Metric Interfaces ────────────────────────── */

export interface TechnicalMetrics {
  coneAgility: string;          // e.g. "4.2s"
  changeOfDirectionSpeed: string; // e.g. "5.8 m/s"
  touchTightness: string;       // e.g. "92%"
  wallReboundAccuracy: string;  // e.g. "87%"
  firstTouchControlCushion: string; // e.g. "0.36 m/s²"
  weakFootRatio: string;        // e.g. "78/22"
  shotVelocity: string;         // e.g. "88 km/h"
  launchAngleElevation: string; // e.g. "14°"
  apexTargetAccuracy: string;   // e.g. "81%"
  /** Touch Tightness Index expressed as cm variance from ideal contact */
  touchTightnessCm: string;     // e.g. "±2.4 cm"
  /** Ball release / strike velocity */
  ballReleaseSpeed: string;     // e.g. "88 km/h"
  /** Classifier for foot-to-ball contact type */
  footStrikeContact: string;    // e.g. "Instep Drive / Laces"
}

export interface SetPieceMetrics {
  curveRate: string;               // e.g. "420 RPM"
  wallClearance: string;           // e.g. "0.8m"
  targetCornerAccuracy: string;    // e.g. "76%"
  penaltyForce: string;            // e.g. "920 N"
  goalkeeperDeceptionIndex: string; // e.g. "64%"
  gaitVectorConsistency: string;   // e.g. "88%"
  throwInSpineFlexion: string;     // e.g. "18°"
  elbowFlexionAcceleration: string; // e.g. "4.2 m/s²"
  releasePointTrajectory: string;  // e.g. "24.5m"
  /** Ball spin & rotation rate */
  ballSpinRPM: string;             // e.g. "420 RPM"
  /** Wall clearance elevation margin in meters */
  wallClearanceElevation: string;  // e.g. "0.8m"
  /** Target destination vector deviation from aim point */
  targetVectorDeviation: string;   // e.g. "+0.2m High-Right"
}

export interface GymPlyoMetrics {
  barVelocity: string;            // e.g. "0.65 m/s"
  eccentricControlRatio: string;  // e.g. "0.82"
  leftRightForceBalancing: string; // e.g. "92%"
  verticalJumpHeight: string;     // e.g. "58 cm"
  groundContactTime: string;      // e.g. "180 ms"
  reactiveStrengthIndex: string;  // e.g. "2.4"
  hipExtensionAngle: string;      // e.g. "168°"
  kneeFlexionAngle: string;       // e.g. "92°"
  /** Flight time vs ground contact time in ms */
  flightTimeMs: string;           // e.g. "420 ms"
  groundContactTimeMs: string;    // e.g. "180 ms"
  /** RSI calculated from flight time / contact time */
  rsi: string;                    // e.g. "2.33"
  /** Torso lean angle from vertical axis */
  torsoLeanAngle: string;         // e.g. "7.2°"
}

export interface StaminaMetrics {
  maxSprintVelocity: string;      // e.g. "31.2 km/h"
  strideAsymmetry: string;        // e.g. "4.2%"
  fatigueDropoff: string;         // e.g. "12% after minute 3"
  strideLengthConsistency: string; // e.g. "94%"
  accelerationBurstTime: string;  // e.g. "1.8s"
  workToRestRatio: string;        // e.g. "2.3:1"
  // Extended stamina metrics (16 total)
  vo2MaxEstimate: string;          // e.g. "52 mL/kg/min"
  heartRateRecovery: string;       // e.g. "22 bpm drop in 1 min"
  lactateThresholdPace: string;    // e.g. "4:30/km"
  strideFrequency: string;         // e.g. "178 spm"
  strideLength: string;            // e.g. "1.24 m"
  verticalOscillation: string;     // e.g. "7.2 cm"
  groundContactBalance: string;    // e.g. "49.2% left/right"
  brakingForceSymmetry: string;    // e.g. "92%"
  strideExtensionAsymmetry: string; // e.g. "3.8%"
  peakVelocity: string;             // e.g. "32.4 km/h"
  mechanicalEfficiencyDropoff: string; // e.g. "12%"
}

export interface FeedbackItem {
  title: string;
  description: string;
  drillName: string;
  drillDetails: string;
}

/* ── Analysis Result ─────────────────────────────────── */
export interface AnalysisResult {
  id: string;
  fileName: string;
  date: Date;
  status: "completed" | "processing" | "failed";
  videoUrl?: string;
  warnings: string[];
  metrics?: {
    torsoLean: number;
    kneeStability: number;
    gaitSymmetry: number;
  };
  /** Expanded pillar metrics (returned by the new server endpoint) */
  technical?: TechnicalMetrics;
  setPieces?: SetPieceMetrics;
  gymPlyo?: GymPlyoMetrics;
  stamina?: StaminaMetrics;
  feedback?: FeedbackItem;
  /** Detected football actions from activity_detector — drives dashboard cards */
  detectedActions?: FootballAction[];
}

/* ── Skill Level ─────────────────────────────────────── */
export type SkillLevel = "Beginner" | "Intermediate" | "Advanced";

/* ── Per-action metric snapshot ──────────────────────── */
/**
 * A flat key→value map of display-ready metric values for a single
 * detected football action (e.g. passing, shooting).
 * Values are strings so the UI can display them without conversion.
 */
export type ActionMetrics = Record<string, string>;

/* ── AI Feedback ─────────────────────────────────────── */
export interface AIFeedback {
  /** One-sentence plain-English summary of the session */
  summary: string;
  /** Things the player did well */
  strengths: string[];
  /** Weak areas identified by the feedback engine */
  weaknesses: string[];
  /** Targeted coaching tips from feedback_engine.py */
  coachingTips: string[];
  /** Motivational closing message (level-aware) */
  motivationalTip: string;
}

/* ── Training Drill ──────────────────────────────────── */
export interface TrainingDrill {
  /** Short drill name, e.g. "Wall Lean Drill" */
  name: string;
  /** Target metric this drill addresses, e.g. "torso_lean" */
  targetMetric: string;
  /** Step-by-step instructions */
  instructions: string;
  /** Estimated time to complete one set */
  duration: string;
  /** Difficulty relative to player level */
  difficulty: SkillLevel;
}

/* ═══════════════════════════════════════════════════════
   Master Session Object
   ═══════════════════════════════════════════════════════
   Everything in the app revolves around this interface.
   Produced by the analysis pipeline and consumed by every
   component — DashboardTab, AnalysisTab, UploadTab, etc.
   ─────────────────────────────────────────────────────── */
export interface FootballSession {
  /** Unique session identifier */
  id: string;

  /** Original video filename */
  fileName: string;

  /** When the session was created */
  date: Date;

  /** Pipeline processing status */
  status: "completed" | "processing" | "failed";

  /** URL of the annotated output video */
  videoUrl?: string;

  /**
   * Activities detected by activity_detector.py.
   * Drives which action cards are rendered in the dashboard.
   * Only non-empty when status === "completed".
   */
  detectedActivities: FootballAction[];

  /**
   * Skill level produced by skill_classifier.py.
   * Determines drill difficulty and motivational messaging.
   */
  playerLevel: SkillLevel;

  /**
   * Per-action metric snapshots keyed by FootballAction.
   * e.g. metrics["passing"] = { "Ball Control": "92%", ... }
   * Also includes core biomechanical scalars (torsoLean, etc.)
   */
  metrics: {
    /** Per-action display metrics from the analyzer pipeline */
    byAction: Partial<Record<FootballAction, ActionMetrics>>;
    /** Core biomechanical scalars from analyze_movement.py */
    torsoLean: number;
    kneeStability: number;
    gaitSymmetry: number;
    /** Raw warning strings from the video analysis */
    warnings: string[];
  };

  /**
   * AI-generated coaching feedback from feedback_engine.py.
   * Includes summary, strengths, weaknesses, tips, and motivation.
   */
  aiFeedback: AIFeedback;

  /**
   * Ordered list of recommended training drills.
   * First item = highest priority (shown in the Correction Hub).
   */
  drills: TrainingDrill[];

  /** Optional expanded pillar metrics (legacy / future backend fields) */
  technical?: TechnicalMetrics;
  setPieces?: SetPieceMetrics;
  gymPlyo?: GymPlyoMetrics;
  stamina?: StaminaMetrics;
}
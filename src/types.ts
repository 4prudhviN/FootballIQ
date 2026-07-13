/* ── Navigation ──────────────────────────────────────── */
export type NavTab = "dashboard" | "upload" | "analysis" | "settings";

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
}
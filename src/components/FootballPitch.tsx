import React from "react";

// ── Pitch dimensions (scaled ×10 for precision) ──────────────
const PITCH_W = 1050; // corresponds to 105 m
const PITCH_H = 680; // corresponds to 68 m

// Helper: center of pitch
const CX = PITCH_W / 2;
const CY = PITCH_H / 2;

// Area dimensions (in scaled units)
const PENALTY_DEPTH = 165; // 16.5 m
const PENALTY_WIDTH = 403; // 40.32 m
const GOAL_AREA_DEPTH = 55; // 5.5 m
const GOAL_AREA_WIDTH = 183; // 18.32 m
const CENTER_RADIUS = 92; // 9.15 m
const CORNER_RADIUS = 10;
const GOAL_WIDTH = 73; // 7.32 m
const GOAL_DEPTH = 24; // 2.44 m
const PENALTY_SPOT = 110; // 11 m from goal line

const LINE_COLOR = "rgba(255,255,255,0.5)";
const LINE_WIDTH = 1.5;
const PITCH_FILL = "#1a472a";
const GOAL_FILL = "rgba(255,255,255,0.08)";

interface FootballPitchProps {
  children?: React.ReactNode;
  className?: string;
  width?: number | string;
  height?: number | string;
}

/** A reusable top-down football pitch SVG component.
 *
 *  The viewBox maps 0–1050 (x) × 0–680 (y), so every unit = 0.1 m.
 *  Pass map and heatmap overlay coordinates should be in the same
 *  scaled space for correct positioning.
 */
const FootballPitch: React.FC<FootballPitchProps> = ({
  children,
  className = "",
  width = "100%",
  height = "100%",
}) => {
  const clipId = "pitch-clip";

  return (
    <svg
      viewBox={`0 0 ${PITCH_W} ${PITCH_H}`}
      width={width}
      height={height}
      className={className}
      role="img"
      aria-label="Football pitch — top down view"
      style={{ display: "block" }}
    >
      {/* ── Clip path keeps everything inside the pitch ── */}
      <defs>
        <clipPath id={clipId}>
          <rect x={0} y={0} width={PITCH_W} height={PITCH_H} rx={4} />
        </clipPath>
      </defs>

      <g clipPath={`url(#${clipId})`}>
        {/* Pitch background */}
        <rect
          x={0}
          y={0}
          width={PITCH_W}
          height={PITCH_H}
          fill={PITCH_FILL}
          rx={4}
        />

        {/* Alternating horizontal stripes (grass pattern) */}
        {Array.from({ length: 20 }, (_, i) => (
          <rect
            key={i}
            x={0}
            y={i * 34}
            width={PITCH_W}
            height={34}
            fill={i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)"}
          />
        ))}

        {/* ── Pitch outline ── */}
        <rect
          x={0}
          y={0}
          width={PITCH_W}
          height={PITCH_H}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
          rx={4}
        />

        {/* ── Halfway line ── */}
        <line
          x1={CX}
          y1={0}
          x2={CX}
          y2={PITCH_H}
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />

        {/* ── Centre circle ── */}
        <circle
          cx={CX}
          cy={CY}
          r={CENTER_RADIUS}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Centre spot */}
        <circle cx={CX} cy={CY} r={3} fill={LINE_COLOR} />

        {/* ── Left penalty area ── */}
        <rect
          x={0}
          y={(PITCH_H - PENALTY_WIDTH) / 2}
          width={PENALTY_DEPTH}
          height={PENALTY_WIDTH}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Left goal area */}
        <rect
          x={0}
          y={(PITCH_H - GOAL_AREA_WIDTH) / 2}
          width={GOAL_AREA_DEPTH}
          height={GOAL_AREA_WIDTH}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Left penalty spot */}
        <circle cx={PENALTY_SPOT} cy={CY} r={3} fill={LINE_COLOR} />
        {/* Left penalty arc (only the outer half) */}
        <path
          d={`M ${PENALTY_DEPTH} ${(PITCH_H - PENALTY_WIDTH) / 2}
              A ${CENTER_RADIUS} ${CENTER_RADIUS} 0 0 1
              ${PENALTY_DEPTH} ${(PITCH_H + PENALTY_WIDTH) / 2}`}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Left goal */}
        <rect
          x={-GOAL_DEPTH}
          y={(PITCH_H - GOAL_WIDTH) / 2}
          width={GOAL_DEPTH}
          height={GOAL_WIDTH}
          fill={GOAL_FILL}
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH * 1.5}
        />

        {/* ── Right penalty area ── */}
        <rect
          x={PITCH_W - PENALTY_DEPTH}
          y={(PITCH_H - PENALTY_WIDTH) / 2}
          width={PENALTY_DEPTH}
          height={PENALTY_WIDTH}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Right goal area */}
        <rect
          x={PITCH_W - GOAL_AREA_DEPTH}
          y={(PITCH_H - GOAL_AREA_WIDTH) / 2}
          width={GOAL_AREA_DEPTH}
          height={GOAL_AREA_WIDTH}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Right penalty spot */}
        <circle cx={PITCH_W - PENALTY_SPOT} cy={CY} r={3} fill={LINE_COLOR} />
        {/* Right penalty arc (only the outer half) */}
        <path
          d={`M ${PITCH_W - PENALTY_DEPTH} ${(PITCH_H - PENALTY_WIDTH) / 2}
              A ${CENTER_RADIUS} ${CENTER_RADIUS} 0 0 0
              ${PITCH_W - PENALTY_DEPTH} ${(PITCH_H + PENALTY_WIDTH) / 2}`}
          fill="none"
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH}
        />
        {/* Right goal */}
        <rect
          x={PITCH_W}
          y={(PITCH_H - GOAL_WIDTH) / 2}
          width={GOAL_DEPTH}
          height={GOAL_WIDTH}
          fill={GOAL_FILL}
          stroke={LINE_COLOR}
          strokeWidth={LINE_WIDTH * 1.5}
        />

        {/* ── Corner arcs ── */}
        <circle cx={CORNER_RADIUS} cy={CORNER_RADIUS} r={CORNER_RADIUS} fill="none" stroke={LINE_COLOR} strokeWidth={LINE_WIDTH} />
        <circle cx={PITCH_W - CORNER_RADIUS} cy={CORNER_RADIUS} r={CORNER_RADIUS} fill="none" stroke={LINE_COLOR} strokeWidth={LINE_WIDTH} />
        <circle cx={CORNER_RADIUS} cy={PITCH_H - CORNER_RADIUS} r={CORNER_RADIUS} fill="none" stroke={LINE_COLOR} strokeWidth={LINE_WIDTH} />
        <circle cx={PITCH_W - CORNER_RADIUS} cy={PITCH_H - CORNER_RADIUS} r={CORNER_RADIUS} fill="none" stroke={LINE_COLOR} strokeWidth={LINE_WIDTH} />

        {/* ── Overlay children (passes, heatmaps, positions) ── */}
        {children}
      </g>
    </svg>
  );
};

export default React.memo(FootballPitch);
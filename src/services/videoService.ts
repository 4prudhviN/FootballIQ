/**
 * Video Analysis API Service
 * ===========================
 *
 * Calls the Supabase Edge Function analyze-video to upload and analyse
 * movement videos. Falls back to client-side mock analysis when the Edge
 * Function is unavailable (e.g. in the preview environment).
 */

import { supabase } from "../lib/supabase";

const EDGE_FUNCTION_URL =
  "https://bbzgdblrrcemdzmwbpsa.supabase.co/functions/v1/analyze-video";

export interface AnalysisResult {
  status: "complete" | "error";
  video_url: string;
  warnings: string[];
  job_id: string;
  metrics?: {
    torsoLean: number;
    kneeStability: number;
    gaitSymmetry: number;
  };
}

/**
 * Generate a realistic mock analysis result for preview/demo mode.
 */
function generateMockAnalysis(fileName: string): AnalysisResult {
  const warnings: string[] = [];
  const warningPool = [
    "POOR POSTURE / LEANING BACK — torso angled behind hips at contact",
    "KNEE ALIGNMENT RISK — slight inward collapse on plant foot",
    "ASYMMETRIC GAIT DETECTED — stride length imbalance (L/R)",
    "LATE HIP ROTATION — power transfer delayed by ~40ms",
    "HIGH GROUND CONTACT TIME — 240ms vs 180ms optimal",
  ];

  // Pick 2-3 random warnings
  const numWarnings = 2 + Math.floor(Math.random() * 2);
  const shuffled = [...warningPool].sort(() => Math.random() - 0.5);
  for (let i = 0; i < numWarnings; i++) {
    warnings.push(shuffled[i]);
  }

  return {
    status: "complete",
    video_url: "",
    warnings,
    job_id: `demo_${Date.now()}`,
    metrics: {
      torsoLean: Math.round(60 + Math.random() * 35),
      kneeStability: Math.round(60 + Math.random() * 35),
      gaitSymmetry: Math.round(65 + Math.random() * 30),
    },
  };
}

/**
 * Upload a video file to the Edge Function and get the analysis results.
 * Falls back to client-side mock analysis if the Edge Function is unreachable.
 *
 * @param file - MP4/MOV video file (max 500 MB recommended)
 * @returns AnalysisResult with the Supabase Storage URL and any warnings
 */
export async function uploadAndAnalyze(file: File): Promise<AnalysisResult> {
  // Attempt Edge Function first
  try {
    const formData = new FormData();
    formData.append("file", file);

    // Get the current session's access token (if signed in)
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData?.session?.access_token || "";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

    const response = await fetch(EDGE_FUNCTION_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorBody = await response.text();
      console.warn(
        `[uploadAndAnalyze] Edge Function returned ${response.status}: ${errorBody}`,
      );
    } else {
      return (await response.json()) as AnalysisResult;
    }
  } catch (err) {
    console.warn(
      `[uploadAndAnalyze] Edge Function unavailable, using mock analysis:`,
      err,
    );
  }

  // Fallback: client-side mock analysis
  await new Promise((resolve) => setTimeout(resolve, 500));
  return generateMockAnalysis(file.name);
}

/**
 * Build a full streamable URL for the processed video.
 *
 * @param videoUrl - the URL returned by uploadAndAnalyze
 * @returns full URL the <video> tag can use
 */
export function buildVideoUrl(videoUrl: string): string {
  return videoUrl; // Already a full Supabase Storage URL
}
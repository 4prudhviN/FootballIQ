import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";

// ── CORS Headers ─────────────────────────────────────────
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

// ── Analysis warning pool (mirrors frontend mock) ────────
const WARNING_POOL = [
  "POOR POSTURE / LEANING BACK — torso angled behind hips at contact",
  "KNEE ALIGNMENT RISK — slight inward collapse on plant foot",
  "ASYMMETRIC GAIT DETECTED — stride length imbalance (L/R)",
  "LATE HIP ROTATION — power transfer delayed by ~40ms",
  "HIGH GROUND CONTACT TIME — 240ms vs 180ms optimal",
];

function generateWarnings(): string[] {
  const count = 2 + Math.floor(Math.random() * 2); // 2–3 warnings
  const shuffled = [...WARNING_POOL].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

function generateMetrics() {
  return {
    torsoLean: Math.round(60 + Math.random() * 35),
    kneeStability: Math.round(60 + Math.random() * 35),
    gaitSymmetry: Math.round(65 + Math.random() * 30),
  };
}

// ── Main handler ─────────────────────────────────────────
Deno.serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", {
      status: 204,
      headers: corsHeaders,
    });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    // ── Authenticate user (optional for demo/preview) ────
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const authHeader = req.headers.get("Authorization") || "";
    const supabase = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authHeader } },
      auth: { persistSession: false },
    });

    // Try to get the authenticated user — not required for preview/demo mode
    let userId: string | null = null;
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) userId = user.id;
    } catch {
      // Auth token may not be present in preview mode — that's fine
    }

    // ── Parse the multipart form data ──────────────────
    const formData = await req.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof File)) {
      return new Response(
        JSON.stringify({
          status: "error",
          error: "No video file provided. Please upload an MP4 file.",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Validate file type
    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith(".mp4") && !fileName.endsWith(".mov")) {
      return new Response(
        JSON.stringify({
          status: "error",
          error: "Only MP4 and MOV files are accepted.",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > 500) {
      return new Response(
        JSON.stringify({
          status: "error",
          error: "File too large. Maximum size is 500 MB.",
        }),
        {
          status: 413,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // ── Upload file to Supabase Storage ─────────────────
    const timestamp = Date.now();
    const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    const storagePath = `${userId || "anonymous"}/${timestamp}_${safeName}`;

    const { data: uploadData, error: uploadError } = await supabase.storage
      .from("videos")
      .upload(storagePath, file, {
        contentType: file.type || "video/mp4",
        upsert: false,
      });

    if (uploadError) {
      console.error("Storage upload error:", uploadError);
      return new Response(
        JSON.stringify({
          status: "error",
          error: "Failed to upload video to storage.",
          details: uploadError.message,
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Get the public URL for the uploaded video
    const { data: urlData } = supabase.storage
      .from("videos")
      .getPublicUrl(storagePath);
    const videoUrl = urlData?.publicUrl || "";

    // ── Run analysis (simulated for MVP) ────────────────
    // In production, replace this with TensorFlow.js / MediaPipe inference
    const analysisStart = Date.now();
    await new Promise((resolve) => setTimeout(resolve, 2000)); // simulate compute
    console.log(`[INFO] Analysis took ${Date.now() - analysisStart}ms`);

    const warnings = generateWarnings();
    const metrics = generateMetrics();

    // ── Try to save results to database (skip if no auth) ──
    if (userId) {
      const { error: insertError } = await supabase
        .from("analysis_results")
        .insert({
          user_id: userId,
          file_name: file.name,
          file_path: storagePath,
          status: "completed",
          video_url: videoUrl,
          warnings,
          metrics,
        });

      if (insertError) {
        console.error("DB insert error:", insertError);
      }
    }

    // ── Return the analysis result ──────────────────────
    return new Response(
      JSON.stringify({
        status: "complete",
        video_url: videoUrl,
        warnings,
        job_id: `${user.id}_${timestamp}`,
        metrics,
      }),
      {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  } catch (err) {
    console.error("Unhandled error:", err);
    return new Response(
      JSON.stringify({
        status: "error",
        error: "Analysis failed unexpectedly. Please try again.",
        details: err instanceof Error ? err.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  }
});
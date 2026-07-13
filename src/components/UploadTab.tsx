import { useState, useCallback, useRef } from "react";
import { Upload, Film, CheckCircle, AlertCircle } from "lucide-react";
import { uploadAndAnalyze, buildVideoUrl } from "../services/videoService";

type UploadStatus = "idle" | "dragging" | "uploading" | "success" | "error" | "analyzing";

interface UploadTabProps {
  onAnalysisComplete: (videoUrl: string, warnings: string[], fileName: string) => void;
}

export default function UploadTab({ onAnalysisComplete }: UploadTabProps) {
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* ── Simulated progress during real upload ──────────── */
  const startProgressSimulation = useCallback(() => {
    setProgress(0);
    progressRef.current = setInterval(() => {
      setProgress((prev) => {
        const next = prev + Math.floor(Math.random() * 8) + 2;
        return next >= 92 ? 92 : next; // cap at 92% — last 8% on response
      });
    }, 300);
  }, []);

  const stopProgressSimulation = useCallback(() => {
    if (progressRef.current) {
      clearInterval(progressRef.current);
      progressRef.current = null;
    }
  }, []);

  /* ── Check if the file looks like a video ──────────── */
  function isValidVideo(file: File): boolean {
    // Check MIME type (reliable in most modern browsers)
    if (file.type.startsWith("video/")) return true;
    // Some browsers/phones report empty or generic MIME types — check extension
    const videoExtensions = [".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"];
    const lowerName = file.name.toLowerCase();
    return videoExtensions.some((ext) => lowerName.endsWith(ext));
  }

  /* ── File handler: upload + backend analysis ──────────── */
  const handleFile = useCallback(
    async (file: File) => {
      if (!isValidVideo(file)) {
        setStatus("error");
        setFileName(file.name);
        setErrorMessage(
          `"${file.name}" doesn't look like a video file. Please upload an MP4, MOV, or AVI file.`,
        );
        return;
      }

      if (file.size > 500 * 1024 * 1024) {
        setStatus("error");
        setFileName(file.name);
        setErrorMessage("File is too large. Maximum size is 500 MB.");
        return;
      }

      setFileName(file.name);
      setErrorMessage(null);
      setStatus("uploading");
      startProgressSimulation();

      try {
        const result = await uploadAndAnalyze(file);

        stopProgressSimulation();
        setProgress(100);

        // Brief success state so the user sees the checkmark
        setStatus("success");

        // Auto-navigate to analysis after a short delay
        setTimeout(() => {
          onAnalysisComplete(buildVideoUrl(result.video_url), result.warnings, file.name);
        }, 800);
      } catch (err) {
        stopProgressSimulation();
        setStatus("error");
        setErrorMessage(
          err instanceof Error
            ? err.message
            : "Failed to connect to the analysis server. Make sure the backend is running on port 8000.",
        );
      }
    },
    [startProgressSimulation, stopProgressSimulation, onAnalysisComplete],
  );

  /* ── Drag events ─────────────────────────────────── */
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setStatus("dragging");
  }, []);

  const handleDragLeave = useCallback(() => {
    setStatus("idle");
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
      else setStatus("idle");
    },
    [handleFile],
  );

  /* ── Click to browse ─────────────────────────────── */
  const handleClick = () => inputRef.current?.click();

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  /* ── Reset ───────────────────────────────────────── */
  const handleReset = () => {
    stopProgressSimulation();
    setStatus("idle");
    setProgress(0);
    setFileName(null);
    setErrorMessage(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  /* ── Drag zone border color ─────────────────────── */
  const borderColor =
    status === "dragging"
      ? "border-primary"
      : status === "error"
        ? "border-destructive"
        : status === "success"
          ? "border-success"
          : "border-border";

  const bgColor =
    status === "dragging"
      ? "bg-primary/5"
      : "bg-surface";

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 mx-auto max-w-xl space-y-8">
      {/* Heading */}
      <div className="text-center">
        <h1 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">
          Upload Drill Video
        </h1>
        <p className="mt-1.5 text-sm text-muted">
          Get AI-powered technique analysis in seconds.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={status === "idle" || status === "dragging" ? handleClick : undefined}
        className={`
          relative cursor-pointer rounded-2xl border-2 border-dashed ${borderColor} ${bgColor}
          p-10 sm:p-14 text-center transition-all duration-300
          ${status === "idle" || status === "dragging" ? "hover:border-primary hover:bg-primary/[0.04]" : "cursor-default"}
        `}
        role="button"
        tabIndex={0}
        aria-label="Upload a video file"
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && (status === "idle" || status === "dragging")) {
            e.preventDefault();
            handleClick();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={handleInputChange}
        />

        {/* State-driven content */}
        {status === "idle" || status === "dragging" ? (
          <>
            <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Upload size={30} className="text-primary" strokeWidth={1.8} />
            </div>
            <p className="font-heading text-lg font-semibold text-foreground">
              {status === "dragging" ? "Release to upload" : "Drop your video here"}
            </p>
            <p className="mt-1.5 text-sm text-muted">
              or <span className="text-primary underline underline-offset-2">browse files</span>
            </p>
            <p className="mt-4 text-xs text-muted/70">
              MP4, MOV, or AVI &middot; Max 60 seconds &middot; Up to 500 MB
            </p>
          </>
        ) : status === "uploading" ? (
          <>
            <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Film size={30} className="text-primary" strokeWidth={1.8} />
            </div>
            <p className="font-heading text-lg font-semibold text-foreground">
              Uploading &amp; Analyzing&hellip;
            </p>
            <p className="mt-1 text-sm text-muted truncate max-w-xs mx-auto">
              {fileName}
            </p>

            {/* Progress bar */}
            <div className="mx-auto mt-6 h-2 w-full max-w-xs overflow-hidden rounded-full bg-border">
              <div
                className="h-full rounded-full bg-primary transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-muted">{progress}% complete</p>
          </>
        ) : status === "success" ? (
          <>
            <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-success/10">
              <CheckCircle size={30} className="text-success" strokeWidth={1.8} />
            </div>
            <p className="font-heading text-lg font-semibold text-foreground">
              Analysis complete!
            </p>
            <p className="mt-1 text-sm text-muted truncate max-w-xs mx-auto">
              {fileName}
            </p>
            <p className="mt-3 text-xs text-muted animate-pulse">
              Redirecting to analysis&hellip;
            </p>
          </>
        ) : (
          <>
            <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-destructive/10">
              <AlertCircle size={30} className="text-destructive" strokeWidth={1.8} />
            </div>
            <p className="font-heading text-lg font-semibold text-foreground">
              Upload failed
            </p>
            <p className="mt-1 text-sm text-muted max-w-xs mx-auto">
              {errorMessage || "Something went wrong. Please try again."}
            </p>
          </>
        )}
      </div>

      {/* Instructions */}
      <div className="rounded-2xl border border-border bg-surface p-5 sm:p-6">
        <h2 className="font-heading text-sm font-semibold uppercase tracking-wider text-muted">
          Recording Tips
        </h2>
        <ul className="mt-3 space-y-2.5 text-sm text-muted">
          <li className="flex items-start gap-2.5">
            <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
            Upload a short video (under 60 seconds) of your shooting or passing drill.
          </li>
          <li className="flex items-start gap-2.5">
            <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
            Ensure the camera is at a stable, fixed side-angle for best analysis.
          </li>
          <li className="flex items-start gap-2.5">
            <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
            Keep the full body in frame — from head to toe.
          </li>
        </ul>
      </div>

      {/* Action buttons */}
      <div className="flex flex-col gap-3">
        <button
          disabled={status !== "success"}
          onClick={() => {
            /* Already auto-navigates on success, but manual click also works */
          }}
          className={`
            flex w-full items-center justify-center gap-2 rounded-xl px-6 py-3.5
            text-base font-semibold transition-all duration-200
            ${status === "success"
              ? "bg-primary text-on-primary hover:bg-primary-dark active:scale-[0.97] cursor-pointer"
              : "bg-surface text-muted/50 cursor-not-allowed"
            }
          `}
        >
          <Film size={18} />
          Processing Complete
        </button>

        {(status === "success" || status === "error") && (
          <button
            onClick={handleReset}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-border px-6 py-3 text-sm font-medium text-muted hover:text-foreground hover:bg-surface-hover transition-all duration-200 cursor-pointer"
          >
            Upload another video
          </button>
        )}
      </div>
    </div>
  );
}
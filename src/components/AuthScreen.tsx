import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  ChevronRight,
  Shield,
  Target,
  Fence,
  Swords,
  Zap,
  Check,
  Loader2,
} from "lucide-react";

/* ── Props ────────────────────────────────────────────── */
interface AuthScreenProps {
  onLogin: (playerName: string) => void;
}

/* ── Types ────────────────────────────────────────────── */
type AuthView = "signin" | "signup";
type Position = "striker" | "midfielder" | "defender" | "winger";
type Foot = "right" | "left" | "ambidextrous";

/* ── Constants ────────────────────────────────────────── */
const POSITIONS: { id: Position; label: string; icon: React.ReactNode }[] = [
  { id: "striker", label: "Striker", icon: <Target size={16} /> },
  { id: "midfielder", label: "Midfielder", icon: <Swords size={16} /> },
  { id: "defender", label: "Defender", icon: <Shield size={16} /> },
  { id: "winger", label: "Winger", icon: <Zap size={16} /> },
];

const FOOT_OPTIONS: { id: Foot; label: string }[] = [
  { id: "right", label: "Dominant Right" },
  { id: "left", label: "Dominant Left" },
  { id: "ambidextrous", label: "Ambidextrous" },
];

/* ── Left Panel — Brand Experience ────────────────────── */
function BrandPanel() {
  return (
    <div className="relative hidden lg:flex h-full w-[45%] flex-col justify-between overflow-hidden">
      {/* Background image layer */}
      <div className="absolute inset-0 bg-[#0B0F12]">
        <div
          className="h-full w-full opacity-[0.25]"
          style={{
            backgroundImage: `
              radial-gradient(ellipse 80% 60% at 50% 30%, rgba(0,255,102,0.08) 0%, transparent 70%),
              radial-gradient(ellipse 60% 50% at 70% 60%, rgba(0,255,102,0.04) 0%, transparent 60%),
              repeating-linear-gradient(90deg, transparent, transparent 60px, rgba(0,255,102,0.03) 60px, rgba(0,255,102,0.03) 61px),
              repeating-linear-gradient(0deg, transparent, transparent 60px, rgba(0,255,102,0.03) 60px, rgba(0,255,102,0.03) 61px)
            `,
          }}
        />
        {/* Floodlight glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-gradient-radial from-primary/8 via-primary/4 to-transparent rounded-full blur-3xl" />
        {/* Pitch perspective lines */}
        <div className="absolute bottom-0 left-0 right-0 h-[50%] overflow-hidden">
          <svg viewBox="0 0 500 300" className="w-full h-full opacity-[0.06]">
            <line x1="50" y1="300" x2="250" y2="0" stroke="#00FF66" strokeWidth="1.5" />
            <line x1="450" y1="300" x2="250" y2="0" stroke="#00FF66" strokeWidth="1.5" />
            <line x1="150" y1="300" x2="250" y2="100" stroke="#00FF66" strokeWidth="1" strokeDasharray="8,6" />
            <line x1="350" y1="300" x2="250" y2="100" stroke="#00FF66" strokeWidth="1" strokeDasharray="8,6" />
            <rect x="180" y="30" width="140" height="100" fill="none" stroke="#00FF66" strokeWidth="1.5" rx="4" />
            <rect x="210" y="50" width="80" height="60" fill="none" stroke="#00FF66" strokeWidth="1" rx="3" />
          </svg>
        </div>
      </div>

      {/* Gradient overlay from left */}
      <div className="absolute inset-0 bg-gradient-to-r from-[#0B0F12]/90 via-[#0B0F12]/30 to-transparent" />

      {/* Content */}
      <div className="relative z-10 flex h-full flex-col justify-between p-10 xl:p-14">
        <div>
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="flex items-center gap-3"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/15 ring-1 ring-primary/30 shadow-lg shadow-primary/10">
              <span className="text-xl font-black text-primary">IQ</span>
            </div>
            <span className="text-2xl font-black tracking-tight text-foreground">
              Football<span className="text-primary">IQ</span>
            </span>
          </motion.div>

          {/* Tagline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
            className="mt-24 xl:mt-32"
          >
            <h1 className="font-heading text-4xl xl:text-5xl font-extrabold leading-tight tracking-tight text-foreground">
              <span className="text-primary">Democratizing</span>
              <br />
              Elite Sports
              <br />
              Intelligence
            </h1>
            <p className="mt-4 max-w-md text-base leading-relaxed text-muted/80">
              Precision biomechanics, AI-driven diagnostics, and pro-level corrective
              programming — now available for every solo athlete, everywhere.
            </p>
          </motion.div>
        </div>

        {/* Bottom stat bar */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex items-center gap-6 text-xs text-muted/60"
        >
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            AI-Powered Analysis
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Pro-Grade Telemetry
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Solo Athlete Ecosystem
          </span>
        </motion.div>
      </div>
    </div>
  );
}

/* ── Input Field ──────────────────────────────────────── */
function AuthInput({
  label,
  type,
  value,
  onChange,
  icon,
  placeholder,
  error,
  autoComplete,
}: {
  label: string;
  type: string;
  value: string;
  onChange: (v: string) => void;
  icon: React.ReactNode;
  placeholder?: string;
  error?: string;
  autoComplete?: string;
}) {
  const [focused, setFocused] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const isPassword = type === "password";
  const resolvedType = isPassword && showPassword ? "text" : type;

  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-semibold uppercase tracking-wider text-muted">
        {label}
      </label>
      <div
        className={`group flex items-center gap-3 rounded-xl border bg-background/60 px-4 py-3 transition-all duration-200 ${
          error
            ? "border-destructive/60 ring-1 ring-destructive/30"
            : focused
              ? "border-primary shadow-[0_0_16px_rgba(0,255,102,0.08)]"
              : "border-border hover:border-border/70"
        }`}
      >
        <span
          className={`shrink-0 transition-colors duration-200 ${
            focused ? "text-primary" : "text-muted/60"
          }`}
        >
          {icon}
        </span>
        <input
          type={resolvedType}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          autoComplete={autoComplete}
          className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted/40 focus:outline-none"
          aria-label={label}
        />
        {isPassword && value && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="shrink-0 text-muted/50 hover:text-muted transition-colors duration-150 cursor-pointer"
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        )}
      </div>
      {error && (
        <p className="text-xs text-destructive pl-1">{error}</p>
      )}
    </div>
  );
}

/* ── Sign In View ─────────────────────────────────────── */
function SignInView({
  onLogin,
  onSwitch,
}: {
  onLogin: (name: string) => void;
  onSwitch: () => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [loading, setLoading] = useState(false);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!email || !password) return;
      setLoading(true);
      setTimeout(() => {
        onLogin(email.split("@")[0] || "Athlete");
      }, 800);
    },
    [email, password, onLogin],
  );

  return (
    <motion.div
      initial={{ opacity: 0, transform: "translateX(24px)" }}
      animate={{ opacity: 1, transform: "translateX(0)" }}
      exit={{ opacity: 0, transform: "translateX(-24px)" }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <form onSubmit={handleSubmit} className="space-y-5">
        <AuthInput
          label="Athlete Email"
          type="email"
          value={email}
          onChange={setEmail}
          icon={<Mail size={16} />}
          placeholder="you@example.com"
          autoComplete="email"
        />
        <AuthInput
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
          icon={<Lock size={16} />}
          placeholder="••••••••"
          autoComplete="current-password"
        />

        <div className="flex items-center justify-between">
          <label className="flex cursor-pointer items-center gap-2 select-none">
            <input
              type="checkbox"
              checked={remember}
              onChange={() => setRemember(!remember)}
              className="peer sr-only"
            />
            <span className="flex h-4 w-4 items-center justify-center rounded border border-border bg-background transition-all duration-150 peer-checked:border-primary peer-checked:bg-primary/20 peer-focus-visible:ring-2 peer-focus-visible:ring-ring/50">
              {remember && (
                <Check size={10} className="text-primary" strokeWidth={3} />
              )}
            </span>
            <span className="text-xs text-muted peer-checked:text-foreground transition-colors">
              Remember Device
            </span>
          </label>
          <button
            type="button"
            className="text-xs font-medium text-primary/70 hover:text-primary transition-colors duration-150 cursor-pointer"
          >
            Forgot Password?
          </button>
        </div>

        <button
          type="submit"
          disabled={loading || !email || !password}
          className="relative w-full overflow-hidden rounded-xl bg-primary px-6 py-3.5 text-sm font-bold text-on-primary transition-all duration-200 hover:brightness-110 active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer group"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 size={16} className="animate-spin" />
              Authenticating...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              Enter Platform
              <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
            </span>
          )}
          {/* Shimmer overlay */}
          <span className="absolute inset-0 -translate-x-full animate-[shimmer_2s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        </button>

        <p className="text-center text-xs text-muted/60">
          New to FootballIQ?{" "}
          <button
            type="button"
            onClick={onSwitch}
            className="font-semibold text-primary hover:text-primary/80 transition-colors cursor-pointer"
          >
            Create Account
          </button>
        </p>
      </form>
    </motion.div>
  );
}

/* ── Sign Up View ──────────────────────────────────────── */
function SignUpView({
  onLogin,
  onSwitch,
}: {
  onLogin: (name: string) => void;
  onSwitch: () => void;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [position, setPosition] = useState<Position | null>(null);
  const [foot, setFoot] = useState<Foot | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = name && email && position && foot;

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!canSubmit) return;
      setLoading(true);
      setTimeout(() => {
        onLogin(name);
      }, 1000);
    },
    [canSubmit, name, onLogin],
  );

  return (
    <motion.div
      initial={{ opacity: 0, transform: "translateX(24px)" }}
      animate={{ opacity: 1, transform: "translateX(0)" }}
      exit={{ opacity: 0, transform: "translateX(-24px)" }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <form onSubmit={handleSubmit} className="space-y-5">
        <AuthInput
          label="Full Name"
          type="text"
          value={name}
          onChange={setName}
          icon={<Shield size={16} />}
          placeholder="Alex Morgan"
          autoComplete="name"
        />
        <AuthInput
          label="Athlete Email"
          type="email"
          value={email}
          onChange={setEmail}
          icon={<Mail size={16} />}
          placeholder="alex@example.com"
          autoComplete="email"
        />

        {/* Position Toggle Grid */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-muted">
            Primary Position
          </label>
          <div className="grid grid-cols-2 gap-2" role="group" aria-label="Select primary position">
            {POSITIONS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => setPosition(p.id)}
                aria-pressed={position === p.id}
                className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-xs font-semibold transition-all duration-200 cursor-pointer ${
                  position === p.id
                    ? "border-primary bg-primary/12 text-primary shadow-[0_0_12px_rgba(0,255,102,0.06)]"
                    : "border-border bg-background/60 text-muted hover:border-border/70 hover:text-foreground"
                }`}
              >
                <span className={position === p.id ? "text-primary" : "text-muted/60"}>
                  {p.icon}
                </span>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Preferred Foot Selector */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-muted">
            Preferred Foot
          </label>
          <div className="flex gap-2" role="group" aria-label="Select preferred foot">
            {FOOT_OPTIONS.map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => setFoot(f.id)}
                aria-pressed={foot === f.id}
                className={`flex-1 rounded-xl border px-3 py-3 text-xs font-semibold transition-all duration-200 cursor-pointer ${
                  foot === f.id
                    ? "border-primary bg-primary/12 text-primary shadow-[0_0_12px_rgba(0,255,102,0.06)]"
                    : "border-border bg-background/60 text-muted hover:border-border/70 hover:text-foreground"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !canSubmit}
          className="relative w-full overflow-hidden rounded-xl bg-primary px-6 py-3.5 text-sm font-bold text-on-primary transition-all duration-200 hover:brightness-110 active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer group"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 size={16} className="animate-spin" />
              Generating Profile...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              Generate Player Profile
              <ChevronRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
            </span>
          )}
          <span className="absolute inset-0 -translate-x-full animate-[shimmer_2s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        </button>

        <p className="text-center text-xs text-muted/60">
          Already have an account?{" "}
          <button
            type="button"
            onClick={onSwitch}
            className="font-semibold text-primary hover:text-primary/80 transition-colors cursor-pointer"
          >
            Sign In
          </button>
        </p>
      </form>
    </motion.div>
  );
}

/* ── Main AuthScreen ──────────────────────────────────── */
export default function AuthScreen({ onLogin }: AuthScreenProps) {
  const [view, setView] = useState<AuthView>("signin");
  const direction = view === "signin" ? -1 : 1;

  return (
    <div className="flex h-screen bg-[#0B0F12] overflow-hidden">
      {/* Left — Brand */}
      <BrandPanel />

      {/* Right — Auth Hub */}
      <div className="flex h-full w-full lg:w-[55%] items-center justify-center p-6 sm:p-10">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="w-full max-w-md"
        >
          {/* Mobile logo (shown only on small screens) */}
          <div className="flex items-center justify-center gap-2.5 mb-8 lg:hidden">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/15 ring-1 ring-primary/30">
              <span className="text-base font-black text-primary">IQ</span>
            </div>
            <span className="text-xl font-black tracking-tight text-foreground">
              Football<span className="text-primary">IQ</span>
            </span>
          </div>

          {/* Glass container */}
          <div className="rounded-3xl border border-border/60 bg-surface/60 backdrop-blur-xl p-6 sm:p-8 shadow-2xl shadow-black/50">
            {/* Toggle header */}
            <div className="flex items-center justify-between mb-7">
              <div>
                <h2 className="font-heading text-xl font-bold tracking-tight text-foreground">
                  {view === "signin" ? "Welcome Back" : "Join the Ecosystem"}
                </h2>
                <p className="mt-1 text-xs text-muted">
                  {view === "signin"
                    ? "Sign in to your player profile"
                    : "Set up your personal coaching calibration"}
                </p>
              </div>
              <div className="flex rounded-xl border border-border bg-background/60 p-0.5">
                <button
                  type="button"
                  onClick={() => setView("signin")}
                  aria-pressed={view === "signin"}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-200 cursor-pointer ${
                    view === "signin"
                      ? "bg-primary/15 text-primary shadow-sm"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => setView("signup")}
                  aria-pressed={view === "signup"}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-200 cursor-pointer ${
                    view === "signup"
                      ? "bg-primary/15 text-primary shadow-sm"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  Sign Up
                </button>
              </div>
            </div>

            {/* Auth forms with AnimatePresence */}
            <AnimatePresence mode="wait" initial={false}>
              {view === "signin" ? (
                <SignInView
                  key="signin"
                  onLogin={onLogin}
                  onSwitch={() => setView("signup")}
                />
              ) : (
                <SignUpView
                  key="signup"
                  onLogin={onLogin}
                  onSwitch={() => setView("signin")}
                />
              )}
            </AnimatePresence>
          </div>

          {/* Footer */}
          <p className="mt-6 text-center text-[10px] text-muted/40">
            &copy; 2025 FootballIQ &middot; Elite Performance Analytics &middot; All rights reserved
          </p>
        </motion.div>
      </div>
    </div>
  );
}
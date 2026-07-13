import { Swords } from "lucide-react";

interface LogoProps {
  className?: string;
  showText?: boolean;
  size?: "sm" | "md" | "lg";
}

const sizeMap = {
  sm: { icon: 20, text: "text-lg" },
  md: { icon: 28, text: "text-2xl" },
  lg: { icon: 36, text: "text-3xl" },
};

export default function Logo({ className = "", showText = true, size = "md" }: LogoProps) {
  const s = sizeMap[size];

  return (
    <a href="/" className={`inline-flex items-center gap-2 no-underline group ${className}`}>
      <div className="relative flex items-center justify-center">
        <Swords
          size={s.icon}
          className="text-primary transition-transform duration-200 group-hover:scale-105"
          strokeWidth={2.2}
        />
      </div>
      {showText && (
        <span
          className={`font-heading font-bold tracking-tight text-foreground ${s.text}`}
        >
          Football<span className="text-primary">IQ</span>
        </span>
      )}
    </a>
  );
}
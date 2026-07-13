import { Settings, Shield, Bell, Monitor, User } from "lucide-react";

const settingsSections = [
  {
    icon: User,
    title: "Profile",
    description: "Account details, display name, and role preferences",
  },
  {
    icon: Bell,
    title: "Notifications",
    description: "Configure alerts for analysis completion and team updates",
  },
  {
    icon: Shield,
    title: "Privacy & Data",
    description: "Video storage retention, data sharing, and deletion policies",
  },
  {
    icon: Monitor,
    title: "Display",
    description: "Theme, language, and visualization preferences",
  },
];

export default function SettingsTab() {
  return (
    <div className="mx-auto max-w-3xl animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight text-foreground">
          Settings
        </h1>
        <p className="mt-1 text-sm text-muted">
          Manage your account and application preferences
        </p>
      </div>

      {/* Settings cards */}
      <div className="space-y-3">
        {settingsSections.map((section) => {
          const Icon = section.icon;
          return (
            <button
              key={section.title}
              type="button"
              className="group flex w-full items-center gap-4 rounded-xl border border-border bg-surface p-4 text-left transition-all duration-200 hover:border-primary/30 hover:bg-surface-hover cursor-pointer active:scale-[0.99]"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary transition-colors group-hover:bg-primary/15">
                <Icon size={20} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-foreground">
                  {section.title}
                </p>
                <p className="mt-0.5 text-xs text-muted">
                  {section.description}
                </p>
              </div>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="shrink-0 text-muted transition-colors group-hover:text-foreground"
              >
                <path d="m9 18 6-6-6-6" />
              </svg>
            </button>
          );
        })}
      </div>
    </div>
  );
}
import { type NavTab } from "../types";

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  collapsed: boolean;
  onToggle: () => void;
}

const navItems: { id: NavTab; label: string; icon: JSX.Element }[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <rect width="7" height="9" x="3" y="3" rx="1" />
        <rect width="7" height="5" x="14" y="3" rx="1" />
        <rect width="7" height="9" x="14" y="12" rx="1" />
        <rect width="7" height="5" x="3" y="16" rx="1" />
      </svg>
    ),
  },
  {
    id: "upload",
    label: "AI Analyzer",
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v4" />
        <path d="M12 18v4" />
        <path d="M2 12h4" />
        <path d="M18 12h4" />
      </svg>
    ),
  },
  {
    id: "analysis",
    label: "Performance",
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  },
  {
    id: "settings",
    label: "Settings",
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v2" />
        <path d="M12 21v2" />
        <path d="M4.22 4.22l1.42 1.42" />
        <path d="M18.36 18.36l1.42 1.42" />
        <path d="M1 12h2" />
        <path d="M21 12h2" />
        <path d="M4.22 19.78l1.42-1.42" />
        <path d="M18.36 5.64l1.42-1.42" />
      </svg>
    ),
  },
];

export default function Sidebar({
  activeTab,
  onTabChange,
  collapsed,
  onToggle,
}: SidebarProps) {
  return (
    <aside
      className={`relative flex flex-col border-r border-border bg-surface transition-all duration-300 ${
        collapsed ? "w-[68px]" : "w-[220px]"
      }`}
    >
      {/* Brand */}
      <div className="flex h-16 shrink-0 items-center border-b border-border px-4">
        {collapsed ? (
          <div className="flex w-full justify-center">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <span className="text-base font-bold text-primary">IQ</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <span className="text-base font-bold text-primary">IQ</span>
            </div>
            <span className="text-lg font-bold tracking-tight text-foreground">
              Football<span className="text-primary">IQ</span>
            </span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-3">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <div className="relative">
              {isActive && !collapsed && (
                <div className="absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-r-full bg-primary shadow-[0_0_8px_theme(colors.primary.DEFAULT)]" />
              )}
              <button
                key={item.id}
                type="button"
                onClick={() => onTabChange(item.id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200 cursor-pointer
                  ${
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted hover:bg-surface-hover hover:text-foreground"
                  }
                  ${collapsed ? "justify-center px-0" : "pl-5"}
                `}
                aria-label={item.label}
                title={collapsed ? item.label : undefined}
              >
                <span
                  className={`shrink-0 transition-all duration-200 ${
                    isActive ? "scale-110" : ""
                  }`}
                >
                  {item.icon}
                </span>
                {!collapsed && <span>{item.label}</span>}
              </button>
            </div>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <div className="border-t border-border p-3">
        <button
          type="button"
          onClick={onToggle}
          className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-muted transition-all duration-200 hover:bg-surface-hover hover:text-foreground cursor-pointer ${
            collapsed ? "justify-center px-0" : ""
          }`}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform duration-300 ${
              collapsed ? "rotate-180" : ""
            }`}
          >
            <path d="m9 18 6-6-6-6" />
          </svg>
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
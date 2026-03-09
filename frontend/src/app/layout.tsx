"use client";
import { Inter } from "next/font/google";
import Link from "next/link";
import { usePathname } from "next/navigation";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

const NAV = [
  { href: "/", label: "Studio", icon: "⚙" },
  { href: "/chat", label: "Chat", icon: "💬" },
  { href: "/monitoring", label: "Monitoring", icon: "📊" },
  { href: "/evaluation", label: "Evaluation", icon: "🧪" },
  { href: "/traces", label: "Traces", icon: "🔍" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen flex`}>
        {/* Sidebar */}
        <aside className="w-56 border-r border-[var(--border)] bg-[var(--bg-card)] flex flex-col fixed h-screen">
          <div className="p-5 border-b border-[var(--border)]">
            <div className="text-base font-semibold text-white tracking-tight">SDLC Agent</div>
            <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Multi-Agent Platform</div>
          </div>
          <nav className="flex-1 p-3 space-y-0.5">
            {NAV.map((item) => {
              const active = pathname === item.href;
              return (
                <Link key={item.href} href={item.href}
                  className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] transition-all ${
                    active
                      ? "bg-[var(--accent)]/10 text-[var(--accent)] font-medium"
                      : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)]"
                  }`}>
                  <span className="text-sm">{item.icon}</span>
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="p-4 border-t border-[var(--border)] text-[11px] text-[var(--text-muted)]">
            LangGraph + MCP
          </div>
        </aside>

        {/* Main */}
        <main className="ml-56 flex-1 p-8 min-h-screen">{children}</main>
      </body>
    </html>
  );
}

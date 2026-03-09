"use client";
import { Inter } from "next/font/google";
import Link from "next/link";
import { usePathname } from "next/navigation";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

const NAV = [
  { href: "/", label: "Studio" },
  { href: "/chat", label: "Chat" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/traces", label: "Traces" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen flex`}>
        <aside className="w-52 border-r border-[var(--border)] bg-[var(--bg-sidebar)] flex flex-col fixed h-screen">
          <div className="px-5 py-4 border-b border-[var(--border)]">
            <div className="text-sm font-semibold text-[var(--text)] tracking-tight">SDLC Agent</div>
            <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Multi-Agent Platform</div>
          </div>
          <nav className="flex-1 p-2.5 space-y-0.5">
            {NAV.map((item) => {
              const active = pathname === item.href;
              return (
                <Link key={item.href} href={item.href}
                  className={`block px-3 py-[7px] rounded-md text-[13px] transition-all ${
                    active
                      ? "bg-[var(--accent-light)] text-[var(--accent)] font-medium"
                      : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)]"
                  }`}>
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="p-4 border-t border-[var(--border)] text-[11px] text-[var(--text-muted)]">
            LangGraph + MCP
          </div>
        </aside>
        <main className="ml-52 flex-1 p-7 min-h-screen">{children}</main>
      </body>
    </html>
  );
}

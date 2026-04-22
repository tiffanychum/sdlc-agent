"use client";
import { Inter } from "next/font/google";
import Link from "next/link";
import { usePathname } from "next/navigation";
import "./globals.css";
import { RegressionRunProvider } from "@/contexts/RegressionRunContext";
import { TeamProvider } from "@/contexts/TeamContext";
import { ChatSessionProvider } from "@/contexts/ChatSessionContext";
import RegressionRunWidget from "@/components/RegressionRunWidget";
import ChatLiveBadge from "@/components/ChatLiveBadge";

const inter = Inter({ subsets: ["latin"] });

const NAV = [
  { href: "/", label: "Studio" },
  { href: "/chat", label: "Chat" },
  { href: "/rag", label: "RAG" },
  { href: "/workflow", label: "Workflow" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/regression", label: "Regression" },
  { href: "/evaluation", label: "Evaluation" },
];

function NavBar() {
  const pathname = usePathname();
  return (
    <aside className="w-52 border-r border-[var(--border)] bg-white flex flex-col fixed h-screen">
      <div className="px-5 py-5 border-b border-[var(--border)]">
        <div className="text-[13px] font-semibold text-[var(--text)] tracking-tight">SDLC Agent</div>
        <div className="text-[11px] text-[var(--text-muted)] mt-0.5">Multi-Agent Platform</div>
      </div>
      <nav className="flex-1 p-3 space-y-px">
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center justify-between px-3 py-2 rounded-lg text-[13px] font-medium transition-all ${
                active
                  ? "bg-[var(--text)] text-white"
                  : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)]"
              }`}
            >
              <span>{item.label}</span>
              {/* C4: live-chat indicator dot surfaces on the /chat nav item
                  whenever ANY team is streaming, so users on other pages
                  know a background chat is still running. */}
              {item.href === "/chat" && <ChatLiveBadge showCount />}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-[var(--border)] text-[11px] text-[var(--text-muted)]">
        LangGraph · MCP
      </div>
    </aside>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen flex`}>
        <TeamProvider>
          <RegressionRunProvider>
            <ChatSessionProvider>
              <NavBar />
              <main className="ml-52 flex-1 p-7 min-h-screen bg-[var(--bg)]">{children}</main>
              {/* Floating widget — persists across all pages */}
              <RegressionRunWidget />
            </ChatSessionProvider>
          </RegressionRunProvider>
        </TeamProvider>
      </body>
    </html>
  );
}

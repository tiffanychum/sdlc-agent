"use client";

/**
 * ChatLiveBadge — tiny pulsing dot surfaced in the NavBar and the team
 * switcher when at least one team has an active chat stream.  Part of
 * C4: cross-team visibility so the user can see that /chat on team A
 * is still running while they browse /regression on team B.
 */

import { useChatSession } from "@/contexts/ChatSessionContext";

interface Props {
  /** If set, only light up when THIS team is running (for inline team-switcher
   * options).  Otherwise light up whenever ANY team has a live stream. */
  teamId?: string;
  /** Extra class override. */
  className?: string;
  /** Show a small numeric count of running teams (nav-only). */
  showCount?: boolean;
}

export default function ChatLiveBadge({ teamId, className, showCount }: Props) {
  const { runningTeamIds, isTeamRunning } = useChatSession();
  const visible = teamId ? isTeamRunning(teamId) : runningTeamIds.length > 0;
  if (!visible) return null;

  return (
    <span
      className={`inline-flex items-center gap-1 ${className || ""}`}
      title={teamId
        ? "This team has a live chat stream"
        : `Live streams running on ${runningTeamIds.length} team(s): ${runningTeamIds.join(", ")}`}
    >
      <span className="relative inline-flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-60 animate-ping" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
      </span>
      {showCount && !teamId && runningTeamIds.length > 1 && (
        <span className="text-[10px] text-emerald-600 font-semibold">
          {runningTeamIds.length}
        </span>
      )}
    </span>
  );
}

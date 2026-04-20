"use client";

/**
 * TeamContext — global, persisted team selection.
 *
 * Problem this solves:
 *   Each page (regression / chat / monitoring / …) used to own its own
 *   `teamId` state, defaulted to the first team returned by /api/teams.
 *   Switching tabs always reset the selection to "default" which made the
 *   team dropdown feel broken and — more seriously — caused regression
 *   history filtered by team to "disappear" when the user navigated away
 *   and back.
 *
 * What this provides:
 *   - A single `teamId` shared by every page.
 *   - Persistence via localStorage so the selection survives reloads and
 *     page navigation.
 *   - The latest team list (so every page can render the dropdown without
 *     re-fetching /api/teams on every mount).
 *   - A `refreshTeams()` helper for Studio to call after it creates / renames
 *     a team, keeping every other open page in sync.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { api } from "@/lib/api";

export interface TeamSummary {
  id: string;
  name: string;
  description?: string;
  decision_strategy?: string;
  agents_count?: number;
  created_at?: string | null;
}

interface TeamContextValue {
  teamId: string;
  setTeamId: (id: string) => void;
  teams: TeamSummary[];
  selectedTeam: TeamSummary | undefined;
  loading: boolean;
  refreshTeams: () => Promise<TeamSummary[]>;
}

const TeamContext = createContext<TeamContextValue | null>(null);

const STORAGE_KEY = "sdlc.selectedTeamId";

function readStoredTeamId(): string {
  if (typeof window === "undefined") return "";
  try { return window.localStorage.getItem(STORAGE_KEY) || ""; } catch { return ""; }
}

function writeStoredTeamId(id: string) {
  if (typeof window === "undefined") return;
  try {
    if (id) window.localStorage.setItem(STORAGE_KEY, id);
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch { /* quota / private mode — non-fatal */ }
}

export function useTeam(): TeamContextValue {
  const ctx = useContext(TeamContext);
  if (!ctx) throw new Error("useTeam must be used inside <TeamProvider>");
  return ctx;
}

export function TeamProvider({ children }: { children: React.ReactNode }) {
  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [teamId, setTeamIdState] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const didInit = useRef(false);

  const setTeamId = useCallback((id: string) => {
    setTeamIdState(id);
    writeStoredTeamId(id);
  }, []);

  const refreshTeams = useCallback(async (): Promise<TeamSummary[]> => {
    try {
      const list: TeamSummary[] = await api.teams.list();
      setTeams(list);
      return list;
    } catch {
      setTeams([]);
      return [];
    }
  }, []);

  // First-load bootstrap — fetch team list AND pick up any stored selection.
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    let cancelled = false;
    (async () => {
      const list = await refreshTeams();
      if (cancelled) return;
      const stored = readStoredTeamId();
      if (stored && list.some(t => t.id === stored)) {
        setTeamIdState(stored);
      } else if (list.length > 0) {
        setTeamIdState(list[0].id);
        writeStoredTeamId(list[0].id);
      }
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [refreshTeams]);

  // If the currently-selected team disappears from the list (deleted in Studio),
  // fall back to the first remaining team so the UI never sits on a stale id.
  useEffect(() => {
    if (!teamId || teams.length === 0) return;
    if (!teams.some(t => t.id === teamId)) {
      const fallback = teams[0].id;
      setTeamIdState(fallback);
      writeStoredTeamId(fallback);
    }
  }, [teams, teamId]);

  // Sync localStorage changes from OTHER tabs (so switching team in one tab
  // propagates to every other open tab).
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key !== STORAGE_KEY) return;
      const next = e.newValue || "";
      if (next && next !== teamId) setTeamIdState(next);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [teamId]);

  const selectedTeam = useMemo(
    () => teams.find(t => t.id === teamId),
    [teams, teamId],
  );

  const value: TeamContextValue = {
    teamId, setTeamId, teams, selectedTeam, loading, refreshTeams,
  };

  return <TeamContext.Provider value={value}>{children}</TeamContext.Provider>;
}

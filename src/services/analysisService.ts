import { supabase } from "../lib/supabase";

export interface AnalysisStep {
  id: string;
  label: string;
  duration: number; // ms to simulate
}

export const ANALYSIS_STEPS: AnalysisStep[] = [
  { id: "detecting_players", label: "Detecting players on the pitch…", duration: 3000 },
  { id: "tracking_movements", label: "Tracking player movements…", duration: 3500 },
  { id: "building_passes", label: "Building pass maps…", duration: 3000 },
  { id: "generating_heatmaps", label: "Generating heatmaps…", duration: 3000 },
  { id: "calculating_ratings", label: "Calculating player ratings…", duration: 2500 },
];

type ProgressCallback = (stepIndex: number, step: AnalysisStep) => void;

const PLAYER_NAMES_HOME = [
  "Alisson Becker", "Trent Alexander-Arnold", "Virgil van Dijk", "Ibrahima Konaté",
  "Andy Robertson", "Dominik Szoboszlai", "Alexis Mac Allister", "Ryan Gravenberch",
  "Mohamed Salah", "Darwin Núñez", "Luis Díaz",
];

const PLAYER_NAMES_AWAY = [
  "Ederson", "Kyle Walker", "Rúben Dias", "John Stones",
  "Joško Gvardiol", "Rodri", "Kevin De Bruyne", "Bernardo Silva",
  "Phil Foden", "Erling Haaland", "Jack Grealish",
];

function randomBetween(min: number, max: number): number {
  return Math.round((Math.random() * (max - min) + min) * 10) / 10;
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function generatePasses(playerIds: string[], matchId: string) {
  const passes: Array<{
    match_id: string;
    player_id: string;
    x: number;
    y: number;
    end_x: number;
    end_y: number;
    minute: number;
    completed: boolean;
  }> = [];

  // Generate ~60-100 passes
  const passCount = randomInt(60, 100);
  for (let i = 0; i < passCount; i++) {
    const playerId = playerIds[randomInt(0, playerIds.length - 1)];
    const x = randomBetween(5, 95);
    const y = randomBetween(5, 95);
    const distance = randomBetween(5, 40);
    const angle = Math.random() * Math.PI * 2;
    const endX = Math.max(0, Math.min(100, x + Math.cos(angle) * distance));
    const endY = Math.max(0, Math.min(100, y + Math.sin(angle) * distance));
    const completed = Math.random() > 0.25; // ~75% completion rate

    passes.push({
      match_id: matchId,
      player_id: playerId,
      x: Math.round(x * 10) / 10,
      y: Math.round(y * 10) / 10,
      end_x: Math.round(endX * 10) / 10,
      end_y: Math.round(endY * 10) / 10,
      minute: randomInt(1, 90),
      completed,
    });
  }

  return passes;
}

function generatePositions(playerIds: string[], matchId: string) {
  const positions: Array<{
    match_id: string;
    player_id: string;
    x: number;
    y: number;
    minute: number;
  }> = [];

  // Generate position samples for each player across the match
  for (const playerId of playerIds) {
    // ~12-20 position samples per player
    const samples = randomInt(12, 20);
    for (let i = 0; i < samples; i++) {
      // Cluster positions according to rough area (defenders more in own half, attackers more in opponent half)
      const isDefender = playerIds.indexOf(playerId) < 4;
      const isForward = playerIds.indexOf(playerId) >= 9;
      let x: number;
      if (isDefender) {
        x = randomBetween(5, 45);
      } else if (isForward) {
        x = randomBetween(55, 95);
      } else {
        x = randomBetween(20, 80);
      }
      const y = randomBetween(5, 95);
      const minute = randomInt(1, 90);

      positions.push({
        match_id: matchId,
        player_id: playerId,
        x: Math.round(x * 10) / 10,
        y: Math.round(y * 10) / 10,
        minute,
      });
    }
  }

  return positions;
}

function generateRatings(playerIds: string[], matchId: string) {
  return playerIds.map((playerId) => ({
    match_id: matchId,
    player_id: playerId,
    overall: Math.round(randomBetween(5.0, 9.5) * 10) / 10,
    passing: Math.round(randomBetween(4.5, 9.5) * 10) / 10,
    positioning: Math.round(randomBetween(4.5, 9.5) * 10) / 10,
    defending: Math.round(randomBetween(4.0, 9.0) * 10) / 10,
  }));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Runs the full analysis pipeline for a match.
 * For MVP this generates mock data; replace with external API calls for production.
 *
 * @param matchId - The match record ID in Supabase
 * @param onProgress - Optional callback fired after each step completes
 */
export async function runAnalysis(
  matchId: string,
  onProgress?: ProgressCallback
): Promise<void> {
  for (let i = 0; i < ANALYSIS_STEPS.length; i++) {
    const step = ANALYSIS_STEPS[i];

    if (onProgress) {
      onProgress(i, step);
    }

    // Simulate step duration
    await delay(step.duration);

    // Perform the actual work for this step
    switch (step.id) {
      case "detecting_players": {
        // Create players for home and away teams
        const homePlayers = PLAYER_NAMES_HOME.map((name, idx) => ({
          match_id: matchId,
          name,
          jersey_number: idx + 1,
          team: "home" as const,
        }));
        const awayPlayers = PLAYER_NAMES_AWAY.map((name, idx) => ({
          match_id: matchId,
          name,
          jersey_number: idx + 1,
          team: "away" as const,
        }));

        const { error: playersError } = await supabase
          .from("players")
          .insert([...homePlayers, ...awayPlayers]);

        if (playersError) throw new Error(`Failed to create players: ${playersError.message}`);
        break;
      }

      case "building_passes": {
        // Fetch saved players to get their IDs
        const { data: players, error: fetchError } = await supabase
          .from("players")
          .select("id")
          .eq("match_id", matchId);

        if (fetchError || !players) {
          throw new Error(`Failed to fetch players: ${fetchError?.message}`);
        }

        const playerIds = players.map((p) => p.id);
        const passes = generatePasses(playerIds, matchId);

        // Insert in batches of 20 to avoid overwhelming the DB
        for (let j = 0; j < passes.length; j += 20) {
          const batch = passes.slice(j, j + 20);
          const { error: passesError } = await supabase
            .from("passes")
            .insert(batch);

          if (passesError) throw new Error(`Failed to save passes: ${passesError.message}`);
        }
        break;
      }

      case "generating_heatmaps": {
        const { data: players, error: fetchError } = await supabase
          .from("players")
          .select("id")
          .eq("match_id", matchId);

        if (fetchError || !players) {
          throw new Error(`Failed to fetch players: ${fetchError?.message}`);
        }

        const playerIds = players.map((p) => p.id);
        const positions = generatePositions(playerIds, matchId);

        // Insert in batches
        for (let j = 0; j < positions.length; j += 30) {
          const batch = positions.slice(j, j + 30);
          const { error: posError } = await supabase
            .from("positions")
            .insert(batch);

          if (posError) throw new Error(`Failed to save positions: ${posError.message}`);
        }
        break;
      }

      case "calculating_ratings": {
        const { data: players, error: fetchError } = await supabase
          .from("players")
          .select("id")
          .eq("match_id", matchId);

        if (fetchError || !players) {
          throw new Error(`Failed to fetch players: ${fetchError?.message}`);
        }

        const playerIds = players.map((p) => p.id);
        const ratings = generateRatings(playerIds, matchId);

        const { error: ratingsError } = await supabase
          .from("ratings")
          .insert(ratings);

        if (ratingsError) throw new Error(`Failed to save ratings: ${ratingsError.message}`);
        break;
      }
    }
  }

  // Mark match as complete
  const { error: updateError } = await supabase
    .from("matches")
    .update({ status: "complete" })
    .eq("id", matchId);

  if (updateError) {
    throw new Error(`Failed to update match status: ${updateError.message}`);
  }
}

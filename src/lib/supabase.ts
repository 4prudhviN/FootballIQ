import { createClient } from "@supabase/supabase-js";

const supabaseUrl = "https://bbzgdblrrcemdzmwbpsa.supabase.co";
const supabaseAnonKey =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJiemdkYmxycmNlbWR6bXdicHNhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM4ODAzNjksImV4cCI6MjA5OTQ1NjM2OX0.tfK7G3T8M812m6V9e9GbzDl5kRft97IAJrVCChnfgkk";

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type MatchStatus = "uploading" | "processing" | "complete" | "error";

export interface Match {
  id: string;
  title: string;
  video_url: string | null;
  status: MatchStatus;
  error_message: string | null;
  created_at: string;
}

export interface Player {
  id: string;
  match_id: string;
  name: string;
  jersey_number: number | null;
}

export interface Pass {
  id: string;
  match_id: string;
  player_id: string | null;
  x_start: number;
  y_start: number;
  x_end: number;
  y_end: number;
  success: boolean;
}

export interface Position {
  id: string;
  match_id: string;
  player_id: string;
  timestamp: number;
  x: number;
  y: number;
}

export interface Rating {
  id: string;
  match_id: string;
  player_id: string;
  rating_type: string;
  score: number;
}
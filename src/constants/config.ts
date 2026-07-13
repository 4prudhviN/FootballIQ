export const CONFIG = {
  supabase: {
    storageBucket: "videos",
  },
  MAX_FILE_SIZE: 500 * 1024 * 1024, // 500MB
  ACCEPTED_MIME_TYPES: ["video/mp4", "video/quicktime"],
  ACCEPTED_EXTENSIONS: [".mp4", ".mov"],
} as const;
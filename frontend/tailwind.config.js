/**
 * Tailwind config — "Bank-grade" palette for the Student Dropout EWS.
 *
 * Aesthetic anchors:
 *   - Surface  : Deep Slate (#0f172a) backgrounds, layered with #111827 / #1e293b
 *   - Brand    : Indigo Primary (#4f46e5) for actions and headings accents
 *   - Status   : Emerald (Graduate)  /  Amber (Enrolled)  /  Rose (Dropout)
 *   - Type     : Inter / Geist sans, strict scale (12 / 13 / 14 / 16 / 20 / 24 / 32)
 *
 * Components consume these via semantic class names like
 * `bg-surface-900`, `text-status-dropout`, `ring-brand-500/30`.
 */
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,ts}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Surfaces — bank-portal canvas
        surface: {
          950: "#0b1220",
          900: "#0f172a", // primary canvas (Deep Slate)
          850: "#111827",
          800: "#1e293b",
          700: "#273449",
          600: "#334155",
          500: "#475569",
          400: "#64748b",
          300: "#94a3b8",
          200: "#cbd5e1",
          100: "#e2e8f0",
          50:  "#f1f5f9",
        },
        // Brand — indigo primary
        brand: {
          50:  "#eef2ff",
          100: "#e0e7ff",
          200: "#c7d2fe",
          300: "#a5b4fc",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5", // primary
          700: "#4338ca",
          800: "#3730a3",
          900: "#312e81",
        },
        // Status — semantic (Graduate / Enrolled / Dropout)
        status: {
          graduate: "#10b981", // emerald-500
          "graduate-bg": "rgba(16, 185, 129, 0.10)",
          "graduate-ring": "rgba(16, 185, 129, 0.35)",
          enrolled: "#f59e0b", // amber-500
          "enrolled-bg": "rgba(245, 158, 11, 0.10)",
          "enrolled-ring": "rgba(245, 158, 11, 0.35)",
          dropout: "#f43f5e", // rose-500
          "dropout-bg": "rgba(244, 63, 94, 0.10)",
          "dropout-ring": "rgba(244, 63, 94, 0.35)",
        },
        // Functional — info / success / warn / danger (separate from status)
        success: "#10b981",
        warning: "#f59e0b",
        danger:  "#ef4444",
        info:    "#3b82f6",
      },
      fontFamily: {
        sans: ['"Inter"', '"Geist Sans"', "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Geist Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        // Strict modular scale
        "2xs":  ["0.6875rem", { lineHeight: "1rem",     letterSpacing: "0.01em" }], // 11
        xs:     ["0.75rem",   { lineHeight: "1.125rem", letterSpacing: "0.005em" }], // 12
        sm:     ["0.8125rem", { lineHeight: "1.25rem"  }],                            // 13
        base:   ["0.875rem",  { lineHeight: "1.375rem" }],                            // 14 (UI default)
        md:     ["1rem",      { lineHeight: "1.5rem"   }],                            // 16
        lg:     ["1.125rem",  { lineHeight: "1.75rem"  }],                            // 18
        xl:     ["1.25rem",   { lineHeight: "1.875rem" }],                            // 20
        "2xl":  ["1.5rem",    { lineHeight: "2rem"     }],                            // 24
        "3xl":  ["2rem",      { lineHeight: "2.375rem", letterSpacing: "-0.01em" }],  // 32
        "4xl":  ["2.5rem",    { lineHeight: "2.875rem", letterSpacing: "-0.015em" }], // 40
      },
      letterSpacing: {
        tightest: "-0.02em",
      },
      borderRadius: {
        xl: "0.75rem",
        "2xl": "1rem",
      },
      boxShadow: {
        // Subtle, layered — matches financial-portal feel
        "card":     "0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 1px 2px rgba(0, 0, 0, 0.4)",
        "card-lg":  "0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 8px 24px rgba(0, 0, 0, 0.35)",
        "ring-brand":   "0 0 0 1px rgba(79, 70, 229, 0.35), 0 0 0 4px rgba(79, 70, 229, 0.18)",
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(to right, rgba(148,163,184,0.06) 1px, transparent 1px), " +
          "linear-gradient(to bottom, rgba(148,163,184,0.06) 1px, transparent 1px)",
        "panel-gradient":
          "linear-gradient(180deg, rgba(99,102,241,0.06) 0%, rgba(15,23,42,0) 60%)",
      },
      backgroundSize: {
        grid: "32px 32px",
      },
      keyframes: {
        shimmer: {
          "0%":   { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" },
        },
        "fade-in-up": {
          "0%":   { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.4s linear infinite",
        "fade-in-up": "fade-in-up 0.18s ease-out both",
      },
    },
  },
  plugins: [],
};

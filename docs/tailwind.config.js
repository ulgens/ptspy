/** @type {import('tailwindcss').Config} */
const defaultTheme = require("tailwindcss/defaultTheme");
const colors = require("tailwindcss/colors");

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    fontSize: {
      xs: ["0.75rem", { lineHeight: "1.5" }],
      sm: ["0.875rem", { lineHeight: "1.5" }],
      base: ["1rem", { lineHeight: "1.5" }],
      lg: ["1.125rem", { lineHeight: "1.2", fontWeight: 400 }],
      xl: ["1.25rem", { lineHeight: "1.2", fontWeight: 400 }],
      "2xl": [
        "1.5rem",
        { lineHeight: "1.2", fontWeight: 500, letterSpacing: "0.005rem" },
      ],
      "3xl": [
        "2.25rem",
        { lineHeight: "1.2", fontWeight: 500, letterSpacing: "0.01rem" },
      ],
      "4xl": [
        "3rem",
        { lineHeight: "1.2", fontWeight: 500, letterSpacing: "0.016rem" },
      ],
      "5xl": [
        "4rem",
        { lineHeight: "1.2", fontWeight: 500, letterSpacing: "0.016rem" },
      ],
      "6xl": [
        "5rem",
        { lineHeight: "1.2", fontWeight: 500, letterSpacing: "0.016rem" },
      ],
    },
    extend: {
      fontFamily: {
        sans: ['"Atkinson Hyperlegible Next"', ...defaultTheme.fontFamily.sans],
        mono: ['"Atkinson Hyperlegible Mono"', ...defaultTheme.fontFamily.mono],
      },
      keyframes: {
        gradient: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
      },
      animation: {
        gradient: "gradient 3s ease infinite",
      },
      colors: {
        gray: {
          50: "#fafafb",
          100: "#f3f3f5",
          200: "#e5e5e7",
          300: "#d1d1d6",
          400: "#9c9ca5",
          500: "#6b6b76",
          600: "#4b4b57",
          700: "#373745",
          800: "#1f1f27",
          900: "#111118",
          950: "#06060c",
        },
      },
    },
    typography: {
      DEFAULT: {
        // this is for prose class
        css: {
          h1: {
            fontWeight: 600,
            fontFamily: "Lexend",
            lineHeight: "1.2",
            letterSpacing: "0.01rem",
            fontSize: "2.75rem",
            marginTop: "2.75rem",
          },
          h2: {
            fontWeight: 500,
            fontFamily: "Lexend",
            letterSpacing: "0.01rem",
            fontSize: "2.25rem",
          },
          h3: {
            fontWeight: 400,
            fontFamily: "Lexend",
            letterSpacing: "0.01rem",
            fontSize: "1.5rem",
          },
          a: {
            textDecoration: "underline",
            "&:hover": {
              textDecoration: "underline",
            },
            "&:visited": {
              textDecoration: "underline",
              color: "#33a",
            },
          },
        },
      },
      sm: {
        css: {
          h1: {
            fontWeight: 500,
            fontFamily: "Lexend",
            lineHeight: "1.2",
            letterSpacing: "0.01rem",
            fontSize: "2rem",
            marginTop: "2rem",
          },
          h2: {
            fontWeight: 500,
            fontFamily: "Lexend",
            letterSpacing: "0.01rem",
            fontSize: "1.75rem",
          },
          h3: {
            fontWeight: 400,
            fontFamily: "Lexend",
            letterSpacing: "0.01rem",
            fontSize: "1.25rem",
          },
          a: {
            textDecoration: "underline",
            "&:hover": {
              textDecoration: "underline",
            },
          },
        },
      },
    },
  },
  plugins: [require("daisyui"), require("@tailwindcss/typography")],
  daisyui: {
    themes: [
      {
        light: {
          ...require("daisyui/src/theming/themes")["lofi"],
        },
      },
    ],
  },
};

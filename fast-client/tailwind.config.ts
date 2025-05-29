import { type Config } from "tailwindcss";
import defaultTheme from "tailwindcss/defaultTheme";

export default {
  darkMode: "class",
  content: [
    "./src/**/*.{ts,tsx}",
    "./node_modules/@/components/ui/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: { sans: ["var(--font-inter)", ...defaultTheme.fontFamily.sans] },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;

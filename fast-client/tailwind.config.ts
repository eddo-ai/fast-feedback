import { type Config } from "tailwindcss";
import { fontFamily } from "tailwindcss/defaultTheme";

export default {
  darkMode: ["class"],
  content: [
    "./src/**/*.{ts,tsx}",
    "./node_modules/@/components/ui/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: { sans: ["var(--font-inter)", ...fontFamily.sans] },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;

import { useEffect, useState } from "react";

export type ThemeType = "light" | "dark";

export const useSystemTheme = (): ThemeType => {
  const [theme, setTheme] = useState<ThemeType>("light");

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    setTheme(mediaQuery.matches ? "dark" : "light");

    const handler = (e: MediaQueryListEvent) => {
      setTheme(e.matches ? "dark" : "light");
    };

    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  return theme;
};

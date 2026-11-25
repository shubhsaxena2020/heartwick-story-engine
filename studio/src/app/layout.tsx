import type { Metadata, Viewport } from "next";
import { Big_Shoulders, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const shoulders = Big_Shoulders({ subsets: ["latin"], axes: ["opsz"], variable: "--font-shoulders", display: "swap" });
const jet = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jet", display: "swap" });

export const metadata: Metadata = {
  title: "Heartwick Studio",
  description: "Production console for the Heartwick Stories channel",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = { themeColor: "#0d0b0a", width: "device-width", initialScale: 1 };

// Motion (continuous animations) is on unless this browser switched it off; set before paint to avoid a flash.
const MOTION = `try{if(localStorage.getItem("hws_motion")!=="0")document.documentElement.classList.add("motion")}catch(e){document.documentElement.classList.add("motion")}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${shoulders.variable} ${jet.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: MOTION }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

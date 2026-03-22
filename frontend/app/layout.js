import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "AI That Knows When It's Wrong | Uncertainty Quantification",
  description:
    "Upload an image and see how a Deep Ensemble of neural networks quantifies its own uncertainty. Built with WideResNet-28-10 and CIFAR-10.",
  keywords: [
    "uncertainty quantification",
    "deep ensemble",
    "OOD detection",
    "CIFAR-10",
    "AI confidence",
  ],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

import Providers from "../components/layout/Providers";
import AppLayout from "../components/layout/AppLayout";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NeuroFlow — Production RAG Platform",
  description: "Next-generation RAG operations dashboard and playground.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <Providers>
          <AppLayout>{children}</AppLayout>
        </Providers>
      </body>
    </html>
  );
}

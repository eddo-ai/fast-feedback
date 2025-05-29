import "@/styles/globals.css";
import { Inter } from "next/font/google";
import { cn } from "@/lib/utils";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={cn("min-h-screen bg-gray-50 text-gray-900", inter.className)}
      >
        <main className="container mx-auto p-4">
          <nav className="mb-4 flex gap-4">
            <Link href="/">Home</Link>
            <Link href="/samples">Samples</Link>
            <Link href="/feedback-queue">Feedback Queue</Link>
          </nav>
          {children}
        </main>
      </body>
    </html>
  );
}

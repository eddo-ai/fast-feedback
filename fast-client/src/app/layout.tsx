import "@/styles/globals.css";
import { cn } from "@/lib/utils";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={cn("min-h-screen bg-gray-50 text-gray-900 font-sans")}> 
        <main className="container mx-auto p-4">{children}</main>
      </body>
    </html>
  );
}

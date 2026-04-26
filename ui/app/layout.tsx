import "./globals.css";

export const metadata = {
  title: "Medical Abstract Summarizer",
  description: "Compress a medical abstract into a 1-2 sentence summary with ROUGE-L.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

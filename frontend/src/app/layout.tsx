import "./globals.css";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";


export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
  <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body style={{ fontFamily: 'system-ui, sans-serif', margin: 0, padding: 16, background: '#0b0c10', color: '#e6edf3' }}>
        <div style={{ maxWidth: 900, margin: '0 auto' }}>
          <h1 style={{ fontSize: 28, marginBottom: 12 }}>AIditing (MVP)</h1>
          {children}
        </div>
      </body>
    </html>
  );
}


"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Page() {
  const [video, setVideo] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<string>("idle");
  const [message, setMessage] = useState<string>("");
  const [downloading, setDownloading] = useState(false);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    const vid = files.find((f) => f.type.includes("video")) || null;
    const aud = files.find((f) => f.type.includes("audio")) || null;
    if (vid) setVideo(vid);
    if (aud) setAudio(aud);
  }, []);

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
  };

  const canSubmit = !!(video && audio && !jobId);

  const submit = async () => {
    if (!video || !audio) return;
    const form = new FormData();
    form.append("video", video);
    form.append("audio", audio);
    const res = await fetch(`${API_BASE}/process`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    setJobId(data.job_id);
    setStatus("running");
    setProgress(1);
  };

  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(async () => {
      const res = await fetch(`${API_BASE}/status/${jobId}`);
      if (!res.ok) return;
      const d = await res.json();
      setProgress(d.progress ?? 0);
      setStatus(d.status ?? "");
      setMessage(d.message ?? "");
      if (d.status === "completed" || d.status === "failed") {
        clearInterval(t);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [jobId]);

  const download = async () => {
    if (!jobId) return;
    setDownloading(true);
    const res = await fetch(`${API_BASE}/result/${jobId}`);
    if (!res.ok) {
      setDownloading(false);
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "out.mp4";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setDownloading(false);
  };

  return (
    <div>
      <p style={{ opacity: 0.85, marginBottom: 16 }}>Glissez-déposez une vidéo (.mp4) et une piste audio propre (.wav). Pas d'IA générative texte, on coupe seulement. Cette démo produit un fichier mock.</p>

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        style={{
          border: dragOver ? "2px dashed #58a6ff" : "2px dashed #30363d",
          padding: 24,
          borderRadius: 12,
          background: dragOver ? "#161b22" : "#0d1117",
          marginBottom: 16,
        }}
      >
        <div style={{ marginBottom: 8 }}>Drag & drop ici</div>
        <div style={{ display: 'flex', gap: 8 }}>
          <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
          <input type="file" accept="audio/*" onChange={(e) => setAudio(e.target.files?.[0] ?? null)} />
        </div>
        <div style={{ marginTop: 8, fontSize: 14, opacity: 0.8 }}>
          {video ? `Vidéo: ${video.name}` : "Vidéo: (non sélectionné)"} · {audio ? `Audio: ${audio.name}` : "Audio: (non sélectionné)"}
        </div>
      </div>

      <button disabled={!canSubmit} onClick={submit} style={{ padding: '8px 14px', borderRadius: 8, background: canSubmit ? '#238636' : '#30363d', color: 'white', border: 'none', cursor: canSubmit ? 'pointer' : 'not-allowed' }}>
        Traiter
      </button>

      {jobId && (
        <div style={{ marginTop: 24 }}>
          <div style={{ marginBottom: 8 }}>Statut: {status} {message ? `— ${message}` : ""}</div>
          <div style={{ height: 12, background: '#30363d', borderRadius: 6, overflow: 'hidden', maxWidth: 480 }}>
            <div style={{ width: `${progress}%`, height: '100%', background: '#58a6ff' }} />
          </div>

          {status === "completed" && (
            <div style={{ marginTop: 16 }}>
              <button onClick={download} disabled={downloading} style={{ padding: '8px 14px', borderRadius: 8, background: '#1f6feb', color: 'white', border: 'none' }}>
                {downloading ? "Téléchargement..." : "Télécharger le résultat"}
              </button>
            </div>
          )}

          {status === "failed" && (
            <div style={{ marginTop: 16, color: '#ffa198' }}>Le traitement a échoué.</div>
          )}
        </div>
      )}

      <div style={{ marginTop: 32, fontSize: 13, opacity: 0.7 }}>
        Backend: {API_BASE}
      </div>
    </div>
  );
}


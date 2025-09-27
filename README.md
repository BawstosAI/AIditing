# AIditing (MVP scaffold)

Outil: uploader une vidéo et une piste audio propre, synchroniser, transcrire (FR), nettoyer (coupure de silences > 0,5 s et répétitions immédiates simples), puis appliquer des zooms automatiques sur 3–5 pics d’emphase. Cette version est une maquette fonctionnelle: le pipeline est mocké et renvoie un `out.mp4` bidon pour valider le flux front↔back.

## Arborescence

```
AIditing/
├─ server/
│  ├─ main.py
│  ├─ pipeline.py
│  ├─ jobs.py
│  ├─ storage.py
│  ├─ config.py
│  ├─ requirements.txt
│  └─ __init__.py
├─ frontend/
│  ├─ package.json
│  ├─ next.config.mjs
│  ├─ tsconfig.json
│  └─ app/
│     ├─ layout.tsx
│     └─ page.tsx
└─ samples/
   ├─ README.txt
   ├─ sample.mp4 (placeholder)
   └─ podcast.wav (placeholder)
```

## Backend (FastAPI)

Endpoints:
- `POST /process` → reçoit `video` + `audio` (multipart), crée un job et lance un traitement mocké, renvoie `job_id`.
- `GET /status/{job_id}` → progression et statut.
- `GET /result/{job_id}` → fichier final `out.mp4`.

Le pipeline réel sera branché dans `server/pipeline.py` avec ces fonctions:
- `extract_audio_from_video`
- `estimate_offset`
- `replace_audio`
- `transcribe_fr`
- `detect_cuts`
- `apply_cuts_and_zooms`

## Frontend (Next.js / React)

- Page unique avec drag & drop (1 vidéo + 1 audio), upload, barre de progression (polling), bouton de téléchargement du résultat.

## Prérequis

- Node.js 18+ et npm
- Python 3.10+
- FFmpeg installé et accessible dans le PATH (`ffmpeg -version`)

## Installation

1) Backend (dans un terminal)

```bash
# Toujours lancer depuis la racine du projet pour respecter le paquet "server"
cd AIditing

# Utiliser le Python de server/.venv (sans activer d'autre venv)
# Windows PowerShell
./server/.venv/Scripts/python.exe -m pip install --upgrade pip
./server/.venv/Scripts/python.exe -m pip install -r ./server/requirements.txt

# Démarrer FastAPI depuis la racine (note: module = server.main:app)
./server/.venv/Scripts/python.exe -m uvicorn server.main:app --reload --port 8000
```

2) Frontend (dans un autre terminal)

```bash
cd frontend
npm install
npm run dev
```

- Frontend: http://localhost:3000
- Backend:  http://localhost:8000/docs

### Dépannage Backend

- Erreur "Could not import module 'main'": lancé depuis la racine avec `main:app`. Corrige en `server.main:app`.
- Erreur "attempted relative import with no known parent package": lancé dans `server` avec `main:app`. Reviens à la racine et utilise `server.main:app`.
- Erreur "No module named 'server'": tu es déjà dans `server` mais t’essaies `server.main:app`. Reviens à la racine.
- Port 8000 occupé:
  - Windows PowerShell:
    ```powershell
    Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | % { Stop-Process -Id $_ -Force }
    ```
  - Relance ensuite la commande uvicorn ci-dessus.

## Test rapide (mock)

- Sur la page d’accueil, glissez `samples/sample.mp4` et `samples/podcast.wav` (placeholders).
- Cliquez "Traiter". La progression augmente jusqu’à 100%.
- Cliquez "Télécharger le résultat" pour récupérer un `out.mp4` factice généré par le backend.

## Justification des choix

- FastAPI: rapidité pour endpoints, BackgroundTasks, simplicité CORS.
- FFmpeg: standard robuste pour manipulation A/V, génération de mock vidéo.
- Pipeline modulaire: les fonctions sont stubées pour itérer facilement vers la vraie logique (MFCC/DTW, Whisper/WhisperX, etc.).
- Next.js (app router): UI minimaliste, drag & drop, polling simple.
- In-memory job store: suffisant pour MVP local; remplaçable par Redis/DB plus tard.

## Étapes futures (non incluses dans ce lot)

- Implémentation réelle: alignement MFCC/DTW + offset global, remplacement piste audio, Whisper/WhisperX pour transcription FR, détection silences >0,5 s et répétitions basiques, découpe/zoom via ffmpeg/moviepy.
- Persistance des jobs, stockage objet, authentification, UI timeline d’édition, sous-titres stylés.


# Darmstadt 3D Pipeline
KI-gestützte Pipeline zur Erstellung von 3D-Stadtmodellen aus LiDAR-Punktwolken.
Bachelorarbeit, TU Darmstadt, 2026.

## Einmalige Einrichtung

1. Python 3.12 installieren: https://www.python.org/downloads/
2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:
```bash
   py -3.12 -m venv venv
   venv\Scripts\activate.bat
   pip install -r requirements.txt
```
3. Datensätze in die richtigen Ordner legen:
   - `data/raw/fullLiWi_pointcloud.laz`
   - `data/sensaturban/train/*.ply`
   - `data/sensaturban/test/*.ply`

## Pipeline ausführen

Doppelklick auf `run_pipeline.bat` — aktiviert die venv automatisch und führt alle Schritte aus.

Einzelne Schritte überspringen:

run_pipeline.bat --skip-load --skip-preprocess


## Ausgaben

Alle Ergebnisse in `data/output/`:
- `*_LOD0/1/2.ply` — Klassenfarben für MeshLab
- `*_LOD0/1/2.glb` — Klassenfarben für Game Engine
- `*_LOD0/1/2_rgb.glb` — Echte RGB-Farben für Game Engine

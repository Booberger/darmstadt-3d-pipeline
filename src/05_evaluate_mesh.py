"""
05_evaluate_mesh.py
-------------------
Evaluation der rekonstruierten Meshes:
- Mesh-Vollstaendigkeit: Anteil der Eingabepunkte innerhalb Schwellenwert
- Mittlerer Abstand Punktwolke zu Mesh
- Maximaler Abstand (Hausdorff-naehrang)
- Polygonanzahl und Dateigrösse pro LOD-Stufe
"""

import open3d as o3d
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
PREPROC_FILE  = "data/processed/preprocessed.npy"
PRED_FILE     = "data/processed/darmstadt_pred.npy"
OUTPUT_DIR    = "data/output"

SCHWELLENWERT = 0.5   # Meter - Vollstaendigkeitsschwelle

KLASSEN = {
    0: "boden",
    1: "vegetation",
    2: "gebaeude",
    3: "strasse"
}

# ── Daten laden ──────────────────────────────────────────────────────────────
print("MESH-EVALUATION")
print("-" * 60)

preproc = np.load(PREPROC_FILE, allow_pickle=True).item()
pred    = np.load(PRED_FILE,    allow_pickle=True).item()

xyz    = preproc['points']
labels = pred['labels']

print(f"Eingabepunkte gesamt: {len(xyz):,}")

# ── Evaluation pro Klasse ────────────────────────────────────────────────────
print(f"\n{'Klasse':12s} | {'Punkte':>8s} | {'Vollst. <{:.2f}m'.format(SCHWELLENWERT):>14s} | {'Mittel (m)':>10s} | {'Max (m)':>8s} | {'LOD0 Dreiecke':>14s} | {'LOD1 Dreiecke':>14s} | {'LOD2 Dreiecke':>14s}")
print("-" * 110)

ergebnisse = {}

for klasse_id, name in KLASSEN.items():
    maske = labels == klasse_id
    n_punkte = int(np.sum(maske))

    if n_punkte < 10:
        print(f"{name:12s} | {'–':>8s} | {'–':>14s} | {'–':>10s} | {'–':>8s} | {'–':>14s} | {'–':>14s} | {'–':>14s}")
        continue

    # Mesh laden
    mesh_path = f"{OUTPUT_DIR}/{name}_LOD0.ply"
    if not os.path.exists(mesh_path):
        print(f"{name:12s} | Mesh nicht gefunden: {mesh_path}")
        continue

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Klassenpunkte als PointCloud
    pcd_klasse = o3d.geometry.PointCloud()
    pcd_klasse.points = o3d.utility.Vector3dVector(xyz[maske])

    # Abstand jedes Punktes zu naechstem Mesh-Vertex
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = mesh.vertices

    dist = np.asarray(pcd_klasse.compute_point_cloud_distance(mesh_pcd))

    vollstaendigkeit = np.mean(dist < SCHWELLENWERT) * 100
    mittel           = dist.mean()
    maximum          = dist.max()

    # LOD Dreiecke
    n_lod0 = len(mesh.triangles)
    n_lod1 = len(o3d.io.read_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD1.ply").triangles)
    n_lod2 = len(o3d.io.read_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD2.ply").triangles)

    ergebnisse[name] = {
        "punkte": n_punkte,
        "vollstaendigkeit": vollstaendigkeit,
        "mittel": mittel,
        "maximum": maximum,
        "lod0": n_lod0,
        "lod1": n_lod1,
        "lod2": n_lod2,
    }

    print(f"{name:12s} | {n_punkte:>8,} | {vollstaendigkeit:>13.1f}% | {mittel:>10.3f} | {maximum:>8.3f} | {n_lod0:>14,} | {n_lod1:>14,} | {n_lod2:>14,}")

# ── Gesamtauswertung ─────────────────────────────────────────────────────────
print("\n")
print("ZUSAMMENFASSUNG")
print("-" * 60)

if ergebnisse:
    gesamt_lod0 = sum(e["lod0"] for e in ergebnisse.values())
    gesamt_lod1 = sum(e["lod1"] for e in ergebnisse.values())
    gesamt_lod2 = sum(e["lod2"] for e in ergebnisse.values())
    gesamt_vol  = np.mean([e["vollstaendigkeit"] for e in ergebnisse.values()])
    gesamt_mit  = np.mean([e["mittel"] for e in ergebnisse.values()])

    print(f"\nGesamte Dreiecke LOD0: {gesamt_lod0:,}")
    print(f"Gesamte Dreiecke LOD1: {gesamt_lod1:,} ({gesamt_lod1/gesamt_lod0*100:.1f}%)")
    print(f"Gesamte Dreiecke LOD2: {gesamt_lod2:,} ({gesamt_lod2/gesamt_lod0*100:.1f}%)")
    print(f"\nMittlere Vollstaendigkeit (<{SCHWELLENWERT}m): {gesamt_vol:.1f}%")
    print(f"Mittlerer Abstand (alle Klassen): {gesamt_mit:.3f}m")

# ── Dateigroessen ─────────────────────────────────────────────────────────────
print("\n")
print("DATEIGROESSEN")
print("-" * 60)
print(f"\n{'Klasse':12s} | {'LOD0 PLY':>10s} | {'LOD1 PLY':>10s} | {'LOD2 PLY':>10s} | {'LOD0 GLB':>10s}")
print("-" * 60)

gesamt_groesse = 0
for name in KLASSEN.values():
    groessen = []
    for lod in ["LOD0", "LOD1", "LOD2"]:
        p = f"{OUTPUT_DIR}/{name}_{lod}.ply"
        groessen.append(os.path.getsize(p)/1e6 if os.path.exists(p) else 0)
    glb = f"{OUTPUT_DIR}/{name}_LOD0.glb"
    glb_g = os.path.getsize(glb)/1e6 if os.path.exists(glb) else 0
    gesamt_groesse += groessen[0]
    print(f"{name:12s} | {groessen[0]:>9.1f}MB | {groessen[1]:>9.1f}MB | {groessen[2]:>9.1f}MB | {glb_g:>9.1f}MB")

print(f"\nGesamtgrösse LOD0 (alle Klassen): {gesamt_groesse:.1f} MB")

print("\n")
print("FERTIG")
print("-" * 60)
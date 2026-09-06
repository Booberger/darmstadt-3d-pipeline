"""
02_preprocess.py
----------------
Schritt 2 der Pipeline: Vorverarbeitung der Punktwolke.
- Rauschfilterung (Statistical Outlier Removal)
- Normalisierung (Koordinaten zentrieren)
- Downsampling (Voxel-Grid-Filter)
- Speichern als numpy Array fuer naechste Schritte
"""

import laspy
import open3d as o3d
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_FILE  = "data/processed/ausschnitt_100m.laz"
OUTPUT_FILE = "data/processed/preprocessed.npy"

VOXEL_SIZE          = 0.25   # Meter - Downsampling-Aufloesung
OUTLIER_NEIGHBORS   = 20     # Nachbarn fuer Outlier-Erkennung
OUTLIER_STD_RATIO   = 2.0    # Standardabweichungs-Schwellenwert

# ── Schritt 1: LAZ einlesen ───────────────────────────────────────────────────
print("=" * 60)
print("SCHRITT 1: Punktwolke einlesen")
print("=" * 60)

print("Lese LAZ-Datei (chunkweise um RAM zu schonen)...")

# Chunkweise einlesen und als numpy sammeln
xyz_list = []
rgb_list = []
intensity_list = []

CHUNK_SIZE = 5_000_000

with laspy.open(INPUT_FILE) as f:
    total = f.header.point_count
    gelesen = 0
    for chunk in f.chunk_iterator(CHUNK_SIZE):
        xyz_list.append(np.column_stack([
            chunk.x.copy(),
            chunk.y.copy(),
            chunk.z.copy()
        ]))
        # RGB normalisieren (16-bit -> 0-1)
        if hasattr(chunk, 'red'):
            rgb_list.append(np.column_stack([
                chunk.red / 65535.0,
                chunk.green / 65535.0,
                chunk.blue / 65535.0
            ]))
        intensity_list.append(chunk.intensity.copy())
        gelesen += len(chunk.x)
        print(f"  {gelesen/total*100:.1f}% gelesen ({gelesen:,} Punkte)")

xyz       = np.vstack(xyz_list);       del xyz_list
intensity = np.concatenate(intensity_list); del intensity_list

print(f"\nPunkte geladen: {len(xyz):,}")
print(f"RAM-Verbrauch XYZ: {xyz.nbytes / 1e6:.1f} MB")

# ── Schritt 2: Open3D PointCloud erstellen ───────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 2: Koordinaten normalisieren")
print("=" * 60)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

if rgb_list:
    rgb = np.vstack(rgb_list); del rgb_list
    pcd.colors = o3d.utility.Vector3dVector(rgb)

# Koordinaten zentrieren (Schwerpunkt auf Ursprung)
center = pcd.get_center()
pcd.translate(-center)
print(f"Koordinaten zentriert. Urspruenglicher Mittelpunkt: {center}")
print(f"Neuer Mittelpunkt: {pcd.get_center()}")

# ── Schritt 3: Rauschfilterung ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 3: Rauschfilterung (Statistical Outlier Removal)")
print("=" * 60)

vorher = len(pcd.points)
pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=OUTLIER_NEIGHBORS,
    std_ratio=OUTLIER_STD_RATIO
)
nachher = len(pcd.points)
print(f"Vor Filterung:  {vorher:,} Punkte")
print(f"Nach Filterung: {nachher:,} Punkte")
print(f"Entfernt:       {vorher - nachher:,} Ausreisser ({(vorher-nachher)/vorher*100:.2f}%)")

# ── Schritt 4: Downsampling ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"SCHRITT 4: Voxel-Grid-Downsampling (Voxelgroesse: {VOXEL_SIZE}m)")
print("=" * 60)

vorher = len(pcd.points)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
nachher = len(pcd.points)
print(f"Vor Downsampling:  {vorher:,} Punkte")
print(f"Nach Downsampling: {nachher:,} Punkte")
print(f"Reduktion: {(1 - nachher/vorher)*100:.1f}%")

# ── Schritt 5: Normalen schaetzen ────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 5: Normalen schaetzen")
print("=" * 60)

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(30)
print("Normalen geschaetzt und orientiert.")

# ── Schritt 6: Als numpy speichern ──────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 6: Ergebnis speichern")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

points  = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# Hoehe ueber Grund als Feature (Z-Koordinate nach Zentrierung)
hoehe = points[:, 2].reshape(-1, 1)

# Intensitaet interpolieren (nach Downsampling nicht mehr direkt verfuegbar)
# Verwende Z-Koordinate als Proxy-Feature
features = np.hstack([normals, hoehe])  # 4 Features: nx, ny, nz, hoehe

daten = {
    "points":   points,
    "normals":  normals,
    "features": features,
    "center":   center
}

np.save(OUTPUT_FILE, daten, allow_pickle=True)
print(f"Gespeichert: {OUTPUT_FILE}")
print(f"Finale Punktanzahl: {len(points):,}")
print(f"Features pro Punkt: {features.shape[1]}")

# Auch als PLY speichern fuer Visualisierung
ply_file = "data/processed/preprocessed.ply"
o3d.io.write_point_cloud(ply_file, pcd)
print(f"PLY gespeichert: {ply_file}")

print("\n" + "=" * 60)
print("FERTIG - Weiter mit 03_segment.py")
print("=" * 60)

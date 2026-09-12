"""
02_preprocess.py
----------------
Schritt 2 der Pipeline: Vorverarbeitung der Punktwolke.
- Rauschfilterung (Statistical Outlier Removal)
- Normalisierung (Koordinaten zentrieren)
- Downsampling (Voxel-Grid-Filter)
- Normalen schaetzen
- Speichern als numpy Array inkl. RGB fuer naechste Schritte
"""

import laspy
import open3d as o3d
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_FILE  = "data/processed/ausschnitt_100m.laz"
OUTPUT_FILE = "data/processed/preprocessed.npy"

VOXEL_SIZE          = 0.25
OUTLIER_NEIGHBORS   = 20
OUTLIER_STD_RATIO   = 2.0

# ── Schritt 1: LAZ einlesen ───────────────────────────────────────────────────
print("=" * 60)
print("SCHRITT 1: Punktwolke einlesen")
print("=" * 60)

print("Lese LAZ-Datei (chunkweise um RAM zu schonen)...")

xyz_list       = []
rgb_list       = []
intensity_list = []

CHUNK_SIZE = 5_000_000

with laspy.open(INPUT_FILE) as f:
    total   = f.header.point_count
    gelesen = 0
    hat_rgb = all(hasattr(f.header.point_format, dim)
                  for dim in ['red', 'green', 'blue'])

    for chunk in f.chunk_iterator(CHUNK_SIZE):
        xyz_list.append(np.column_stack([
            chunk.x.copy(),
            chunk.y.copy(),
            chunk.z.copy()
        ]))

        # RGB - 16-bit auf 0-1 normalisieren
        if hasattr(chunk, 'red'):
            rgb_list.append(np.column_stack([
                np.array(chunk.red,   dtype=np.float32) / 65535.0,
                np.array(chunk.green, dtype=np.float32) / 65535.0,
                np.array(chunk.blue,  dtype=np.float32) / 65535.0,
            ]))
        else:
            rgb_list.append(np.zeros((len(chunk.x), 3), dtype=np.float32))

        intensity_list.append(np.array(chunk.intensity, dtype=np.float32))

        gelesen += len(chunk.x)
        print(f"  {gelesen/total*100:.1f}% gelesen ({gelesen:,} Punkte)")

xyz       = np.vstack(xyz_list);       del xyz_list
rgb       = np.vstack(rgb_list);       del rgb_list
intensity = np.concatenate(intensity_list); del intensity_list

print(f"\nPunkte geladen: {len(xyz):,}")
print(f"RAM XYZ: {xyz.nbytes/1e6:.1f} MB")
print(f"RAM RGB: {rgb.nbytes/1e6:.1f} MB")

# ── Schritt 2: Open3D PointCloud erstellen ───────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 2: Koordinaten normalisieren")
print("=" * 60)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

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

# Intensity ebenfalls filtern. Das Array der Intensitäten wird auf die Indizes der gefilterten Punkte reduziert.
intensity = intensity[ind]

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
# Normalen nach oben orientieren (ALS Perspektive von oben)
pcd.orient_normals_consistent_tangent_plane(30)
print("Normalen geschaetzt und nach oben orientiert.")

# ── Schritt 6: Als numpy speichern ──────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 6: Ergebnis speichern")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

points  = np.asarray(pcd.points,  dtype=np.float32)
normals = np.asarray(pcd.normals, dtype=np.float32)
colors  = np.asarray(pcd.colors,  dtype=np.float32)
#hoehe   = points[:, 2:3]

# Features fuer Random Forest: XYZ + RGB + VDVI
r = colors[:, 0:1]
g = colors[:, 1:2]
b = colors[:, 2:3]
nenner = 2*g + r + b + 1e-8
vdvi   = (2*g - r - b) / nenner
features = np.hstack([points, colors, vdvi])  # 7 Features

daten = {
    "points":    points,
    "normals":   normals,
    "colors":    colors,
    #"intensity": intensity_down,
    "features":  features,
    "center":    center
}

np.save(OUTPUT_FILE, daten, allow_pickle=True)
print(f"Gespeichert: {OUTPUT_FILE}")
print(f"Finale Punktanzahl: {len(points):,}")
print(f"Features pro Punkt: {features.shape[1]}")
print(f"RGB gespeichert: {colors.shape}")

# Auch als PLY speichern fuer Visualisierung
ply_file = "data/processed/preprocessed.ply"
o3d.io.write_point_cloud(ply_file, pcd)
print(f"PLY gespeichert: {ply_file}")

print("\n" + "=" * 60)
print("FERTIG - Weiter mit 03_segment_rf.py")
print("=" * 60)
"""
03_reconstruct.py
-----------------
Schritt 3 der Pipeline: Mesh-Rekonstruktion aus vorverarbeiteter Punktwolke.
Nutzt Screened Poisson Surface Reconstruction (Open3D).
Da keine Segmentierung verfuegbar ist, wird die gesamte Punktwolke
als ein Mesh rekonstruiert und anschliessend nach Hoehe grob klassifiziert.
"""

import open3d as o3d
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_NPY   = "data/processed/preprocessed.npy"
INPUT_PLY   = "data/processed/preprocessed.ply"
OUTPUT_DIR  = "data/output"

POISSON_DEPTH    = 9     # Octree-Tiefe fuer Poisson Reconstruction
LOD0_TARGET      = None  # Original behalten
LOD1_REDUCTION   = 0.1   # 10% der Dreiecke behalten

# ── Schritt 1: Punktwolke laden ──────────────────────────────────────────────
print("=" * 60)
print("SCHRITT 1: Punktwolke laden")
print("=" * 60)

pcd = o3d.io.read_point_cloud(INPUT_PLY)
print(f"Punkte geladen: {len(pcd.points):,}")
print(f"Hat Normalen: {pcd.has_normals()}")

# Normalen neu schaetzen falls nicht vorhanden
if not pcd.has_normals():
    print("Normalen werden geschaetzt...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    print("Normalen geschaetzt.")

# ── Schritt 2: Höhenbasierte grobe Klassifikation ────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 2: Hoehenbasierte Klassifikation (Proxy fuer Segmentierung)")
print("=" * 60)

points = np.asarray(pcd.points)
z = points[:, 2]

# Schwellenwerte basierend auf Z-Verteilung
z_min = z.min()
z_max = z.max()
z_range = z_max - z_min

# Klassen nach Hoehe
boden_maske      = z < (z_min + z_range * 0.35)
vegetation_maske = (z >= (z_min + z_range * 0.35)) & (z < (z_min + z_range * 0.65))
gebaeude_maske   = z >= (z_min + z_range * 0.65)

print(f"Z-Bereich: {z_min:.2f}m bis {z_max:.2f}m (Spanne: {z_range:.2f}m)")
print(f"Boden     (< {z_min + z_range*0.35:.2f}m): {boden_maske.sum():,} Punkte")
print(f"Vegetation ({z_min + z_range*0.35:.2f}m - {z_min + z_range*0.65:.2f}m): {vegetation_maske.sum():,} Punkte")
print(f"Gebaeude  (> {z_min + z_range*0.65:.2f}m): {gebaeude_maske.sum():,} Punkte")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Schritt 3: Mesh-Rekonstruktion pro Klasse ────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 3: Mesh-Rekonstruktion (Screened Poisson)")
print("=" * 60)

def rekonstruiere_mesh(pcd_klasse, name, depth=POISSON_DEPTH):
    """Rekonstruiert ein Mesh aus einer Punktwolke und gibt es zurueck."""
    print(f"\n  Rekonstruiere {name} ({len(pcd_klasse.points):,} Punkte)...")
    
    if len(pcd_klasse.points) < 100:
        print(f"  Zu wenige Punkte fuer {name} - ueberspringe.")
        return None
    
    # Normalen schaetzen falls nicht vorhanden
    if not pcd_klasse.has_normals():
        pcd_klasse.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )
        pcd_klasse.orient_normals_consistent_tangent_plane(30)
    
    # Poisson Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_klasse, depth=depth
    )
    
    # Schwach belegte Bereiche entfernen (Artefakte)
    densities = np.asarray(densities)
    schwellenwert = np.percentile(densities, 10)
    vertices_zu_entfernen = densities < schwellenwert
    mesh.remove_vertices_by_mask(vertices_zu_entfernen)
    
    print(f"  Dreiecke: {len(mesh.triangles):,}")
    print(f"  Vertices: {len(mesh.vertices):,}")
    
    return mesh

# Klassen-Punktwolken erstellen
klassen = {
    "boden":      boden_maske,
    "vegetation": vegetation_maske,
    "gebaeude":   gebaeude_maske
}

meshes = {}
for name, maske in klassen.items():
    idx = np.where(maske)[0]
    pcd_klasse = pcd.select_by_index(idx.tolist())
    mesh = rekonstruiere_mesh(pcd_klasse, name)
    if mesh is not None:
        meshes[name] = mesh

# ── Schritt 4: LOD-Generierung ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 4: LOD-Generierung (Mesh-Decimation)")
print("=" * 60)

for name, mesh in meshes.items():
    n_dreiecke = len(mesh.triangles)
    
    # LOD0 speichern (original)
    lod0_path = f"{OUTPUT_DIR}/{name}_LOD0.ply"
    o3d.io.write_triangle_mesh(lod0_path, mesh)
    print(f"\n{name} LOD0: {n_dreiecke:,} Dreiecke → {lod0_path}")
    
    # LOD1 erzeugen (10% der Dreiecke)
    ziel_dreiecke = max(100, int(n_dreiecke * LOD1_REDUCTION))
    mesh_lod1 = mesh.simplify_quadric_decimation(ziel_dreiecke)
    lod1_path = f"{OUTPUT_DIR}/{name}_LOD1.ply"
    o3d.io.write_triangle_mesh(lod1_path, mesh_lod1)
    print(f"{name} LOD1: {len(mesh_lod1.triangles):,} Dreiecke → {lod1_path}")

# ── Schritt 5: glTF Export ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 5: glTF Export")
print("=" * 60)

farben = {
    "boden":      [0.6, 0.5, 0.3],   # Braun
    "vegetation": [0.2, 0.7, 0.2],   # Gruen
    "gebaeude":   [0.8, 0.8, 0.8],   # Grau
}

for name, mesh in meshes.items():
    # Farbe zuweisen
    mesh.paint_uniform_color(farben[name])
    
    # glTF Export
    gltf_path = f"{OUTPUT_DIR}/{name}_LOD0.glb"
    o3d.io.write_triangle_mesh(gltf_path, mesh)
    
    groesse = os.path.getsize(gltf_path) / 1e6
    print(f"{name}: {gltf_path} ({groesse:.1f} MB)")

print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
for name, mesh in meshes.items():
    lod0_size = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0.ply") / 1e6
    lod1_size = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD1.ply") / 1e6
    print(f"{name:12}: LOD0={len(mesh.triangles):>8,} Dreiecke ({lod0_size:.1f}MB) | LOD1={int(len(mesh.triangles)*LOD1_REDUCTION):>6,} Dreiecke ({lod1_size:.1f}MB)")

print("\n" + "=" * 60)
print("FERTIG - Output in data/output/")
print("=" * 60)

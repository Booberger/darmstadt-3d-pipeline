"""
04_reconstruct.py
-----------------
Schritt 4 der Pipeline: Mesh-Rekonstruktion aus segmentierter Punktwolke.
- Connected Component Filtering (Artefaktentfernung)
- Screened Poisson Surface Reconstruction pro Klasse
- Mesh-Bereinigung
- 3-stufige LOD-Generierung via Quadric Decimation (Li & Nan 2021)
  LOD0: Original | LOD1: 10% | LOD2: 1%
- glTF Export mit Klassenfarben UND echten RGB-Farben
"""

import open3d as o3d
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_PRED    = "data/processed/darmstadt_pred.npy"
INPUT_PREPROC = "data/processed/preprocessed.npy"
OUTPUT_DIR    = "data/output"

POISSON_DEPTH     = 9
LOD1_REDUCTION    = 0.10   # 10% der Dreiecke
LOD2_REDUCTION    = 0.01   # 1%  der Dreiecke
DICHTE_PERCENTIL  = 10
DBSCAN_EPS        = 2.0
DBSCAN_MIN_POINTS = 10
MIN_CLUSTER_PUNKTE = 50

KLASSEN = {
    0: "boden",
    1: "vegetation",
    2: "gebaeude",
    3: "strasse"
}

KLASSEN_FARBEN = {
    0: [0.6, 0.5, 0.3],
    1: [0.2, 0.7, 0.2],
    2: [0.8, 0.8, 0.8],
    3: [0.3, 0.3, 0.3],
}

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def filtere_artefakte(pcd):
    if len(pcd.points) < MIN_CLUSTER_PUNKTE:
        return pcd
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=False
    ))
    if len(labels) == 0 or labels.max() < 0:
        return pcd
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    grosse_cluster = unique[counts >= MIN_CLUSTER_PUNKTE]
    if len(grosse_cluster) == 0:
        return pcd
    maske = np.isin(labels, grosse_cluster)
    pcd_f = pcd.select_by_index(np.where(maske)[0].tolist())
    entfernt = len(pcd.points) - len(pcd_f.points)
    print(f"    Artefakte entfernt: {entfernt:,} ({entfernt/len(pcd.points)*100:.1f}%)")
    return pcd_f


def rekonstruiere_klasse(xyz_all, labels_all, normals_all, colors_all,
                          klasse_id, name):
    maske    = labels_all == klasse_id
    n_punkte = int(np.sum(maske))

    if n_punkte < 100:
        print(f"  {name}: Zu wenige Punkte ({n_punkte}) - ueberspringe.")
        return None, None

    print(f"\n  {name}: {n_punkte:,} Punkte")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_all[maske])

    if normals_all is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals_all[maske])
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=1.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(xyz_all[maske])
    if colors_all is not None:
        pcd_rgb.colors  = o3d.utility.Vector3dVector(colors_all[maske])
        pcd_rgb.normals = pcd.normals

    pcd = filtere_artefakte(pcd)
    if len(pcd.points) < 100:
        return None, None

    print(f"    Poisson Reconstruction (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )

    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(
        densities < np.percentile(densities, DICHTE_PERCENTIL)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    print(f"    Dreiecke: {len(mesh.triangles):,}")
    print(f"    Vertices: {len(mesh.vertices):,}")

    return mesh, pcd_rgb


def erzeuge_lod(mesh, farbe, n, reduktion, suffix, output_dir, name):
    """Erzeugt eine LOD-Stufe als PLY und GLB."""
    ziel      = max(10, int(n * reduktion))
    mesh_lod  = mesh.simplify_quadric_decimation(ziel)
    mesh_lod.paint_uniform_color(farbe)

    ply_path  = f"{output_dir}/{name}_{suffix}.ply"
    glb_path  = f"{output_dir}/{name}_{suffix}.glb"
    o3d.io.write_triangle_mesh(ply_path, mesh_lod)
    o3d.io.write_triangle_mesh(glb_path, mesh_lod)

    return mesh_lod, len(mesh_lod.triangles), os.path.getsize(ply_path)/1e6


# ── Schritt 1: Daten laden ───────────────────────────────────────────────────
print("=" * 60)
print("SCHRITT 1: Segmentierungsergebnis laden")
print("=" * 60)

daten   = np.load(INPUT_PRED,    allow_pickle=True).item()
preproc = np.load(INPUT_PREPROC, allow_pickle=True).item()

xyz     = daten['xyz']    if 'xyz'    in daten else daten['points']
labels  = daten['labels']
normals = daten.get('normals', None)
colors  = preproc.get('colors', None)

if normals is not None and np.mean(normals[:, 2]) < 0:
    normals = -normals
    print("Normalen umgekehrt")

print(f"Punkte: {len(xyz):,}")
print(f"RGB verfuegbar: {colors is not None}")
print(f"Klassenverteilung:")
for k, name in KLASSEN.items():
    n = int(np.sum(labels == k))
    print(f"  {k} {name:12s}: {n:>6,} ({n/len(labels)*100:.1f}%)")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Schritt 2: Mesh-Rekonstruktion ──────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 2: Mesh-Rekonstruktion mit Artefaktfilterung")
print("=" * 60)

meshes = {}
for klasse_id, name in KLASSEN.items():
    mesh, pcd_rgb = rekonstruiere_klasse(
        xyz, labels, normals, colors, klasse_id, name
    )
    if mesh is not None:
        meshes[name] = (klasse_id, mesh, pcd_rgb)

# ── Schritt 3: LOD-Generierung ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 3: 3-stufige LOD-Generierung")
print("=" * 60)
print(f"{'Klasse':12s} | {'LOD0':>10s} | {'LOD1 (10%)':>12s} | {'LOD2 (1%)':>11s}")
print("-" * 55)

for name, (klasse_id, mesh, _) in meshes.items():
    farbe = KLASSEN_FARBEN[klasse_id]
    n     = len(mesh.triangles)

    # LOD0 — Original
    mesh_lod0 = o3d.geometry.TriangleMesh(mesh)
    mesh_lod0.paint_uniform_color(farbe)
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0.ply", mesh_lod0)
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0.glb", mesh_lod0)
    g0 = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0.ply") / 1e6

    # LOD1 — 10%
    _, n1, g1 = erzeuge_lod(mesh, farbe, n, LOD1_REDUCTION, "LOD1", OUTPUT_DIR, name)

    # LOD2 — 1%
    _, n2, g2 = erzeuge_lod(mesh, farbe, n, LOD2_REDUCTION, "LOD2", OUTPUT_DIR, name)

    print(f"{name:12s} | {n:>7,} ({g0:.1f}MB) | {n1:>7,} ({g1:.1f}MB) | {n2:>6,} ({g2:.1f}MB)")

# ── Schritt 4: glTF Export mit echten RGB-Farben ─────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 4: glTF Export (Echte RGB-Farben)")
print("=" * 60)

if colors is not None:
    for name, (klasse_id, mesh, pcd_rgb) in meshes.items():
        if pcd_rgb is None or not pcd_rgb.has_colors():
            continue

        mesh_rgb = o3d.geometry.TriangleMesh(mesh)
        vertices = np.asarray(mesh_rgb.vertices)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_rgb)
        vertex_colors = np.zeros((len(vertices), 3))

        for i, v in enumerate(vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            vertex_colors[i] = np.asarray(pcd_rgb.colors)[idx[0]]

        mesh_rgb.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # LOD0 RGB
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0_rgb.glb", mesh_rgb)

        # LOD1 RGB
        mesh_lod1_rgb = mesh_rgb.simplify_quadric_decimation(
            max(10, int(len(mesh_rgb.triangles) * LOD1_REDUCTION))
        )
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD1_rgb.glb", mesh_lod1_rgb)

        # LOD2 RGB
        mesh_lod2_rgb = mesh_rgb.simplify_quadric_decimation(
            max(10, int(len(mesh_rgb.triangles) * LOD2_REDUCTION))
        )
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD2_rgb.glb", mesh_lod2_rgb)

        g = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0_rgb.glb") / 1e6
        print(f"{name:12s}: LOD0/LOD1/LOD2 RGB exportiert ({g:.1f}MB)")
else:
    print("Keine RGB-Daten verfuegbar.")

# ── Zusammenfassung ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print("Ausgabedateien pro Klasse:")
print("  *_LOD0/LOD1/LOD2.ply     - Klassenfarben (MeshLab)")
print("  *_LOD0/LOD1/LOD2.glb     - Klassenfarben (Game Engine)")
print("  *_LOD0/LOD1/LOD2_rgb.glb - Echte RGB-Farben (Game Engine)")

print("\n" + "=" * 60)
print("FERTIG - Output in data/output/")
print("=" * 60)
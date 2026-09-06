"""
04_reconstruct.py
-----------------
Schritt 4 der Pipeline: Mesh-Rekonstruktion aus segmentierter Punktwolke.
- Connected Component Filtering (Artefaktentfernung)
- Screened Poisson Surface Reconstruction pro Klasse
- Mesh-Bereinigung
- LOD-Generierung via Quadric Decimation (Li & Nan 2021)
- glTF Export mit Klassenfarben UND echten RGB-Farben
"""

import open3d as o3d
import numpy as np
import os

# ── Konfiguration ────────────────────────────────────────────────────────────
INPUT_PRED   = "data/processed/darmstadt_pred.npy"
INPUT_PREPROC = "data/processed/preprocessed.npy"
OUTPUT_DIR   = "data/output"

POISSON_DEPTH     = 9
LOD1_REDUCTION    = 0.1
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

def filtere_artefakte(pcd, name):
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
    pcd_gefiltert = pcd.select_by_index(np.where(maske)[0].tolist())
    entfernt = len(pcd.points) - len(pcd_gefiltert.points)
    print(f"    Artefakte entfernt: {entfernt:,} ({entfernt/len(pcd.points)*100:.1f}%)")
    return pcd_gefiltert


def rekonstruiere_klasse(xyz_all, labels_all, normals_all, colors_all,
                          klasse_id, name):
    maske    = labels_all == klasse_id
    n_punkte = int(np.sum(maske))

    if n_punkte < 100:
        print(f"  {name}: Zu wenige Punkte ({n_punkte}) - ueberspringe.")
        return None, None

    print(f"\n  {name}: {n_punkte:,} Punkte")

    # PointCloud mit Normalen
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

    # PointCloud mit echten RGB (separat fuer spaeter)
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(xyz_all[maske])
    if colors_all is not None:
        pcd_rgb.colors = o3d.utility.Vector3dVector(colors_all[maske])
        pcd_rgb.normals = pcd.normals

    # Artefaktfilterung
    pcd = filtere_artefakte(pcd, name)
    if len(pcd.points) < 100:
        return None, None

    # Poisson Reconstruction
    print(f"    Poisson Reconstruction (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )

    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.percentile(densities, DICHTE_PERCENTIL))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    print(f"    Dreiecke: {len(mesh.triangles):,}")
    print(f"    Vertices: {len(mesh.vertices):,}")

    return mesh, pcd_rgb


# ── Schritt 1: Daten laden ───────────────────────────────────────────────────
print("=" * 60)
print("SCHRITT 1: Segmentierungsergebnis laden")
print("=" * 60)

daten   = np.load(INPUT_PRED, allow_pickle=True).item()
xyz     = daten['xyz']    if 'xyz'    in daten else daten['points']
labels  = daten['labels']
normals = daten.get('normals', None)

# Echte RGB aus preprocessed.npy laden
preproc = np.load(INPUT_PREPROC, allow_pickle=True).item()
colors  = preproc.get('colors', None)

# Normalen nach oben orientieren falls noetig
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
print("SCHRITT 3: LOD-Generierung")
print("=" * 60)

for name, (klasse_id, mesh, _) in meshes.items():
    n = len(mesh.triangles)

    # Klassenfarben PLY
    mesh_colored = o3d.geometry.TriangleMesh(mesh)
    mesh_colored.paint_uniform_color(KLASSEN_FARBEN[klasse_id])
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD0.ply", mesh_colored)

    mesh_lod1 = mesh.simplify_quadric_decimation(max(100, int(n * LOD1_REDUCTION)))
    mesh_lod1.paint_uniform_color(KLASSEN_FARBEN[klasse_id])
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD1.ply", mesh_lod1)

    g0 = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0.ply") / 1e6
    g1 = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD1.ply") / 1e6
    print(f"{name:12s}: LOD0={n:>8,} ({g0:.1f}MB) | LOD1={len(mesh_lod1.triangles):>6,} ({g1:.1f}MB)")

# ── Schritt 4: glTF Export mit Klassenfarben ─────────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 4: glTF Export (Klassenfarben)")
print("=" * 60)

for name, (klasse_id, mesh, _) in meshes.items():
    mesh_col = o3d.geometry.TriangleMesh(mesh)
    mesh_col.paint_uniform_color(KLASSEN_FARBEN[klasse_id])
    gltf0 = f"{OUTPUT_DIR}/{name}_LOD0.glb"
    o3d.io.write_triangle_mesh(gltf0, mesh_col)

    mesh_lod1 = mesh.simplify_quadric_decimation(
        max(100, int(len(mesh.triangles) * LOD1_REDUCTION))
    )
    mesh_lod1.paint_uniform_color(KLASSEN_FARBEN[klasse_id])
    o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD1.glb", mesh_lod1)
    print(f"{name:12s}: {gltf0} ({os.path.getsize(gltf0)/1e6:.1f}MB)")

# ── Schritt 5: glTF Export mit echten RGB-Farben ─────────────────────────────
print("\n" + "=" * 60)
print("SCHRITT 5: glTF Export (Echte RGB-Farben)")
print("=" * 60)

if colors is not None:
    for name, (klasse_id, mesh, pcd_rgb) in meshes.items():
        if pcd_rgb is None or not pcd_rgb.has_colors():
            print(f"{name:12s}: Keine RGB-Daten verfuegbar")
            continue

        # RGB-Farben auf Mesh-Vertices uebertragen via naechster Nachbar
        mesh_rgb = o3d.geometry.TriangleMesh(mesh)
        vertices = np.asarray(mesh_rgb.vertices)

        # KD-Tree auf RGB-Punktwolke
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_rgb)
        vertex_colors = np.zeros((len(vertices), 3))

        for i, v in enumerate(vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            vertex_colors[i] = np.asarray(pcd_rgb.colors)[idx[0]]

        mesh_rgb.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        gltf_rgb = f"{OUTPUT_DIR}/{name}_LOD0_rgb.glb"
        o3d.io.write_triangle_mesh(gltf_rgb, mesh_rgb)
        print(f"{name:12s}: {gltf_rgb} ({os.path.getsize(gltf_rgb)/1e6:.1f}MB)")

        # LOD1 mit RGB
        mesh_lod1_rgb = mesh_rgb.simplify_quadric_decimation(
            max(100, int(len(mesh_rgb.triangles) * LOD1_REDUCTION))
        )
        o3d.io.write_triangle_mesh(f"{OUTPUT_DIR}/{name}_LOD1_rgb.glb", mesh_lod1_rgb)
else:
    print("Keine RGB-Daten in preprocessed.npy verfuegbar.")

# ── Zusammenfassung ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
for name, (klasse_id, mesh, _) in meshes.items():
    n  = len(mesh.triangles)
    g0 = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD0.ply") / 1e6
    g1 = os.path.getsize(f"{OUTPUT_DIR}/{name}_LOD1.ply") / 1e6
    print(f"{name:12s}: LOD0={n:>8,} ({g0:.1f}MB) | LOD1={int(n*LOD1_REDUCTION):>6,} ({g1:.1f}MB)")

print("\nAusgabedateien:")
print("  *_LOD0/LOD1.ply  - Klassenfarben (MeshLab)")
print("  *_LOD0/LOD1.glb  - Klassenfarben (Game Engine)")
print("  *_LOD0/LOD1_rgb.glb - Echte RGB-Farben (Game Engine)")

print("\n" + "=" * 60)
print("FERTIG - Output in data/output/")
print("=" * 60)
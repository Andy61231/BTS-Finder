#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_demo.py - Master runner for the Cellular BTS Localization Worked Example.

Usage:
  # Run the default worked example on the 9 target eNodeBs:
  python run_demo.py

  # Run on a specific eNodeB from the dataset:
  python run_demo.py --enodeb 80609

  # Ingest your own custom export from the Android Network Survey app:
  python run_demo.py --input data/user_input/my_survey.csv
  python run_demo.py --input path/to/craxiom-export.csv --enodeb 101811
"""

import os
import sys
import glob
import json
import argparse
import time
import math
from typing import Dict, List

# Ensure package root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.loader import load_measurements, get_dataset_summary
from src.preprocessing import SpatialProjector, spatial_deduplicate
from src.path_loss import PathLossModel
from src.solver import BTSLocalizationSolver
from src.sectors import generate_cell_sectors
from src.clustering import detect_colocated_enodebs, apply_visual_offset
from src.visualization import (
    plot_localization_triangulation,
    plot_path_loss_model,
    plot_sector_coverage,
    build_interactive_map
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cellular BTS Localization - Worked Example & Academic Reproduction Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to CSV survey file. Supports both preprocessed CSVs and raw Network Survey exports. "
             "If omitted, automatically checks data/user_input/ or defaults to data/sample_lte_measurements.csv."
    )
    parser.add_argument(
        "--enodeb", "-e",
        type=str,
        default=None,
        help="Target eNodeB ID to locate (e.g. 80609, 101811, 240151, 950063). "
             "If 'all' or omitted, processes all eligible eNodeBs with >= 10 samples."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=os.path.join(BASE_DIR, "output"),
        help="Directory where output figures, interactive map, and summary reports are saved."
    )
    parser.add_argument(
        "--min-samples", "-m",
        type=int,
        default=8,
        help="Minimum number of samples required to attempt localization (default: 8)."
    )
    return parser.parse_args()


def resolve_input_file(arg_path: str) -> str:
    """Finds the active input file, checking user_input folder or defaulting to sample data."""
    if arg_path and os.path.isfile(arg_path):
        return arg_path

    # Check data/user_input folder for custom files dropped by the user
    user_input_dir = os.path.join(BASE_DIR, "data", "user_input")
    if os.path.isdir(user_input_dir):
        user_csvs = glob.glob(os.path.join(user_input_dir, "*.csv"))
        if user_csvs and arg_path is None:
            # Pick newest file in user_input
            user_csvs.sort(key=os.path.getmtime, reverse=True)
            candidate = user_csvs[0]
            # Only use if not the placeholder example
            if not os.path.basename(candidate).startswith("example_raw"):
                print(f"[INFO] Detected custom user input in data/user_input/: {os.path.basename(candidate)}")
                return candidate

    # Default fallback to sample dataset
    sample_csv = os.path.join(BASE_DIR, "data", "sample_lte_measurements.csv")
    if os.path.isfile(sample_csv):
        return sample_csv

    raise FileNotFoundError("Could not find sample_lte_measurements.csv or specified input file.")


def load_ground_truth() -> Dict:
    """Loads reference localizations if present."""
    gt_path = os.path.join(BASE_DIR, "data", "ground_truth_reference.json")
    if os.path.isfile(gt_path):
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main():
    args = parse_arguments()
    t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    input_file = resolve_input_file(args.input)

    print("=" * 75)
    print("      CELLULAR BTS LOCALIZATION - WORKED EXAMPLE & CITATION DEMO")
    print("=" * 75)
    print(f"[*] Input File     : {input_file}")
    print(f"[*] Output Dir     : {args.output_dir}")

    # Step 1: Load Data
    print(f"[*] Parsing measurements (supports raw Network Survey headers)...")
    df_raw = load_measurements(input_file)
    print(f"[+] Loaded {len(df_raw)} valid GPS cellular measurement records.")

    # Dataset inventory
    inventory = get_dataset_summary(df_raw)
    print("\n" + "-" * 75)
    print(f"{'eNodeB ID':<12} | {'Vendor / Type':<25} | {'Points':<8} | {'TA Count':<8} | {'Mean RSRP'}")
    print("-" * 75)
    for item in inventory[:15]:
        rsrp_str = f"{item['mean_rsrp_dbm']} dBm" if item['mean_rsrp_dbm'] else "N/A"
        print(f"{item['enodeb_id']:<12} | {item['vendor'] + ' ' + item['site_type']:<25} | {item['total_points']:<8} | {item['ta_points']:<8} | {rsrp_str}")
    if len(inventory) > 15:
        print(f"... and {len(inventory) - 15} more eNodeBs")
    print("-" * 75 + "\n")

    # Step 2: Determine target eNodeBs
    target_ids = []
    if args.enodeb and args.enodeb.lower() != 'all':
        try:
            target_ids = [int(args.enodeb)]
        except ValueError:
            print(f"[ERR] Invalid eNodeB ID: {args.enodeb}")
            sys.exit(1)
    else:
        # Filter eNodeBs with sufficient points
        target_ids = [item['enodeb_id'] for item in inventory if item['total_points'] >= args.min_samples]

    if not target_ids:
        print(f"[WARN] No eNodeBs found with at least {args.min_samples} samples. Trying top candidate...")
        if inventory:
            target_ids = [inventory[0]['enodeb_id']]
        else:
            print("[ERR] Dataset contains no valid eNodeB records.")
            sys.exit(1)

    print(f"[*] Target eNodeBs to locate: {target_ids}")

    # Load ground truth reference for accuracy comparison
    ground_truth = load_ground_truth()

    # Step 3: Localization Loop with Co-location Topology Detection
    projector = SpatialProjector(median_lon=float(df_raw['longitude'].median()))
    solver = BTSLocalizationSolver(projector=projector)

    # Detect co-located base stations (consecutive eNodeB IDs sharing the same physical mast)
    print("\n[*] Analyzing cellular topology for co-located multi-carrier sites...")
    cluster_map = detect_colocated_enodebs(df_raw, max_distance_m=500.0)
    df_raw['cluster_id'] = df_raw['enodeb_id'].map(cluster_map).fillna(-1).astype(int)

    # Build group execution tasks
    # Co-located groups are solved jointly using their pooled measurement data
    execution_groups = []
    processed_clusters = set()

    for eid in target_ids:
        cid = cluster_map.get(eid, -1)
        if cid != -1:
            if cid not in processed_clusters:
                cluster_members = sorted(df_raw[df_raw['cluster_id'] == cid]['enodeb_id'].unique().tolist())
                cluster_df = df_raw[df_raw['cluster_id'] == cid].copy()
                execution_groups.append({
                    'type': 'cluster',
                    'cluster_id': cid,
                    'members': [int(m) for m in cluster_members],
                    'representative_id': int(cluster_members[0]),
                    'df': cluster_df
                })
                processed_clusters.add(cid)
        else:
            enb_df = df_raw[df_raw['enodeb_id'] == eid].copy()
            execution_groups.append({
                'type': 'singleton',
                'cluster_id': -1,
                'members': [int(eid)],
                'representative_id': int(eid),
                'df': enb_df
            })

    solutions = []
    sectors_by_enodeb = {}
    summary_report = []

    for group in execution_groups:
        grp_type = group['type']
        members = group['members']
        rep_id = group['representative_id']
        grp_df = group['df']

        if len(grp_df) < 3:
            continue

        if grp_type == 'cluster':
            print(f"\n---> Processing Co-located Mast Cluster (eNodeBs: {members}, {len(grp_df)} total records)...")
        else:
            print(f"\n---> Processing eNodeB {rep_id} ({len(grp_df)} raw records)...")

        # Deduplicate spatially
        grp_dedup = spatial_deduplicate(grp_df, grid_size_m=10.0, projector=projector)
        print(f"     [+] Deduplicated to {len(grp_dedup)} spatial grid cells.")

        # Solve master coordinates
        sol_master = solver.solve(grp_dedup, eNodeB_id=rep_id)
        if sol_master.get('status') != 'ok':
            print(f"     [!] Localization skipped: {sol_master.get('reason')}")
            continue

        base_lat = sol_master['lat']
        base_lon = sol_master['lon']

        # Process each eNodeB member
        for idx, eid in enumerate(members):
            # Apply circular visual offset if co-located
            lat_display, lon_display = apply_visual_offset(
                base_lat, base_lon, idx, len(members), radius_m=10.0
            )

            sol = sol_master.copy()
            sol['eNodeB_ID'] = eid
            sol['lat'] = lat_display
            sol['lon'] = lon_display
            sol['is_colocated'] = (len(members) > 1)
            sol['cluster_members'] = [m for m in members if m != eid]

            solutions.append(sol)

            # Generate Sectors starting from the mast
            member_df = grp_dedup[grp_dedup['enodeb_id'] == eid] if grp_type == 'cluster' else grp_dedup
            if len(member_df) < 3:
                member_df = grp_dedup
            sectors = generate_cell_sectors(member_df, tower_lat=lat_display, tower_lon=lon_display)
            sectors_by_enodeb[eid] = sectors

            # Ground truth comparison
            gt_info = ground_truth.get(str(eid))
            dev_m = None
            if gt_info:
                d_lat = (sol['lat'] - gt_info['lat']) * 111139.0
                d_lon = (sol['lon'] - gt_info['lon']) * (111139.0 * math.cos(math.radians(gt_info['lat'])))
                dev_m = math.hypot(d_lat, d_lon)
                print(f"     [+] eNodeB {eid} Ground Truth Deviation: {dev_m:.1f} meters")

            print(f"     [+] eNodeB {eid} Estimated Location: Lat={sol['lat']:.6f}°, Lon={sol['lon']:.6f}° (Grade {sol['confidence_grade']}, RMSE: {sol['rmse_m']:.1f}m)")

            # Generate Figures for this eNodeB
            fig1_path = os.path.join(args.output_dir, f"fig1_bts_localization_{eid}.png")
            plot_localization_triangulation(
                points_df=member_df,
                sol=sol,
                out_path=fig1_path,
                ground_truth=gt_info,
                sectors=sectors
            )

            fig2_path = os.path.join(args.output_dir, f"fig2_path_loss_model_{eid}.png")
            plot_path_loss_model(
                points_df=member_df,
                path_loss=solver.path_loss,
                tower_lat=sol['lat'],
                tower_lon=sol['lon'],
                out_path=fig2_path,
                enodeb_id=eid
            )

            fig3_path = os.path.join(args.output_dir, f"fig3_sector_coverage_{eid}.png")
            plot_sector_coverage(
                points_df=member_df,
                sectors=sectors,
                tower_lat=sol['lat'],
                tower_lon=sol['lon'],
                out_path=fig3_path,
                enodeb_id=eid
            )

            summary_report.append({
                'eNodeB_ID': eid,
                'status': sol['status'],
                'lat': sol['lat'],
                'lon': sol['lon'],
                'rmse_m': sol['rmse_m'],
                'confidence_grade': sol['confidence_grade'],
                'confidence_score': sol['confidence_score'],
                'method': sol['method_used'],
                'total_points': sol['total_points'],
                'ta_points': sol['ta_points'],
                'is_colocated': sol['is_colocated'],
                'cluster_members': sol['cluster_members'],
                'path_loss_n': sol['path_loss_params']['path_loss_exponent_n'],
                'path_loss_PL0_dB': sol['path_loss_params']['PL_0_dB'],
                'ground_truth_deviation_m': round(dev_m, 1) if dev_m is not None else None,
                'figures': {
                    'triangulation': os.path.basename(fig1_path),
                    'path_loss': os.path.basename(fig2_path),
                    'sectors': os.path.basename(fig3_path)
                }
            })

    # Step 4: Build Global Interactive Map (Folium)
    interactive_html = os.path.join(args.output_dir, "interactive_map.html")
    print(f"\n[*] Rendering interactive GIS map to {interactive_html}...")
    build_interactive_map(
        solutions=solutions,
        sectors_by_enodeb=sectors_by_enodeb,
        points_df=df_raw[df_raw['enodeb_id'].isin(target_ids)],
        out_html_path=interactive_html
    )

    # Save summary report JSON
    metrics_json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"   DEMO EXECUTION COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 75)
    print(f"[OK] Successful Localizations : {len(solutions)} / {len(target_ids)}")
    print(f"[OK] Figures Generated In     : {args.output_dir}")
    print(f"[OK] Interactive HTML Map     : {interactive_html}")
    print(f"[OK] Metrics Summary JSON     : {metrics_json_path}")
    print("=" * 75)


if __name__ == "__main__":
    import math
    main()

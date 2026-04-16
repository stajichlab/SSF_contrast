#!/usr/bin/env python3
"""
Produce one per-species TSV summarising PFAM, SignalP, MEROPS, TMHMM, and CAZy
annotations from annotation_pieces/.  Each row is a protein; multi-value fields
are pipe-separated within a cell (e.g. multiple PFAM domains).
"""

import os
import csv
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

EVALUE_THRESHOLD = 1e-3  # PFAM significance cutoff


def load_pfam(path):
    """Return {protein_id: [pfam_acc:name:evalue, ...]}"""
    data = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            protein_id = parts[0]
            pfam_name  = parts[3]
            pfam_acc   = parts[4].split(".")[0]  # strip version
            evalue     = float(parts[6])          # full-sequence E-value
            if evalue <= EVALUE_THRESHOLD:
                data[protein_id].append(f"{pfam_acc}:{pfam_name}:{evalue:.2e}")
    return data


def load_signalp(path):
    """Return {protein_id: (start, end, probability)} — only SP rows."""
    data = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            # col 0: "ID ID" (space-separated ID pair in SignalP-6 GFF3)
            protein_id = parts[0].split()[0]
            start = parts[3]
            end   = parts[4]
            prob  = parts[5]
            data[protein_id] = (start, end, prob)
    return data


def load_merops(path):
    """Return {protein_id: (merops_id, pct_id, evalue)} — best hit per protein."""
    best = {}
    with gzip.open(path, "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 12:
                continue
            protein_id = row[0]
            merops_id  = row[1]
            pct_id     = row[2]
            evalue     = float(row[10])
            bitscore   = row[11]
            if protein_id not in best or evalue < float(best[protein_id][2]):
                best[protein_id] = (merops_id, pct_id, f"{evalue:.2e}")
    return best


def load_tmhmm(path):
    """Return {protein_id: (PredHel, ExpAA, Topology)} — only proteins with TM helices."""
    data = {}
    with gzip.open(path, "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 6:
                continue
            protein_id = row[0]
            kv = {k: v for k, v in (field.split("=", 1) for field in row[1:] if "=" in field)}
            pred_hel = kv.get("PredHel", "0")
            if pred_hel == "0":
                continue
            exp_aa   = kv.get("ExpAA", "")
            topology = kv.get("Topology", "")
            data[protein_id] = (pred_hel, exp_aa, topology)
    return data


def load_cazy(species_dir):
    """Return {protein_id: [(cazyme_fam, EC, substrate), ...]}"""
    data = defaultdict(list)
    overview = os.path.join(species_dir, "overview.tsv.gz")
    if not os.path.exists(overview):
        return data
    with gzip.open(overview, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            protein_id = row.get("Gene_ID", "").strip()
            if not protein_id:
                continue
            fam       = row.get("cazyme_fam", "").strip()
            ec        = row.get("EC", "").strip()
            substrate = row.get("Substrate", "").strip()
            data[protein_id].append((fam, ec, substrate))
    return data


def write_species_tsv(species, pfam, signalp, merops, tmhmm, cazy, outdir):
    all_proteins = (
        set(pfam)
        | set(signalp)
        | set(merops)
        | set(tmhmm)
        | set(cazy)
    )
    if not all_proteins:
        return

    out_path = os.path.join(outdir, f"{species}.annotation_summary.tsv")
    header = [
        "protein_id",
        "pfam_domains",            # pfam_acc:name:evalue|...
        "signalp_start", "signalp_end", "signalp_prob",
        "merops_id", "merops_pct_id", "merops_evalue",
        "tmhmm_pred_hel", "tmhmm_exp_aa", "tmhmm_topology",
        "cazy_family", "cazy_EC", "cazy_substrate",
    ]

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        for pid in sorted(all_proteins):
            pfam_str = "|".join(pfam.get(pid, [])) or ""
            sp = signalp.get(pid)
            sp_start, sp_end, sp_prob = (sp[0], sp[1], sp[2]) if sp else ("", "", "")
            mer = merops.get(pid)
            mer_id, mer_pct, mer_eval = (mer[0], mer[1], mer[2]) if mer else ("", "", "")
            tm = tmhmm.get(pid)
            tm_hel, tm_exp, tm_top = (tm[0], tm[1], tm[2]) if tm else ("", "", "")
            cazy_entries = cazy.get(pid, [])
            cazy_fam  = "|".join(e[0] for e in cazy_entries) or ""
            cazy_ec   = "|".join(e[1] for e in cazy_entries) or ""
            cazy_sub  = "|".join(e[2] for e in cazy_entries) or ""

            writer.writerow([
                pid,
                pfam_str,
                sp_start, sp_end, sp_prob,
                mer_id, mer_pct, mer_eval,
                tm_hel, tm_exp, tm_top,
                cazy_fam, cazy_ec, cazy_sub,
            ])

    print(f"  wrote {len(all_proteins):>6} proteins → {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir", default="annotation_pieces",
        help="Root of annotation_pieces/ (default: annotation_pieces)"
    )
    parser.add_argument(
        "--outdir", default="annotation_summaries",
        help="Output directory for per-species TSV files (default: annotation_summaries)"
    )
    args = parser.parse_args()

    base    = Path(args.annotation_dir)
    pfam_d  = base / "pfam"
    signalp_d = base / "signalp"
    merops_d  = base / "merops"
    tmhmm_d   = base / "tmhmm"
    cazy_d    = base / "cazy"
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect species names from PFAM directory (authoritative list)
    species_list = sorted(
        f.name.replace(".domtblout.gz", "")
        for f in pfam_d.glob("*.domtblout.gz")
    )
    print(f"Found {len(species_list)} species in {pfam_d}")

    for species in species_list:
        print(f"Processing {species} ...")

        pfam_file    = pfam_d    / f"{species}.domtblout.gz"
        signalp_file = signalp_d / f"{species}.signalp.gff3.gz"
        merops_file  = merops_d  / f"{species}.blasttab.gz"
        tmhmm_file   = tmhmm_d   / f"{species}.tmhmm_short.tsv.gz"
        cazy_dir     = cazy_d    / species

        pfam    = load_pfam(pfam_file)    if pfam_file.exists()    else {}
        signalp = load_signalp(signalp_file) if signalp_file.exists() else {}
        merops  = load_merops(merops_file)   if merops_file.exists()  else {}
        tmhmm   = load_tmhmm(tmhmm_file)     if tmhmm_file.exists()   else {}
        cazy    = load_cazy(str(cazy_dir))

        write_species_tsv(species, pfam, signalp, merops, tmhmm, cazy, outdir)

    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download and verify the real-data benchmark inputs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
M5_FILES = {
  "calendar.csv": (
    "https://zenodo.org/api/records/10203108/files/calendar.csv/content",
    "md5",
    "3ffeab2991b0c8e861d008b39ea4c95c",
  ),
  "sales_train_evaluation.csv": (
    (
      "https://zenodo.org/api/records/10203108/files/"
      "sales_train_evaluation.csv/content"
    ),
    "md5",
    "b806dfc9f30a745102b708c09951f6aa",
  ),
  "sell_prices.csv": (
    "https://zenodo.org/api/records/10203108/files/sell_prices.csv/content",
    "md5",
    "08c591caa99e55daf3e0ccac913f7c85",
  ),
}
BEIJING_URL = (
  "https://archive.ics.uci.edu/static/public/501/"
  "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
)
BEIJING_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
DCRNN_COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
METR_LA_FILE_ID = "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
METR_LA_SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"
METR_LA_GRAPH_FILES = {
  "graph_sensor_locations.csv": (
    (
      f"https://raw.githubusercontent.com/liyaguang/DCRNN/{DCRNN_COMMIT}/"
      "data/sensor_graph/graph_sensor_locations.csv"
    ),
    "eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f",
  ),
  "distances_la_2012.csv": (
    (
      f"https://raw.githubusercontent.com/liyaguang/DCRNN/{DCRNN_COMMIT}/"
      "data/sensor_graph/distances_la_2012.csv"
    ),
    "a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5",
  ),
}


def checksum(path: Path, algorithm: str) -> str:
  digest = hashlib.new(algorithm)
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def download(url: str, destination: Path, algorithm: str, expected: str) -> None:
  if destination.is_file() and checksum(destination, algorithm) == expected:
    print(f"verified existing {destination}")
    return
  destination.parent.mkdir(parents=True, exist_ok=True)
  partial = destination.with_suffix(destination.suffix + ".part")
  with urllib.request.urlopen(url) as response, partial.open("wb") as output:
    shutil.copyfileobj(response, output)
  actual = checksum(partial, algorithm)
  if actual != expected:
    raise ValueError(f"checksum mismatch for {destination.name}: {actual}")
  partial.replace(destination)
  print(f"downloaded and verified {destination}")


def download_google_drive(file_id: str, destination: Path, expected: str) -> None:
  if destination.is_file() and checksum(destination, "sha256") == expected:
    print(f"verified existing {destination}")
    return
  destination.parent.mkdir(parents=True, exist_ok=True)
  partial = destination.with_suffix(destination.suffix + ".part")
  result = gdown.download(id=file_id, output=str(partial), quiet=False)
  if result is None or not partial.is_file():
    raise RuntimeError(f"Google Drive download failed for {destination.name}")
  actual = checksum(partial, "sha256")
  if actual != expected:
    raise ValueError(f"checksum mismatch for {destination.name}: {actual}")
  partial.replace(destination)
  print(f"downloaded and verified {destination}")


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
  destination_root = destination.resolve()
  for member in archive.infolist():
    member_path = (destination / member.filename).resolve()
    if destination_root not in member_path.parents and member_path != destination_root:
      raise ValueError(f"unsafe zip member: {member.filename}")
  archive.extractall(destination)


def prepare_m5() -> None:
  root = RAW_ROOT / "m5"
  for filename, (url, algorithm, expected) in M5_FILES.items():
    download(url, root / filename, algorithm, expected)


def prepare_beijing() -> None:
  root = RAW_ROOT / "beijing"
  outer = root / "beijing-multi-site-air-quality-data.zip"
  download(BEIJING_URL, outer, "sha256", BEIJING_SHA256)
  extracted = root / "extracted"
  stations = root / "stations"
  with zipfile.ZipFile(outer) as archive:
    safe_extract(archive, extracted)
  inner = extracted / "PRSA2017_Data_20130301-20170228.zip"
  with zipfile.ZipFile(inner) as archive:
    safe_extract(archive, stations)
  print(f"extracted Beijing station files under {stations}")


def prepare_metr_la() -> None:
  root = RAW_ROOT / "metr_la"
  download_google_drive(METR_LA_FILE_ID, root / "metr-la.h5", METR_LA_SHA256)
  graph_root = root / "graph"
  for filename, (url, expected) in METR_LA_GRAPH_FILES.items():
    download(url, graph_root / filename, "sha256", expected)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset",
    choices=["all", "both", "m5", "beijing", "metr-la"],
    default="both",
  )
  args = parser.parse_args()
  if args.dataset in {"all", "both", "m5"}:
    prepare_m5()
  if args.dataset in {"all", "both", "beijing"}:
    prepare_beijing()
  if args.dataset in {"all", "metr-la"}:
    prepare_metr_la()


if __name__ == "__main__":
  main()

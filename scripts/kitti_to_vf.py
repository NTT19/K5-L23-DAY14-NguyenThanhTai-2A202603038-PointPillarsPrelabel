"""Convert a standard KITTI dataset to the point-cloud layout used by VF.

The converter deliberately does not run PointPillars or create Day 14
student tasks.  It only prepares ``sequence_XX`` folders so that the normal
morning self-test/pre-label commands and the CVAT importer can consume them.

Supported inputs
----------------
* KITTI object detection: ``velodyne/*.bin`` with optional ``image_2``,
  ``calib`` and ``label_2`` directories.
* KITTI tracking: ``velodyne/<sequence>/*.bin`` with optional ``image_02``
  (or ``image_2``), ``calib/<sequence>.txt`` and ``label_02/<sequence>.txt``.

KITTI lidar uses ``x=front, y=left, z=up``.  VF uses ``x=left, y=back,
z=up``; therefore the point conversion is ``(x_vf, y_vf, z_vf) =
(y_kitti, -x_kitti, z_kitti)``.  Intensities are copied unchanged.

Example::

    python scripts/kitti_to_vf.py \
        --input C:\\datasets\\KITTI\\training \
        --output C:\\datasets\\Data-VF \
        --chunk-size 60 --max-sequences 5

The optional ``--max-sequences 5`` example produces the five 60-frame
sequences expected by the Day 14 frame selector when the source has enough
frames.  It is not needed for a general CVAT import.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import vf_common as vf  # noqa: E402


KITTI_TO_VF = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

KITTI_LABEL_TO_VF = {
    "Car": ("Car", 1),
    "Van": ("Truck", 2),
    "Truck": ("Truck", 2),
    "Bus": ("Bus", 3),
    "Tram": ("Bus", 3),
    "Cyclist": ("Cyclist", 4),
    "Pedestrian": ("Pedestrian", 5),
    "Person_sitting": ("Pedestrian", 5),
}


@dataclass(frozen=True)
class SourceFrame:
    """One KITTI lidar frame and its optional sidecar files."""

    stamp: str
    lidar: Path
    image: Path | None
    label_file: Path | None
    calibration: Path | None
    tracking_index: int | None = None


@dataclass(frozen=True)
class SourceStream:
    """A KITTI object split or one KITTI tracking sequence."""

    name: str
    frames: tuple[SourceFrame, ...]


def _natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 10**12, path.stem)


def _first_dir(parent: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = parent / name
        if path.is_dir():
            return path
    return None


def _image_for(directory: Path | None, stamp: str) -> Path | None:
    if directory is None:
        return None
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stamp}{suffix}"
        if path.is_file():
            return path
    return None


def _resolve_kitti_root(input_path: Path) -> Path:
    root = input_path.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"KITTI input không phải thư mục: {root}")
    if root.name == "velodyne" and root.is_dir():
        root = root.parent
    candidates = [root]
    if root.name not in {"training", "testing"}:
        candidates.extend([root / "training", root / "testing"])
    for candidate in candidates:
        if (candidate / "velodyne").is_dir():
            return candidate
    raise SystemExit(
        f"Không thấy thư mục velodyne trong {root}. Đây phải là KITTI object/tracking "
        "hoặc đường dẫn tới training/testing."
    )


def _find_tracking_sidecar(parent: Path, names: tuple[str, ...], sequence: str) -> Path | None:
    directory = _first_dir(parent, names)
    if directory is None:
        return None
    for candidate in (directory / f"{sequence}.txt", directory / f"{int(sequence):04d}.txt"):
        if candidate.is_file():
            return candidate
    return None


def discover_streams(input_path: Path) -> tuple[Path, list[SourceStream], str]:
    """Discover KITTI object/tracking frames without reading point data."""

    root = _resolve_kitti_root(input_path)
    velodyne = root / "velodyne"
    nested = sorted((p for p in velodyne.iterdir() if p.is_dir()), key=_natural_key)
    if nested:
        image_root = _first_dir(root, ("image_02", "image_2", "image02"))
        label_root = _first_dir(root, ("label_02", "label2", "labels"))
        calib_root = _first_dir(root, ("calib", "calibration"))
        streams: list[SourceStream] = []
        for sequence_dir in nested:
            sequence = sequence_dir.name
            bins = sorted(sequence_dir.glob("*.bin"), key=_natural_key)
            if not bins:
                continue
            images = image_root / sequence if image_root is not None else None
            label_file = _find_tracking_sidecar(root, ("label_02", "label2", "labels"), sequence)
            calibration = None
            if calib_root is not None:
                for candidate in (calib_root / f"{sequence}.txt", calib_root / f"{int(sequence):04d}.txt"):
                    if candidate.is_file():
                        calibration = candidate
                        break
            frames = tuple(
                SourceFrame(
                    stamp=path.stem,
                    lidar=path,
                    image=_image_for(images, path.stem),
                    label_file=label_file,
                    calibration=calibration,
                    tracking_index=int(path.stem) if path.stem.isdigit() else None,
                )
                for path in bins
            )
            streams.append(SourceStream(sequence, frames))
        if not streams:
            raise SystemExit(f"Không thấy file .bin trong các sequence tracking dưới {velodyne}.")
        return root, streams, "tracking"

    bins = sorted(velodyne.glob("*.bin"), key=_natural_key)
    if not bins:
        raise SystemExit(f"Không thấy file .bin trong {velodyne}.")
    image_dir = _first_dir(root, ("image_2", "image_02", "image2"))
    label_dir = _first_dir(root, ("label_2", "label02", "labels"))
    calib_dir = _first_dir(root, ("calib", "calibration"))
    frames = tuple(
        SourceFrame(
            stamp=path.stem,
            lidar=path,
            image=_image_for(image_dir, path.stem),
            label_file=(label_dir / f"{path.stem}.txt") if label_dir is not None and (label_dir / f"{path.stem}.txt").is_file() else None,
            calibration=(calib_dir / f"{path.stem}.txt") if calib_dir is not None and (calib_dir / f"{path.stem}.txt").is_file() else None,
        )
        for path in bins
    )
    return root, [SourceStream(root.name, frames)], "object"


def _read_calibration(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        nums = [float(value) for value in raw.split()]
        if len(nums) == 12:
            values[key.strip()] = np.asarray(nums, dtype=np.float64).reshape(3, 4)
        elif len(nums) == 9:
            values[key.strip()] = np.asarray(nums, dtype=np.float64).reshape(3, 3)
    p2 = values.get("P2")
    rect = values.get("R0_rect")
    if rect is None:
        rect = values.get("R_rect_00")
    velo = values.get("Tr_velo_to_cam")
    if velo is None:
        velo = values.get("Tr_velo_to_cam_0")
    if p2 is None or rect is None or velo is None:
        raise ValueError(f"{path}: thiếu P2, R0_rect/R_rect_00 hoặc Tr_velo_to_cam")
    return {"P2": p2, "rect": rect, "velo": velo}


def _to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (4, 4):
        return matrix
    out = np.eye(4, dtype=np.float64)
    out[: matrix.shape[0], : matrix.shape[1]] = matrix
    return out


def _rect_to_velo(calibration: dict[str, np.ndarray]) -> np.ndarray:
    rectified_from_velo = _to_homogeneous(calibration["rect"]) @ _to_homogeneous(calibration["velo"])
    return np.linalg.inv(rectified_from_velo)


def _convert_points(source: Path, destination: Path) -> int:
    raw = np.fromfile(source, dtype=np.float32)
    if raw.size % 4:
        raise SystemExit(f"{source}: kích thước không chia hết cho 4 float32.")
    points = raw.reshape(-1, 4)
    converted = points.copy()
    converted[:, 0] = points[:, 1]
    converted[:, 1] = -points[:, 0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.astype(np.float32, copy=False).tofile(destination)
    return len(converted)


def _label_fields(parts: list[str], tracking: bool) -> tuple[str, float, float, float, float, float, float, float] | None:
    try:
        if tracking:
            label = parts[2]
            h, w, length = (float(v) for v in parts[10:13])
            x, y, z = (float(v) for v in parts[13:16])
            rotation_y = float(parts[16])
        else:
            label = parts[0]
            h, w, length = (float(v) for v in parts[8:11])
            x, y, z = (float(v) for v in parts[11:14])
            rotation_y = float(parts[14])
    except (IndexError, ValueError):
        return None
    if label not in KITTI_LABEL_TO_VF:
        return None
    return label, h, w, length, x, y, z, rotation_y


def _label_lines(frame: SourceFrame, calibration: dict[str, np.ndarray] | None, tracking: bool) -> tuple[list[str], int]:
    if frame.label_file is None:
        return [], 0
    if calibration is None:
        raise ValueError(f"{frame.label_file}: có nhãn nhưng không có calibration để đổi camera → lidar.")
    rect_to_velo = _rect_to_velo(calibration)
    lines: list[str] = []
    skipped = 0
    for raw_line in frame.label_file.read_text(encoding="utf-8-sig").splitlines():
        parts = raw_line.split()
        if tracking and (frame.tracking_index is None or not parts or not parts[0].isdigit() or int(parts[0]) != frame.tracking_index):
            continue
        fields = _label_fields(parts, tracking)
        if fields is None:
            skipped += 1
            continue
        _label, height, width, length, x, y, z, rotation_y = fields
        # KITTI location is the bottom centre in rectified camera coordinates;
        # camera +Y points down, hence y - h/2 is the 3-D centre.
        center_rect = np.array([x, y - height / 2.0, z, 1.0], dtype=np.float64)
        center_velo = (rect_to_velo @ center_rect)[:3]
        length_axis_rect = np.array([math.sin(rotation_y), 0.0, math.cos(rotation_y)], dtype=np.float64)
        length_axis_velo = rect_to_velo[:3, :3] @ length_axis_rect
        length_axis_vf = KITTI_TO_VF @ length_axis_velo
        yaw = math.atan2(float(length_axis_vf[1]), float(length_axis_vf[0])) % (2.0 * math.pi)
        vf_label, class_id = KITTI_LABEL_TO_VF[_label]
        lines.append(
            f"{center_velo[1]:.6f} {-center_velo[0]:.6f} {center_velo[2]:.6f} "
            f"{length:.6f} {width:.6f} {height:.6f} {yaw:.8f} {class_id}"
        )
    return lines, skipped


def _save_image(source: Path, destination: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Thiếu Pillow để chuyển ảnh KITTI sang JPG; chạy pip install -r requirements.txt") from exc
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(destination, format="JPEG", quality=95)
            return rgb.width, rgb.height
    except OSError as exc:
        raise SystemExit(f"Không đọc được ảnh KITTI {source}: {exc}") from exc


def _write_calib(destination: Path, calibration: dict[str, np.ndarray], camera_name: str,
                 image_size: tuple[int, int] | None, source: str) -> None:
    camera_to_vf = _to_homogeneous(KITTI_TO_VF) @ _rect_to_velo(calibration)
    p2 = calibration["P2"]
    payload = {
        "meta_info": {
            "image_size": list(image_size or (1920, 1020)),
            "source": source,
            "coordinate_transform": "VF x=KITTI y, VF y=-KITTI x, VF z=KITTI z",
        },
        "calib": {
            camera_name: {
                "cam_to_velo": camera_to_vf[:3, :].tolist(),
                "cam_intrinsic": p2[:, :3].tolist(),
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        },
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _all_frame_names(sequence_count: int, chunk_size: int) -> str:
    return f"s01f00..s{sequence_count:02d}f{chunk_size - 1:02d}"


def convert(args: argparse.Namespace) -> int:
    source_root, streams, mode = discover_streams(Path(args.input))
    output = Path(args.output).expanduser().resolve()
    if output == source_root or output.is_relative_to(source_root):
        raise SystemExit("Output không được nằm trong thư mục KITTI input.")
    if args.chunk_size < 1 or args.chunk_size > 99:
        raise SystemExit("--chunk-size phải trong khoảng 1..99 để tên frame VF giữ dạng f00..f98.")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise SystemExit("--max-sequences phải lớn hơn 0.")
    if not re.fullmatch(r"cam\d{2}", args.camera_name):
        raise SystemExit("--camera-name phải có dạng cam01, cam03, ...")
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output đã tồn tại: {output}. Dùng --force nếu muốn tạo lại.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(source_root),
        "source_format": f"kitti-{mode}",
        "coordinate_transform": {"x_vf": "y_kitti", "y_vf": "-x_kitti", "z_vf": "z_kitti", "intensity": "unchanged"},
        "chunk_size": args.chunk_size,
        "sequences": [],
        "skipped_labels": 0,
        "points": 0,
        "frames": 0,
    }
    output_sequence = 0
    stopped = False
    for stream in streams:
        for offset in range(0, len(stream.frames), args.chunk_size):
            if args.max_sequences is not None and output_sequence >= args.max_sequences:
                stopped = True
                break
            chunk = stream.frames[offset:offset + args.chunk_size]
            output_sequence += 1
            sequence_dir = output / f"sequence_{output_sequence:02d}"
            lidar_dir = sequence_dir / "lidar"
            image_dir = sequence_dir / "cameras" / args.camera_name
            sequence_info = {
                "output_sequence": output_sequence,
                "source_stream": stream.name,
                "source_start": offset,
                "frames": len(chunk),
            }
            calib_written = False
            image_size: tuple[int, int] | None = None
            for frame in chunk:
                out_lidar = lidar_dir / f"{frame.stamp}.bin"
                manifest["points"] += _convert_points(frame.lidar, out_lidar)
                manifest["frames"] += 1
                if args.images and frame.image is not None:
                    image_size = _save_image(frame.image, image_dir / f"{frame.stamp}.jpg")
                if args.labels and frame.label_file is not None:
                    try:
                        calibration = _read_calibration(frame.calibration) if frame.calibration is not None else None
                        label_lines, skipped = _label_lines(frame, calibration, mode == "tracking")
                    except (OSError, ValueError) as exc:
                        if args.strict_labels:
                            raise SystemExit(str(exc)) from exc
                        label_lines, skipped = [], 1
                    if label_lines or frame.label_file is not None:
                        label_path = sequence_dir / f"{frame.stamp}.txt"
                        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
                    manifest["skipped_labels"] += skipped
                if args.write_calib and not calib_written and frame.calibration is not None:
                    try:
                        calibration = _read_calibration(frame.calibration)
                        _write_calib(sequence_dir / "calib.json", calibration, args.camera_name, image_size, str(frame.calibration))
                        calib_written = True
                    except (OSError, ValueError) as exc:
                        if args.strict_labels:
                            raise SystemExit(str(exc)) from exc
            sequence_info["calibration"] = calib_written
            sequence_info["images"] = bool(args.images and any(frame.image is not None for frame in chunk))
            sequence_info["labels"] = bool(args.labels and any(frame.label_file is not None for frame in chunk))
            manifest["sequences"].append(sequence_info)
        if stopped:
            break
    if not manifest["frames"]:
        raise SystemExit("Không có frame nào được chuyển đổi.")
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Đã chuyển {manifest['frames']} frame, {manifest['points']} điểm → {output}")
    print(f"Tạo {len(manifest['sequences'])} sequence; coordinate: x_vf=y_kitti, y_vf=-x_kitti, z_vf=z_kitti")
    if manifest["skipped_labels"]:
        print(f"Cảnh báo: bỏ qua {manifest['skipped_labels']} dòng nhãn không hỗ trợ/thiếu calibration.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    vf.configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="gốc KITTI, hoặc thư mục training/testing")
    parser.add_argument("--output", required=True, help="thư mục output Data-VF")
    parser.add_argument("--chunk-size", type=int, default=60, help="số frame tối đa mỗi sequence (mặc định: 60)")
    parser.add_argument("--max-sequences", type=int, help="chỉ ghi N sequence đầu tiên")
    parser.add_argument("--camera-name", default="cam05", help="camera VF nhận image_2/image_02 (mặc định: cam05)")
    parser.add_argument("--no-images", dest="images", action="store_false", help="không chuyển ảnh sang JPG")
    parser.add_argument("--no-labels", dest="labels", action="store_false", help="không chuyển label_2/label_02")
    parser.add_argument("--no-calib", dest="write_calib", action="store_false", help="không ghi calib.json")
    parser.add_argument("--strict-labels", action="store_true", help="dừng nếu nhãn có lỗi hoặc thiếu calibration")
    parser.add_argument("--force", action="store_true", help="xóa output hiện có rồi tạo lại")
    parser.set_defaults(images=True, labels=True, write_calib=True)
    return convert(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

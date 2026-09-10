"""Scientific PNG16/TIFF16/TIFF32 export from linear result arrays (W13).

The image is the atomic commit point. Immutable generation-named companions
are published first, so an interrupted replacement never pairs an old image
with new metadata. A process kill can leave unreferenced companions; it cannot
publish a partial image. No optional compression runtimes are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import uuid
import zlib

import numpy as np
import tifffile

from planetrecon.result import ReconstructionResult

TIFF_JSON_HEADER = "PlanetRecon\n"


def parse_tiff_description(text: str) -> dict:
    """Parse JSON from a PlanetRecon TIFF ImageDescription."""
    if text.startswith(TIFF_JSON_HEADER):
        text = text[len(TIFF_JSON_HEADER):]
    return json.loads(text)


class ExportCancelled(Exception):
    pass


@dataclass(frozen=True)
class ExportConfig:
    encoding: str = "tiff32"
    black: float | None = None
    white: float | None = None
    display_gamma: float | None = None

    def __post_init__(self):
        if self.encoding not in ("png16", "tiff16", "tiff32"):
            raise ValueError("encoding must be png16, tiff16 or tiff32")
        if self.encoding == "tiff32":
            if any(v is not None for v in (self.black, self.white, self.display_gamma)):
                raise ValueError("float TIFF preserves linear scale; mapping/gamma must be omitted")
        else:
            if (self.black is None or self.white is None
                    or not math.isfinite(self.black) or not math.isfinite(self.white)
                    or not 0 < self.white - self.black < float("inf")):
                raise ValueError("integer export requires finite black < white with finite range")
        if self.display_gamma is not None:
            if not math.isfinite(self.display_gamma) or not 0.1 <= self.display_gamma <= 10:
                raise ValueError("display gamma must be finite and in [0.1, 10]")


@dataclass(frozen=True)
class ExportReport:
    path: Path
    sidecar: Path
    coverage_path: Path | None
    metadata: dict


def _collapse_replicated_rgb(array):
    """Keep true per-channel masks; store replicated RGB planes as spatial maps."""
    array = np.ascontiguousarray(array)
    if array.ndim == 3 and array.shape[-1] == 3:
        first = array[..., 0]
        if np.array_equal(first, array[..., 1]) and np.array_equal(first, array[..., 2]):
            return np.ascontiguousarray(first)
    return array


def _prepare(result, config):
    if result.spatial_stride != 1:
        raise ValueError("export requires a full-resolution result, not a downsampled preview")
    image = np.asarray(result.image)
    if (image.ndim not in (2, 3) or not all(image.shape)
            or image.dtype.kind not in "fiu"
            or (image.ndim == 3 and image.shape[-1] != 3)):
        raise ValueError("export requires a nonempty mono or RGB numeric image")
    expected_order = "mono" if image.ndim == 2 else "RGB"
    if result.channel_order != expected_order:
        raise ValueError("image shape and channel order disagree; expected mono or RGB")
    image = np.array(image, dtype=np.float64, copy=True)
    coverage = np.array(result.coverage, dtype=np.float64, copy=True)
    valid = np.array(result.validity, copy=True)
    allowed = (image.shape, image.shape[:2])
    if valid.shape not in allowed or valid.dtype.kind != "b" or coverage.shape not in allowed:
        raise ValueError("validity must be boolean and masks must match image or spatial shape")
    if not np.all(np.isfinite(coverage)) or np.any(coverage < 0):
        raise ValueError("coverage must be finite and non-negative")
    if image.ndim == 3:
        if valid.ndim == 2:
            valid = valid[..., None]
        if coverage.ndim == 2:
            coverage = coverage[..., None]
    coverage = np.broadcast_to(coverage, image.shape).copy()
    valid = np.broadcast_to(valid, image.shape).copy()
    nonfinite = ~np.isfinite(image)
    valid &= ~nonfinite & (coverage > 0)
    layers = {}
    for name, array in result.layer_coverage.items():
        arr = np.array(array, dtype=np.float64, copy=True)
        if arr.shape != image.shape[:2] or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"invalid coverage for layer {name!r}")
        layers[name] = arr
    stats = {"invalid_samples": int(np.count_nonzero(~valid)),
             "nonfinite_samples": int(np.count_nonzero(nonfinite)),
             "invalid_pixels": int(np.count_nonzero(np.any(~valid, axis=-1) if image.ndim == 3 else ~valid)),
             "clipped_low_samples": 0, "clipped_high_samples": 0,
             "clipped_pixels": 0}
    mapping = None
    if config.encoding == "tiff32":
        if np.any(np.abs(image[valid]) > np.finfo(np.float32).max):
            raise ValueError("valid intensity exceeds float32 range")
        image[~valid] = np.nan
        pixels = image.astype(np.float32)
    else:
        below, above = valid & (image < config.black), valid & (image > config.white)
        clipped = below | above
        stats.update(clipped_low_samples=int(np.count_nonzero(below)),
                     clipped_high_samples=int(np.count_nonzero(above)),
                     clipped_pixels=int(np.count_nonzero(np.any(clipped, axis=-1) if image.ndim == 3 else clipped)))
        image[~valid] = config.black
        unit = (np.clip(image, config.black, config.white) - config.black) / (config.white - config.black)
        if config.display_gamma is not None:
            unit = unit ** (1.0 / config.display_gamma)
        pixels = np.floor(unit * 65535.0 + 0.5).astype(np.uint16)
        mapping = {"black": float(config.black), "white": float(config.white),
                   "rounding": "nearest, ties upward", "shared_across_channels": True,
                   "formula": "65535 * clip((linear - black) / (white - black), 0, 1) ** (1 / gamma)",
                   "gamma": config.display_gamma or 1.0}
    # Do not embed a JSON "shape" key: tifffile treats ImageDescription containing
    # '"shape":' as its own series metadata and reports the file as corrupted.
    metadata = {"schema": "planetrecon-export", "schema_version": "1.1",
                "encoding": config.encoding, "image_shape": list(image.shape),
                "rendering": "display-rendered" if config.display_gamma is not None else "scientific-linear",
                "transfer": {"function": "power" if config.display_gamma is not None else "linear",
                             "gamma": config.display_gamma or 1.0,
                             "colour_primaries": "unspecified; no colour-space conversion"},
                "mapping": mapping, "counts": stats, "resampling": "none",
                "invalid_policy": "NaN" if config.encoding == "tiff32" else "zero with separate validity mask",
                "coverage_description": "per-sample accumulation weights; not a calibrated uncertainty",
                "result": result.metadata()}
    return pixels, _collapse_replicated_rgb(valid), _collapse_replicated_rgb(coverage), layers, metadata


def _chunk(stream, kind, payload):
    stream.write(struct.pack(">I", len(payload)))
    stream.write(kind)
    stream.write(payload)
    stream.write(struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def _write_png(stream, pixels, metadata, check_cancel=lambda: None):
    """Dedicated true 16-bit network-order PNG writer; lossless zlib/filter 0."""
    height, width = pixels.shape[:2]
    stream.write(b"\x89PNG\r\n\x1a\n")
    _chunk(stream, b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0 if pixels.ndim == 2 else 2, 0, 0, 0))
    _chunk(stream, b"gAMA", struct.pack(">I", round(100000 / metadata["transfer"]["gamma"])))
    _chunk(stream, b"iTXt", b"PlanetRecon\0\0\0\0\0" + _json(metadata))
    compressor = zlib.compressobj()
    for row in pixels:
        check_cancel()
        encoded = compressor.compress(b"\0" + row.astype(">u2").tobytes())
        if encoded:
            _chunk(stream, b"IDAT", encoded)
    _chunk(stream, b"IDAT", compressor.flush())
    _chunk(stream, b"IEND", b"")


def _write_tiff(stream, pixels, valid, coverage, layers, metadata, check_cancel=lambda: None):
    # Uncompressed TIFF uses no imagecodecs/FFmpeg dependency. BigTIFF when needed.
    # The visible image is a single IFD. Masks live in a companion file so ordinary
    # viewers are not asked to convert 64-bit coverage pages as extra images.
    size = 0 if pixels is None else pixels.nbytes
    if valid is not None:
        size += valid.nbytes + coverage.nbytes + sum(a.nbytes for a in layers.values())
    with tifffile.TiffWriter(stream, byteorder="<", bigtiff=size > 2**32 - 2**25) as writer:
        def page(array, description, *, image=False):
            check_cancel()
            text = _json(description).decode("ascii")
            if image:
                text = TIFF_JSON_HEADER + text
            writer.write(np.ascontiguousarray(array),
                         photometric="rgb" if array.ndim == 3 else "minisblack",
                         planarconfig="contig" if array.ndim == 3 else None,
                         compression=None, metadata=None,
                         description=text,
                         software="PlanetRecon W13", extratags=[(274, "H", 1, 1, False)])
        if pixels is not None:
            page(pixels, metadata, image=True)
        if valid is not None:
            page(valid.astype(np.uint8, copy=False),
                 {"role": "validity", "values": "0 invalid, 1 valid", "export_id": metadata["export_id"]})
            page(np.ascontiguousarray(coverage, dtype=np.float64),
                 {"role": "coverage", "units": "accumulation weight", "export_id": metadata["export_id"]})
            for name, array in sorted(layers.items()):
                page(array, {"role": "layer_coverage", "layer": name,
                             "units": ("independent equal-variance sample equivalents"
                                       if name.startswith("iid_effective_samples_") else "accumulation weight"),
                             "export_id": metadata["export_id"]})


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii")


def _staged_file(path, writer):
    stream = tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{path.name}-", suffix=".tmp",
                                         dir=path.parent, delete=False)
    name = stream.name
    try:
        with stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _publish(staged, destination, overwrite=False):
    if overwrite:
        os.replace(staged, destination)
    else:
        # Atomic no-clobber, including another writer racing after preflight.
        os.link(staged, destination)


def export_result(result: ReconstructionResult, path: Path, config: ExportConfig = ExportConfig(),
                  *, overwrite: bool = False, should_cancel=None) -> ExportReport:
    """Write a detached result; overwrite=True means the caller obtained consent.

    Call on an owned full-resolution snapshot (e.g. a pipeline callback or loaded
    checkpoint), not a concurrently mutated accumulator or copy_preview().
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise ExportCancelled("save cancelled")

    check_cancel()
    path = Path(path)
    suffixes = (".png",) if config.encoding == "png16" else (".tif", ".tiff")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"{config.encoding} requires suffix {' or '.join(suffixes)}")
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(path)
    pixels, valid, coverage, layers, metadata = _prepare(result, config)
    check_cancel()
    generation = uuid.uuid4().hex
    sidecar = path.with_name(f"{path.name}.{generation}.json")
    masks = path.with_name(f"{path.name}.{generation}.coverage.tif")
    metadata.update(export_id=generation, sidecar=sidecar.name,
                    coverage_file=masks.name,
                    encoder={"png": "PlanetRecon PNG16 1.0 / zlib", "tiff": f"tifffile {tifffile.__version__}"})
    # Validate serialization before creating files.
    _json(metadata)
    staged, companions = [], []
    committed = False
    try:
        temp = _staged_file(masks, lambda s: _write_tiff(s, None, valid, coverage, layers, metadata, check_cancel))
        staged.append(temp)
        check_cancel()
        _publish(temp, masks)
        companions.append(masks)
        write = (lambda s: _write_png(s, pixels, metadata, check_cancel)) if config.encoding == "png16" else (
            lambda s: _write_tiff(s, pixels, None, None, None, metadata, check_cancel))
        image_temp = _staged_file(path, write)
        staged.append(image_temp)
        check_cancel()
        digest = hashlib.sha256()
        with image_temp.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                check_cancel()
                digest.update(block)
        side_metadata = metadata | {"image_sha256": digest.hexdigest()}
        temp = _staged_file(sidecar, lambda s: s.write(_json(side_metadata) + b"\n"))
        staged.append(temp)
        check_cancel()
        _publish(temp, sidecar)
        companions.append(sidecar)
        check_cancel()
        _publish(image_temp, path, overwrite)
        committed = True
    finally:
        for temp in staged:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass  # An unreferenced staging file must not invalidate a committed image.
        if not committed:
            for companion in companions:
                companion.unlink(missing_ok=True)
    return ExportReport(path, sidecar, masks, side_metadata)

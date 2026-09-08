"""Explicit read-only compatibility bridge for an exact archived input family.

This does not change gate_errors, certificates or original generation identities.
The reviewed bridge binds immutable input bytes and the complete conservative
simulation/validation dependency set. New solvers need independent qualification.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT/'results/r10-full-grid-inputs/compatibility.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dependency_errors(bridge, root=ROOT):
    return [f'changed simulation/validation dependency: {name}'
            for name, expected in bridge['dependencies'].items()
            if not (root/name).is_file() or digest(root/name) != expected]


def archived_input_errors(path):
    from planetrecon.hdf5io import schema_errors
    from planetrecon.provenance import (fingerprint_from_h5, load_stored_certificate,
        certificate_is_current, fingerprints_compatible, current_fingerprint_for_file,
        PHYSICS_KEYS)
    bridge = json.loads(BRIDGE.read_text())
    errors = dependency_errors(bridge)
    manifest_path = ROOT/bridge['input_manifest']
    if digest(manifest_path) != bridge['input_manifest_sha256']:
        errors.append('changed archived input manifest')
    if errors:
        return errors
    manifest = json.loads(manifest_path.read_text())
    identity = digest(path)
    matches = [r for r in manifest['files'] if r['sha256'] == identity]
    if len(matches) != 1 or matches[0]['gate_errors']:
        return ['input is not an exact certified member of the archived family']
    errors = schema_errors(path)
    if errors:
        return errors
    generated = fingerprint_from_h5(path)
    certificate = load_stored_certificate(path)
    if generated['source_hash'] != bridge['original_package_source_hash']:
        errors.append('unexpected original generation source')
    if not certificate or not certificate_is_current(certificate, generated):
        errors.append('original certificate does not match original generation')
    if not certificate_is_current(matches[0]['method_fingerprint'], generated):
        errors.append('input fingerprint differs from original manifest')
    current = current_fingerprint_for_file(generated)
    if not fingerprints_compatible(generated, current, keys=PHYSICS_KEYS+('simulator_operator_version',)):
        errors.append('current physics/sampling differs from archived input')
    return errors

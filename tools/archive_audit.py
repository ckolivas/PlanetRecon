"""Losslessly compress generated case traces and update their artifact hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def archive(directory):
    report_path = directory/'report.json'
    report = json.loads(report_path.read_text())
    for entry in report['cases']:
        path = directory/entry['path']
        if path.suffix == '.gz':
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'case hash mismatch: {path}')
        compressed = path.with_suffix(path.suffix+'.gz')
        compressed.write_bytes(gzip.compress(raw, mtime=0))
        entry['uncompressed_sha256'] = entry['sha256']
        entry['sha256'] = hashlib.sha256(compressed.read_bytes()).hexdigest()
        entry['path'] = compressed.name
        if gzip.decompress(compressed.read_bytes()) != raw:
            raise ValueError(f'archive verification failed: {path}')
        path.unlink()
    report['case_encoding'] = 'gzip JSON; lossless archival after execution'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    protocol_path = directory/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if 'parent_report_sha256' in protocol:
        protocol.setdefault('parent_report_original_sha256', protocol['parent_report_sha256'])
        protocol['parent_report_sha256'] = hashlib.sha256((Path(protocol['parent'])/'report.json').read_bytes()).hexdigest()
        protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    for directory in parser.parse_args().directories:
        archive(directory)

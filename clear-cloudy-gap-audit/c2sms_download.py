"""Download the complete registered C2S-MS S2 inventory; never skip failures."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

A = Path(__file__).resolve().parent
R = A / 'results'
ROOT = Path(os.environ.get('C2SMS_ROOT', '/content/c2sms_data'))
inventory = json.loads((R / 'c2sms_floods_input_inventory.json').read_text())
OUT = R / 'c2sms_floods_download_manifest.json'
assert not OUT.exists(), 'STOP completed download manifest already exists'
assert (A / 'C2SMS_FROZEN_CONTROL_RULE.md').exists()
ROOT.mkdir(parents=True, exist_ok=True)
assert sum(x['bytes'] for x in inventory['files']) + 3 * 2**30 < shutil.disk_usage(ROOT).free

def fetch(row):
    key = row['key']
    assert key.startswith('c2smsfloods/chips/') and '/s2/' in key and '..' not in Path(key).parts
    dest = ROOT / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        url = 'https://radiant-mlhub.s3.us-west-2.amazonaws.com/' + urllib.parse.quote(key, safe='/')
        with urllib.request.urlopen(url, timeout=120) as response, dest.open('xb') as f:
            shutil.copyfileobj(response, f)
    assert dest.stat().st_size == row['bytes'], 'STOP incomplete file: ' + key
    blob = dest.read_bytes()
    etag = row['etag'].strip('"')
    assert re.fullmatch('[0-9a-f]{32}', etag), 'STOP ETag is not a simple MD5: ' + key
    assert hashlib.md5(blob).hexdigest() == etag, 'STOP source ETag mismatch: ' + key
    return dict(key=key, path=str(dest), bytes=len(blob), etag=etag, sha256=hashlib.sha256(blob).hexdigest())

records = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for i, row in enumerate(pool.map(fetch, inventory['files'])):
        records.append(row)
        if (i + 1) % 300 == 0:
            print('C2SMS DOWNLOAD', i + 1, '/', len(inventory['files']), flush=True)

assets = A / 'prithvi_assets'
assets.mkdir(exist_ok=True)
model_files = inventory['model_files']
for row in model_files:
    dest = assets / row['name']
    if not dest.exists():
        url = 'https://huggingface.co/' + inventory['model_id'] + '/resolve/' + inventory['model_revision'] + '/' + row['name']
        done = 0
        with urllib.request.urlopen(url, timeout=240) as response, dest.open('xb') as f:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done % (128 * 1024 * 1024) == 0:
                    print('FROZEN PRITHVI DOWNLOAD', done, flush=True)
    assert dest.stat().st_size == row['bytes'], 'STOP incomplete model asset'
    h = hashlib.sha256()
    with dest.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == row['sha256'], 'STOP model asset checksum mismatch'

with OUT.open('x') as f:
    json.dump(dict(dataset='C2S-MS Floods', doi='10.34911/rdnt.oz32gz', complete=True,
                   files=records, model_id=inventory['model_id'], model_revision=inventory['model_revision'],
                   model_files=model_files), f, indent=2)
print('C2SMS ALL INPUTS VERIFIED', len(records), flush=True)

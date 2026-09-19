"""Assemble aligned stage-specific exports without pretending they share weights.

python -m src.assemble_evaluation --reduced-dir ... --twelve-dir ... \
    --lead-set 2-lead --split val --out-dir artifacts/evaluation/paired-val

Repeat with the same two frozen checkpoints for test AFTER policy fitting.
The bundle identity hashes the explicit lead-to-checkpoint mapping; validation
and audit must have identical mappings. Source artifacts remain hash-verified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from .config import LEAD_SUBSETS
from .evaluate import atomic_json_dump, sha256_file


def verify_sources(directory: Path, manifest: dict) -> None:
    if manifest.get('checkpoint', {}).get('kind') != 'stage_model_bundle':
        return
    mapping = manifest['checkpoint']['stage_checkpoint_sha256']
    identity = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    if identity != manifest['checkpoint']['sha256']:
        raise ValueError('stage model bundle identity mismatch')
    for source in manifest['stage_sources']:
        path = Path(source['path'])
        if not path.is_file() or sha256_file(path) != source['sha256']:
            raise ValueError('stage source artifact hash mismatch')
    for name, digest in manifest['bundle_hashes'].items():
        if sha256_file(directory / name) != digest:
            raise ValueError('assembled prediction hash mismatch')


def assemble(reduced_dir: Path, twelve_dir: Path, lead_set: str, split: str, out_dir: Path) -> Path:
    if lead_set not in LEAD_SUBSETS or lead_set == '12-lead' or split not in ('val', 'test'):
        raise ValueError('choose a canonical reduced lead set and val/test split')
    if out_dir.exists():
        raise FileExistsError(out_dir)
    sources, manifests, frames, inventories = [], {}, {}, {}
    for lead, directory in ((lead_set, reduced_dir.resolve()), ('12-lead', twelve_dir.resolve())):
        path = directory / 'manifest.json'
        manifest = json.loads(path.read_text())
        evaluation = manifest['evaluation']
        checkpoint = manifest['checkpoint']
        if (manifest.get('status') != 'complete' or evaluation.get('splits') != [split]
                or evaluation.get('max_n') is not None
                or bool(evaluation.get('test_evaluated')) != (split == 'test')):
            raise ValueError('stage sources must be complete, unlimited, single-split exports')
        if evaluation.get('lead_sets', {}).get(lead) != list(LEAD_SUBSETS[lead]):
            raise ValueError('stage export lead definition differs')
        if checkpoint.get('kind') == 'stage_model_bundle' or not checkpoint.get('sha256'):
            raise ValueError('stage source must identify one frozen model')
        regime = checkpoint.get('training_lead_set', 'legacy-random')
        if regime not in ('random', 'legacy-random', lead):
            raise ValueError('fixed model cannot supply a different stage lead set')
        prediction = directory / 'predictions' / f'{split}__{lead}.csv.gz'
        inventory = directory / f'records__{split}.csv.gz'
        frame = pd.read_csv(prediction)
        if frame.empty or frame[['diagnosis_group', 'diagnosis', 'ecg_id']].duplicated().any():
            raise ValueError('empty/duplicate stage predictions')
        if not ((frame.split == split) & (frame.lead_set == lead)).all():
            raise ValueError('stage prediction lead/split mismatch')
        frames[lead] = frame
        inventories[lead] = pd.read_csv(inventory).sort_values('ecg_id').reset_index(drop=True)
        manifests[lead] = manifest
        for source in (path, prediction, inventory):
            sources.append({'lead_set': lead, 'path': str(source), 'sha256': sha256_file(source)})
    full, reduced = manifests['12-lead'], manifests[lead_set]
    # Dataset location changes when moving exports between a Mac and Runpod.
    dataset_identity = lambda m: {k:v for k,v in (m.get('dataset') or {}).items() if k != 'root'}
    if full['label_schema'] != reduced['label_schema'] or dataset_identity(full) != dataset_identity(reduced):
        raise ValueError('stage label schemas or dataset identities differ')
    if not inventories[lead_set].equals(inventories['12-lead']):
        raise ValueError('stage ECG/patient metadata populations differ')
    keys = ['diagnosis_group', 'diagnosis', 'ecg_id']
    reference = frames['12-lead'].set_index(keys).sort_index()
    candidate = frames[lead_set].set_index(keys).sort_index()
    if not reference[['patient_id', 'target']].equals(candidate[['patient_id', 'target']]):
        raise ValueError('stage predictions disagree on ECGs, patients, or targets')
    inventory = inventories['12-lead']
    if inventory.ecg_id.duplicated().any() or inventory.patient_id.isna().any():
        raise ValueError('invalid record inventory')
    for lead, frame in frames.items():
        for _, target in frame.groupby(['diagnosis_group', 'diagnosis']):
            actual = target[['ecg_id', 'patient_id']].sort_values('ecg_id').reset_index(drop=True)
            if not actual.equals(inventory[['ecg_id', 'patient_id']]):
                raise ValueError('incomplete stage population')
    mapping = {lead: manifests[lead]['checkpoint']['sha256'] for lead in manifests}
    checkpoint = {'kind': 'stage_model_bundle', 'stage_checkpoint_sha256': mapping,
                  'sha256': hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
                  'calibration_independent': all(m['checkpoint'].get('calibration_independent') is True for m in manifests.values())}
    out_dir.mkdir(parents=True)
    (out_dir / 'predictions').mkdir()
    names = [f'records__{split}.csv.gz']
    shutil.copyfile(twelve_dir / names[0], out_dir / names[0])
    for lead, directory in ((lead_set, reduced_dir), ('12-lead', twelve_dir)):
        name = f'predictions/{split}__{lead}.csv.gz'
        shutil.copyfile(directory / name, out_dir / name)
        names.append(name)
    bundle = {'schema_version': 2, 'status': 'complete', 'checkpoint': checkpoint,
              'label_schema': full['label_schema'], 'dataset': full.get('dataset'),
              'evaluation': {'splits': [split], 'test_evaluated': split == 'test', 'max_n': None,
                             'lead_sets': {lead: list(LEAD_SUBSETS[lead]) for lead in manifests}},
              'stage_sources': sources, 'stage_checkpoints': {lead: m['checkpoint'] for lead, m in manifests.items()},
              'bundle_hashes': {name: sha256_file(out_dir / name) for name in names}}
    atomic_json_dump(bundle, out_dir / 'manifest.json')
    verify_sources(out_dir, bundle)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reduced-dir', type=Path, required=True)
    parser.add_argument('--twelve-dir', type=Path, required=True)
    parser.add_argument('--lead-set', choices=[x for x in LEAD_SUBSETS if x != '12-lead'], required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    print(assemble(args.reduced_dir, args.twelve_dir, args.lead_set, args.split, args.out_dir))


if __name__ == '__main__':
    main()

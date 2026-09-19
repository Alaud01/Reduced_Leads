"""Regression checks for validation precision and AMP-skipped updates."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch
import warnings

import pandas as pd
import numpy as np
import torch

from src.config import LEAD_SUBSETS, RHYTHMS, SUBCODES, SUPERCLASSES, ModelCfg, TrainCfg
from src.train import (
    build_dataloaders,
    build_parser,
    configure_training_regime,
    evaluate_subset,
    load_resume_state,
    make_warmup_scheduler,
    resumable_checkpoint,
    selection_lead_set,
    train_one_epoch,
    training_leads,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, lead_mask):
        return self.weight * x


class TinyLoss:
    def __call__(self, output, batch):
        loss = output.sum() * batch['multiplier']
        return {key: loss for key in (
            'loss', 'loss_super', 'loss_sub', 'loss_rhythm', 'loss_lead_presence',
        )}


class EvalModel(torch.nn.Module):
    def forward(self, x, lead_mask):
        return {key: torch.zeros(2, len(names)) for key, names in (
            ('cls_super', SUPERCLASSES), ('cls_sub', SUBCODES), ('aux_rhythm', RHYTHMS),
        )}


class TrainingAMPTest(unittest.TestCase):
    def test_validation_respects_amp_option_and_device(self):
        batch = {'x': torch.zeros(2, 1), 'lead_mask': torch.ones(2, 1)}
        for key, names in (
            ('y_super', SUPERCLASSES), ('y_sub', SUBCODES), ('y_rhythm', RHYTHMS),
        ):
            batch[key] = torch.zeros(2, len(names))
        for device, use_amp, expected in (
            ('mps', False, False), ('cuda', False, False),
            ('mps', True, True), ('cuda', True, True), ('cpu', True, False),
        ):
            with self.subTest(device=device, use_amp=use_amp), \
                 patch('src.train.PTBXLDataset'), \
                 patch('src.train.DataLoader', return_value=[batch]), \
                 patch('src.train.move_batch', side_effect=lambda b, d: b), \
                 patch('src.train.torch.amp.autocast', return_value=nullcontext()) as autocast:
                evaluate_subset(EvalModel(), pd.DataFrame(), {}, ['I', 'II'],
                                torch.device(device), use_amp=use_amp)
                self.assertEqual(autocast.call_args.kwargs['enabled'], expected)

    def run_batch(self, model, optimizer, scheduler, scaler, multiplier, step):
        batch = {'x': torch.ones(1, 1), 'lead_mask': torch.ones(1, 1),
                 'multiplier': multiplier}
        with warnings.catch_warnings(record=True) as caught:
            _, next_step, logs = train_one_epoch(
                model, TinyLoss(), optimizer, scheduler, [batch], torch.device('cpu'),
                1.0, 1, step, scaler=scaler,
            )
        self.assertFalse(any('lr_scheduler.step()' in str(w.message) for w in caught))
        self.assertEqual(next_step, step + 1)  # logging retains its batch axis
        return logs[0]

    def test_overflow_holds_schedule_then_success_advances_it(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=2)
        original_epoch = scheduler.last_epoch
        log = self.run_batch(model, optimizer, scheduler, scaler, float('inf'), 0)
        self.assertFalse(log['optimizer_updated'])
        self.assertEqual(model.weight.item(), 1.)
        self.assertEqual(scheduler.last_epoch, original_epoch)
        self.assertEqual(optimizer.param_groups[0]['lr'], .1)
        self.assertEqual(scaler.get_scale(), 4.)
        for step in (1, 2):
            log = self.run_batch(model, optimizer, scheduler, scaler, 1., step)
            self.assertTrue(log['optimizer_updated'])
            self.assertEqual(scheduler.last_epoch, original_epoch + step)
        self.assertLess(model.weight.item(), 1.)
        self.assertEqual(scaler.get_scale(), 8.)  # growth also counts as success
        self.assertEqual(optimizer.param_groups[0]['lr'], .025)

    def test_fp32_and_disabled_scaler_advance_schedule(self):
        for scaler in (None, torch.amp.GradScaler('cpu', enabled=False)):
            with self.subTest(scaler=scaler):
                model = TinyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
                epoch = scheduler.last_epoch
                log = self.run_batch(model, optimizer, scheduler, scaler, 1., 0)
                self.assertTrue(log['optimizer_updated'])
                self.assertLess(model.weight.item(), 1.)
                self.assertEqual(scheduler.last_epoch, epoch + 1)


class TrainingRegimeTest(unittest.TestCase):
    def test_cli_exposes_random_and_fixed_comparators(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).train_lead_set, None)
        self.assertEqual(
            parser.parse_args(['--train-lead-set', '12-lead']).train_lead_set,
            '12-lead',
        )
        self.assertEqual(
            parser.parse_args(['--train-lead-set', '2-lead']).train_lead_set,
            '2-lead',
        )

    def test_fixed_regime_drives_training_inputs_and_model_selection(self):
        frame = pd.DataFrame(
            {'fold_role': ['train', 'train', 'val']}, index=[1, 2, 3],
        )
        norm = {'mean': np.zeros(12), 'std': np.ones(12)}
        cfg = TrainCfg(batch_size=1, training_lead_set='2-lead')
        self.assertEqual(configure_training_regime(cfg), '2-lead')
        self.assertEqual(cfg.eval_subsets, [list(LEAD_SUBSETS['2-lead'])])
        self.assertEqual(cfg.w_lead_presence, 0.0)
        with patch('src.train.PTBXLDataset') as dataset, \
             patch('src.train.DataLoader', side_effect=lambda ds, **kwargs: ds):
            build_dataloaders(frame, norm, cfg, seed=42)
        self.assertEqual(training_leads(cfg), list(LEAD_SUBSETS['2-lead']))
        self.assertEqual(selection_lead_set(cfg), '2-lead')
        self.assertEqual(dataset.call_args_list[0].kwargs['keep_leads'], ['I', 'II'])
        self.assertNotIn('drop_max', dataset.call_args_list[0].kwargs)
        self.assertEqual(dataset.call_args_list[1].kwargs['keep_leads'], LEAD_SUBSETS['2-lead'])

    def test_random_regime_keeps_augmentation_and_selects_twelve_lead(self):
        cfg = TrainCfg(training_lead_set='random', drop_min=1, drop_max=10)
        self.assertIsNone(training_leads(cfg))
        self.assertEqual(selection_lead_set(cfg), '12-lead')

    def test_resumable_checkpoint_round_trip_restores_training_state(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = make_warmup_scheduler(optimizer, 1, 10)
        (model(torch.ones(1), torch.ones(1)).sum()).backward()
        optimizer.step()
        scheduler.step()
        payload = resumable_checkpoint(
            epoch=3, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=None, model_cfg=ModelCfg(), train_cfg=TrainCfg(),
            norm={'mean': np.zeros(12), 'std': np.ones(12)},
            label_schema={'superclass': [], 'subcode': [], 'rhythm': []},
            seed=42, run_id='run', global_step=17, best_metric=.81,
            selection_lead='12-lead', current_metric=.8,
        )

        restored_model = TinyModel()
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=.1)
        restored_scheduler = make_warmup_scheduler(restored_optimizer, 1, 10)
        start_epoch, global_step, best = load_resume_state(
            payload, restored_model, restored_optimizer, restored_scheduler, None,
        )
        self.assertEqual((start_epoch, global_step, best), (4, 17, .81))
        self.assertEqual(restored_model.weight.item(), model.weight.item())
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_legacy_checkpoint_cannot_claim_exact_resume(self):
        with self.assertRaisesRegex(ValueError, 'predates resumable'):
            load_resume_state(
                {'model_state': {}}, TinyModel(),
                torch.optim.SGD(TinyModel().parameters(), lr=.1),
                torch.optim.lr_scheduler.StepLR(
                    torch.optim.SGD(TinyModel().parameters(), lr=.1), 1,
                ),
                None,
            )



class ResearchResumeTest(unittest.TestCase):
    def test_next_dropout_update_matches_after_serialized_resume(self):
        import io
        from src.train import seed_everything
        seed_everything(42)
        def setup():
            model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.5), torch.nn.Linear(5, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = make_warmup_scheduler(optimizer, 0, 20)
            return model, optimizer, scheduler
        def step(model, optimizer, scheduler):
            optimizer.zero_grad()
            model(torch.ones(4, 3)).square().mean().backward()
            optimizer.step()
            scheduler.step()
        model, optimizer, scheduler = setup()
        step(model, optimizer, scheduler)
        payload = resumable_checkpoint(
            epoch=1, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
            model_cfg=ModelCfg(), train_cfg=TrainCfg(), norm={'mean': np.zeros(12), 'std': np.ones(12)},
            label_schema={}, seed=42, run_id='test', global_step=1, best_metric=.5,
            selection_lead='2-lead', current_metric=.5)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        step(model, optimizer, scheduler)
        restored, opt, sched = setup()
        buffer.seek(0)
        load_resume_state(torch.load(buffer, weights_only=True), restored, opt, sched, None)
        step(restored, opt, sched)
        for actual, expected in zip(restored.parameters(), model.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])

    def test_epoch_shuffle_and_masks_resume_independently_of_uptime(self):
        from src.training_state import prepare_epoch
        from torch.utils.data import DataLoader, Dataset
        class RandomDataset(Dataset):
            rng = np.random.default_rng()
            def __len__(self): return 12
            def __getitem__(self, i): return i, int(self.rng.integers(1000000))
        def loader():
            return DataLoader(RandomDataset(), batch_size=3, shuffle=True, generator=torch.Generator())
        first = loader()
        prepare_epoch(first, 42, 1)
        list(first)
        prepare_epoch(first, 42, 2)
        uninterrupted = list(first)
        resumed = loader()
        prepare_epoch(resumed, 42, 2)
        for x, y in zip(uninterrupted, list(resumed)):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(x,y)))

    def test_resume_rejects_population_and_label_changes(self):
        from src.training_state import resume_contract, validate_resume
        frame = pd.DataFrame({'fold_role': ['train', 'val'], 'label': [0, 1]}, index=[1,2])
        contract = resume_contract(frame.iloc[:1], frame.iloc[1:], eval_max_n=None, quick=False, total_steps=20)
        schema = {'rhythm': ['AFIB', 'AFLT']}
        checkpoint = {'checkpoint_schema_version': 3, 'label_schema': schema, 'resume_contract': contract}
        validate_resume(checkpoint, schema, contract)
        with self.assertRaisesRegex(ValueError, 'label schema'):
            validate_resume(checkpoint, {'rhythm':['AFLT','AFIB']}, contract)
        for key, value in [('eval_max_n', 128), ('total_steps', 21), ('validation_ecg_ids', [3])]:
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate_resume(checkpoint, schema, {**contract, key: value})

    def test_matched_random_comparator_can_select_two_leads_without_auxiliary_loss(self):
        cfg = TrainCfg(training_lead_set='random', selection_lead='2-lead', w_lead_presence=0.)
        self.assertEqual(configure_training_regime(cfg), '2-lead')
        self.assertEqual(cfg.w_lead_presence, 0.)

    def test_selection_split_reserves_fold_nine_and_rejects_patient_overlap(self):
        from src.training_state import apply_research_split
        frame = pd.DataFrame({'fold_role': ['train', 'train', 'val', 'test']})
        meta = pd.DataFrame({'strat_fold': [1,8,9,10], 'patient_id':[1,2,3,4]})
        result = apply_research_split(frame, meta)
        self.assertEqual(result.fold_role.tolist(), ['train','val','calibration','test'])
        meta.loc[2, 'patient_id'] = 1
        with self.assertRaisesRegex(ValueError, 'patient leakage'):
            apply_research_split(frame, meta)

    def test_training_main_interrupt_resume_matches_uninterrupted(self):
        import contextlib
        import io
        import tempfile
        import json
        from pathlib import Path
        from src import train
        from src.training_state import resume_contract as actual_contract
        class MiniModel(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.dropout = torch.nn.Dropout(.4)
                self.linear = torch.nn.Linear(12, len(SUPERCLASSES)+len(SUBCODES)+len(RHYTHMS))
            def num_parameters(self): return sum(p.numel() for p in self.parameters())
            def forward(self, x, lead_mask):
                out = self.linear(self.dropout(x.mean(-1)))
                a, b = len(SUPERCLASSES), len(SUPERCLASSES)+len(SUBCODES)
                return {'cls_super':out[:,:a], 'cls_sub':out[:,a:b], 'aux_rhythm':out[:,b:],
                        'aux_lead_presence': torch.zeros(len(x), 12)}
        # Keep the real epoch optimizer/scheduler path, using a tiny loss/model.
        class MiniLoss:
            def __init__(self, **kwargs): pass
            def __call__(self, out, batch):
                loss = sum(v.square().mean() for k,v in out.items() if k != 'aux_lead_presence')
                return {key:loss for key in ('loss','loss_super','loss_sub','loss_rhythm','loss_lead_presence')}
        frame = pd.DataFrame({'fold_role':['train']*6+['val']*2+['test']*2,
                              'filename_lr':['dummy']*10}, index=range(10))
        for prefix, labels in [('superclass',SUPERCLASSES),('subcode',SUBCODES),('rhythm',RHYTHMS)]:
            for label in labels: frame[f'{prefix}_{label}'] = np.arange(10)%2
        metadata = pd.DataFrame({'strat_fold':[1,1,1,1,8,8,9,9,10,10], 'patient_id':range(10)})
        real_epoch = train.train_one_epoch
        def interrupted(*args, **kwargs):
            if args[7] == 2: raise RuntimeError('simulated interruption')
            return real_epoch(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch('src.train.load_metadata', return_value=metadata), \
             patch('src.train.build_label_frame', return_value=frame), \
             patch('src.train.compute_train_norm', return_value={'mean':np.zeros(12), 'std':np.ones(12)}), \
             patch('src.train.LeadAwareTransformer', MiniModel), \
             patch('src.train.MultiTaskLoss', MiniLoss), \
             patch('src.train.plot_training_curves'), \
             patch('src.data.load_signal', return_value=np.ones((1000,12),dtype=np.float32)), \
             patch('src.train.resume_contract', side_effect=lambda *a,**kw:actual_contract(*a,**{**kw,'data_root':None})):
            root = Path(tmp)
            base = ['train', '--epochs','2','--batch-size','2','--num-workers','0','--device','cpu','--no-amp', '--train-lead-set','2-lead']
            with patch('sys.argv', base+['--out-dir',str(root/'continuous')]): train.main()
            with patch('sys.argv', base+['--out-dir',str(root/'interrupted')]), \
                 patch('src.train.train_one_epoch', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated'): train.main()
            (root/'interrupted'/'best.pt').unlink()  # last.pt must recover it on migration
            with patch('sys.argv', ['train','--resume',str(root/'interrupted'/'last.pt')]): train.main()
            self.assertTrue((root/'interrupted'/'best.pt').exists())
            a = torch.load(root/'continuous'/'last.pt', weights_only=True)
            b = torch.load(root/'interrupted'/'last.pt', weights_only=True)
            for key in a['model_state']:
                self.assertTrue(torch.equal(a['model_state'][key], b['model_state'][key]), key)
            self.assertEqual(a['global_step'],b['global_step'])
            rows=[json.loads(x) for x in (root/'interrupted'/'train_log.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows if r.get('type')=='epoch'],[1,2])


if __name__ == '__main__':
    unittest.main()

"""Regression checks for validation precision and AMP-skipped updates."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch
import warnings

import pandas as pd
import torch

from src.config import RHYTHMS, SUBCODES, SUPERCLASSES
from src.train import evaluate_subset, train_one_epoch


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


if __name__ == '__main__':
    unittest.main()

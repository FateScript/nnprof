#!/usr/bin/python3
# -*- coding:utf-8 -*-
import unittest
import torch
import torchvision

from nnprof import profile, ProfileMode, InfoTable, TreeTable


class TestProfile(unittest.TestCase):

    model = torchvision.models.alexnet(pretrained=False)
    x = torch.rand([1, 3, 224, 224])

    def test_table_args(self):
        with profile(self.model, profile_memory=True) as prof:
            _ = self.model(self.x)
        sorted_keys = ["cpu_time", "self_cpu_time"]
        for k in sorted_keys:
            table = prof.table(sorted_by=k)
            self.assertIsInstance(table, InfoTable)

        for average in [True, False]:
            table = prof.table(average=average)
            self.assertIsInstance(table, InfoTable)

    def test_profile_mode(self):
        for mode in ProfileMode:
            with profile(self.model, profile_memory=True, mode=mode) as prof:
                _ = self.model(self.x)
            table = prof.table()
            if mode != ProfileMode.LAYER_TREE:
                self.assertIsInstance(table, InfoTable)
            else:
                self.assertIsInstance(table, TreeTable)

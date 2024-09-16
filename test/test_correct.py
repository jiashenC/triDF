import torch
import unittest

import triton

from tridf.kernel.filter import filter_kernel
from tridf.kernel.aggregation import count_nonzero_kernel, block_count_nonzero_kernel
from tridf.kernel.gather import materialize_kernel


class TestCorrect(unittest.TestCase):
    def test_filter(self):
        inp = torch.arange(0, 100, 1).to(torch.int32).to("cuda")
        inp_a, inp_b = inp % 2, inp % 3
        bitmap = torch.ones(100).to(torch.bool).to("cuda")
        grid = lambda meta: (triton.cdiv(100, meta["BLOCK_SIZE"]), )
        filter_kernel[grid](inp_a, bitmap, "=", 0, 100, BLOCK_SIZE=16)
        filter_kernel[grid](inp_b, bitmap, "=", 0, 100, BLOCK_SIZE=16)
        for i, value in enumerate(bitmap.cpu().tolist()):
            if i % 6 == 0:
                self.assertTrue(value)
            else:
                self.assertFalse(value)

    def test_agg_count_zero(self):
        inp = torch.arange(0, 100, 1).to(torch.int32).to("cuda")
        inp_a = inp % 3
        bitmap = torch.ones(100).to(torch.bool).to("cuda")
        grid = lambda meta: (triton.cdiv(100, meta["BLOCK_SIZE"]), )
        filter_kernel[grid](inp_a, bitmap, "=", 0, 100, BLOCK_SIZE=16)
        res = torch.zeros(1).to(torch.int32).to("cuda")
        count_nonzero_kernel[grid](bitmap, res, 100, BLOCK_SIZE=16)
        tot = 0
        for val in inp_a.cpu().tolist():
            if val % 3 == 0:
                tot += 1
        self.assertEqual(res[0], tot)

    def test_agg_block_count_zero(self):
        inp = torch.arange(0, 100, 1).to(torch.int32).to("cuda")
        inp_a = inp % 3
        bitmap = torch.ones(100).to(torch.bool).to("cuda")
        grid = lambda meta: (triton.cdiv(100, meta["BLOCK_SIZE"]), )
        filter_kernel[grid](inp_a, bitmap, "=", 0, 100, BLOCK_SIZE=16)
        res = torch.zeros(100 // 16 + 1).to(torch.int32).to("cuda")
        block_count_nonzero_kernel[grid](bitmap, res, 100, BLOCK_SIZE=16)
        for i, val in enumerate(res.cpu().tolist()):
            count = 0
            for num in range(i * 16, min((i + 1) * 16, 101)):
                if num % 3 == 0:
                    count += 1
            self.assertEqual(val, count)

    def test_materialize(self):
        inp = torch.arange(0, 100, 1).to(torch.int32).to("cuda")
        inp_a = inp % 3
        bitmap = torch.ones(100).to(torch.bool).to("cuda")
        grid = lambda meta: (triton.cdiv(100, meta["BLOCK_SIZE"]), )
        filter_kernel[grid](inp_a, bitmap, "=", 0, 100, BLOCK_SIZE=16)
        block_count_nonzero = torch.zeros(100 // 16 + 1).to(torch.int32).to("cuda")
        block_count_nonzero_kernel[grid](bitmap, block_count_nonzero, 100, BLOCK_SIZE=16)
        prefix_sum = 0
        block_count_nonzero = block_count_nonzero.cpu().tolist()
        for i, num in enumerate(block_count_nonzero):
            block_count_nonzero[i] = num + prefix_sum
            prefix_sum += num
        block_count_nonzero = torch.tensor(block_count_nonzero).to(torch.int32).to("cuda")
        nonzero = torch.zeros(1).to(torch.int32).to("cuda")
        count_nonzero_kernel[grid](bitmap, nonzero, 100, BLOCK_SIZE=16)
        outp = torch.zeros(nonzero).to("cuda")
        materialize_kernel[grid](
            inp,
            bitmap,
            block_count_nonzero,
            outp,
            100,
            BLOCK_SIZE=16
        )
        for i, val in enumerate(outp.cpu().tolist()):
            self.assertEqual(val, i * 3)

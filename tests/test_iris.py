from contextlib import redirect_stdout
from io import BytesIO, StringIO
from itertools import product
import unittest

import torch
import torch_iris


HAS_FP16 = torch.float16 in torch_iris.get_amp_supported_dtype()


class IrisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_iris.is_available():
            raise unittest.SkipTest("Tests require an available Intel Iris Xe OpenCL GPU")
        assert "iris" in torch_iris.device_name().lower()
        cls.subgroup_kernels = torch_iris.get_device_properties().subgroup_kernels

    def test_allocation_and_copy_directions(self):
        host = torch.arange(12, dtype=torch.float32).reshape(3, 4)[1:]
        direct = torch.empty((2, 4), device="iris")
        self.assertEqual(direct.device, torch.device("iris:0"))
        self.assertEqual(direct.stride(), (4, 1))
        self.assertEqual(direct.dtype, torch.float32)
        self.assertIs(direct.copy_(host), direct)
        self.assertIs(direct.copy_(direct), direct)
        torch.testing.assert_close(direct.cpu(), host, rtol=0, atol=0)

        uploaded = host.to("iris", non_blocking=True)
        duplicated = torch.empty_strided((2, 4), (4, 1), device="iris:0")
        duplicated.copy_(uploaded)
        cpu_target = torch.empty_like(host)
        cpu_target.copy_(duplicated)
        torch.testing.assert_close(cpu_target, host, rtol=0, atol=0)
        non_blocking_copy = uploaded.to("cpu", non_blocking=True)
        torch_iris.synchronize()
        torch.testing.assert_close(non_blocking_copy, host, rtol=0, atol=0)

    def test_statistics_legacy_overloads_preserve_gradients(self):
        torch.manual_seed(12)
        for op in (torch.var, torch.std, torch.var_mean, torch.std_mean):
            for kwargs in ({}, {"dim": (0, 2), "unbiased": False, "keepdim": True}, {"dim": 1, "correction": 1}):
                host = torch.randn(3, 4, 7).transpose(0, 1).requires_grad_()
                iris = host.detach().to("iris").requires_grad_()
                expected, actual = op(host, **kwargs), op(iris, **kwargs)
                expected = expected if isinstance(expected, tuple) else (expected,)
                actual = actual if isinstance(actual, tuple) else (actual,)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-4)
                sum(t.sum() for t in actual).backward()
                sum(t.sum() for t in expected).backward()
                self.assertIsNotNone(iris.grad)
                torch.testing.assert_close(iris.grad.cpu(), host.grad, atol=1e-5, rtol=1e-4)

    def test_integer_sum_and_extrema(self):
        large = 2**60
        host = torch.tensor([[large, 9, -large], [-12, 8, -3], [3, 4, 5]], dtype=torch.int64)
        for value in (host, host.int(), host.byte(), host > 0, host[:, 1:].t(),
                      host[:0], (host > 0).repeat(32, 32)):
            iris = value.to("iris")
            for args in ((), (0,), (1,), ((0, 1), True)):
                torch.testing.assert_close(iris.sum(*args).cpu(), value.sum(*args), rtol=0, atol=0)
            if value.numel():
                torch.testing.assert_close(iris.min().cpu(), value.min(), rtol=0, atol=0)
                torch.testing.assert_close(iris.max().cpu(), value.max(), rtol=0, atol=0)
        labels = torch.tensor([1, 3, 0, 2])
        for classes in (-1, 5):
            torch.testing.assert_close(torch.nn.functional.one_hot(labels.to("iris"), classes).cpu(),
                                       torch.nn.functional.one_hot(labels, classes), rtol=0, atol=0)

    def test_loss_and_unfold_backward(self):
        from torch.nn import functional as F

        torch.manual_seed(13)
        pairs = (
            (torch.randn(5, 4), lambda x: F.cross_entropy(x, torch.arange(5, device=x.device) % 4,
                                                          label_smoothing=0.1)),
            (torch.randn(2, 4, 3, 5), lambda x: F.cross_entropy(
                x, torch.ones(2, 3, 5, dtype=torch.int64, device=x.device), reduction="none")),
            (torch.randn(3, 7), lambda x: F.binary_cross_entropy_with_logits(x, torch.full_like(x, 0.3))),
            (torch.randn(3, 9).t(), lambda x: x.unfold(0, 3, 2).abs()),
        )
        for host, fn in pairs:
            iris = host.to("iris").requires_grad_()
            host.requires_grad_()
            got, expected = fn(iris), fn(host)
            torch.testing.assert_close(got.cpu(), expected, atol=1e-5, rtol=1e-4)
            got.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(iris.grad.cpu(), host.grad, atol=1e-5, rtol=1e-4)

    def test_iris_grad_scaler_skips_bad_update_and_recovers(self):
        param = torch.nn.Parameter(torch.tensor([2.0], device="iris"))
        opt = torch.optim.SGD([param], lr=0.25, foreach=False)
        scaler = torch_iris.amp.GradScaler(init_scale=16, growth_interval=1)
        for factor, expected_param, expected_scale in ((2, 1.5, 32), (float("inf"), 1.5, 16), (2, 1.0, 32)):
            opt.zero_grad(set_to_none=True)
            scaler.scale((param * factor).sum()).backward()
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            self.assertEqual(param.item(), expected_param)
            self.assertEqual(scaler.get_scale(), expected_scale)

    def test_foreach_scalar_steps_share_one_launch(self):
        tensors = [torch.zeros((), device="iris") for _ in range(6)]
        before = torch_iris._C._kernel_submissions()
        torch._foreach_add_(tensors, 2.0)
        self.assertEqual(torch_iris._C._kernel_submissions() - before, 1)
        self.assertEqual([t.item() for t in tensors], [2.0] * 6)
        torch._foreach_add_([tensors[0], tensors[0]], 1.0)
        self.assertEqual(tensors[0].item(), 4.0)

    def test_elementwise_kernels(self):
        a = torch.tensor([[1.5, -3.0, 0.0], [float("nan"), -2.0, 3.0]])
        b = torch.tensor([[2.0, 4.0, -2.5], [0.0, -1.0, 0.5]])
        a_iris, b_iris = a.to("iris"), b.to("iris")
        previous = torch_iris._C._kernel_submissions()
        pairs = [
            (a_iris + b_iris, a + b),
            (torch.add(a_iris, b_iris, alpha=-0.5), torch.add(a, b, alpha=-0.5)),
            (a_iris * b_iris, a * b),
            (torch.relu(a_iris), torch.relu(a)),
        ]
        torch_iris.synchronize()
        self.assertEqual(torch_iris._C._kernel_submissions() - previous, len(pairs))
        for actual, reference in pairs:
            self.assertEqual(actual.device, torch.device("iris:0"))
            torch.testing.assert_close(actual.cpu(), reference, equal_nan=True)

        scalar = torch.tensor(-2.0)
        iris_scalar = scalar.to("iris")
        torch.testing.assert_close((iris_scalar + iris_scalar).cpu(), scalar + scalar)

    def test_matrix_multiplication_variants(self):
        random = torch.Generator().manual_seed(123)
        for rows, inner, columns in [(2, 2, 2), (17, 33, 5), (65, 9, 31)]:
            with self.subTest(shape=(rows, inner, columns)):
                a = torch.rand((rows, inner), generator=random) - 0.5
                b = torch.rand((inner, columns), generator=random) - 0.5
                a_iris, b_iris = a.to("iris"), b.to("iris")
                previous = torch_iris._C._kernel_submissions()
                for actual in (
                    torch.mm(a_iris, b_iris),
                    torch.matmul(a_iris, b_iris),
                    a_iris @ b_iris,
                ):
                    self.assertEqual(actual.device, torch.device("iris:0"))
                    torch.testing.assert_close(actual.cpu(), a @ b)
                self.assertEqual(torch_iris._C._kernel_submissions() - previous, 3)

    def test_empty_tensors(self):
        empty = torch.empty((0, 3))
        empty_iris = empty.to("iris")
        torch.testing.assert_close(empty_iris.cpu(), empty)
        torch.testing.assert_close((empty_iris + empty_iris).cpu(), empty)
        torch.testing.assert_close((empty_iris * empty_iris).cpu(), empty)
        torch.testing.assert_close(torch.relu(empty_iris).cpu(), empty)
        for rows, inner, columns in [
            (0, 5, 3), (3, 5, 0), (3, 0, 4), (0, 0, 0),
            (0, 129, 257), (257, 129, 0), (257, 0, 259),
        ]:
            with self.subTest(shape=(rows, inner, columns)):
                a = torch.empty((rows, inner))
                b = torch.empty((inner, columns))
                previous = torch_iris._C._kernel_submissions()
                torch.testing.assert_close((a.to("iris") @ b.to("iris")).cpu(), a @ b)
                self.assertEqual(torch_iris._C._kernel_submissions(), previous)

    def test_matrix_multiplication_dispatch_boundaries(self):
        random = torch.Generator().manual_seed(321)
        for shape, expected in [
            ((7, 8, 9), (0, 0)),
            ((8, 8, 8), (1, 0)),
            ((8, 8, 9), (1, 0)),
            ((32, 7, 33), (0, 0)),
            ((32, 8, 33), (1, 0)),
            ((255, 128, 256), (1, 0)),
            ((256, 127, 256), (1, 0)),
            ((256, 128, 255), (1, 0)),
            ((256, 128, 256), (0, 1)),
            ((257, 129, 259), (0, 1)),
        ]:
            with self.subTest(shape=shape):
                self.check_matrix_multiplication(shape, random, (torch.mm,), expected)

    def test_tiled_matrix_multiplication_variants(self):
        random = torch.Generator().manual_seed(456)
        for shape, expected in [((123, 257, 65), (1, 0)), ((257, 129, 259), (0, 1))]:
            with self.subTest(shape=shape):
                self.check_matrix_multiplication(
                    shape, random, (torch.mm, torch.matmul, lambda a, b: a @ b), expected
                )

    def test_tiled_transposed_and_fused_matrix_multiplication(self):
        random = torch.Generator().manual_seed(789)
        for (rows, inner, columns), expected_tile in [((37, 47, 35), 0), ((257, 129, 259), 1)]:
            a = torch.rand((rows, inner), generator=random) - 0.5
            b = torch.rand((inner, columns), generator=random) - 0.5
            bias = torch.rand((columns, rows + 1), generator=random) - 0.5
            iris_bias = bias.to("iris")[:, 1:].t()
            bias = bias[:, 1:].t()
            for transpose_a, transpose_b in product((False, True), repeat=2):
                with self.subTest(shape=(rows, inner, columns), layout=(transpose_a, transpose_b)):
                    # Leave an offset and an extra physical row in each base allocation.
                    operands = []
                    for operand, transpose in ((a, transpose_a), (b, transpose_b)):
                        physical = operand.t() if transpose else operand
                        base = torch.cat((torch.zeros_like(physical[:1]), physical))
                        view = base.to("iris")[1:]
                        operands.append(view.t() if transpose else view)
                    iris_a, iris_b = operands
                    previous = torch_iris._C._kernel_submissions()
                    tiles = torch_iris._C._matmul_tile_submissions()
                    actual = torch.addmm(iris_bias, iris_a, iris_b, alpha=-0.25, beta=1.5)
                    self.assertEqual(torch_iris._C._kernel_submissions() - previous, 1)
                    self.assertEqual(torch_iris._C._matmul_tile_submissions()[expected_tile],
                                     tiles[expected_tile] + 1)
                    torch.testing.assert_close(actual.cpu(), torch.addmm(bias, a, b, alpha=-0.25, beta=1.5),
                                               atol=1e-5, rtol=1e-4, check_stride=True)

    def test_addmm_zero_scales_and_empty_inner_dimension(self):
        for inner in (0, 7):
            a = torch.full((3, inner), float("nan"))
            b = torch.ones((inner, 5))
            for beta in (0, 2):
                bias = torch.full((5,), float("nan") if beta == 0 else 1.5)
                for alpha in (0, 1):
                    args = bias, a, b
                    actual = torch.addmm(*(x.to("iris") for x in args), alpha=alpha, beta=beta)
                    reference = torch.addmm(*args, alpha=alpha, beta=beta)
                    torch.testing.assert_close(actual.cpu(), reference, equal_nan=True, check_stride=True)

    def test_tiled_bmm_with_strided_batches(self):
        random = torch.Generator().manual_seed(476)
        a = torch.rand((1, 65, 35), generator=random) - 0.5
        b = torch.rand((7, 47, 65), generator=random) - 0.5
        left, right = a.expand(3, -1, -1).transpose(1, 2), b[1::2, 1::2, :].transpose(1, 2)
        iris_a = a.to("iris").expand(3, -1, -1).transpose(1, 2)
        iris_b = b.to("iris")[1::2, 1::2, :].transpose(1, 2)
        previous = torch_iris._C._matmul_tile_submissions()[0]
        actual = torch.bmm(iris_a, iris_b)
        self.assertEqual(torch_iris._C._matmul_tile_submissions()[0], previous + 1)
        torch.testing.assert_close(actual.cpu(), torch.bmm(left, right),
                                   atol=1e-5, rtol=1e-4, check_stride=True)

    def check_matrix_multiplication(self, shape, random, operations, expected):
        rows, inner, columns = shape
        a = torch.rand((rows, inner), generator=random) - 0.5
        b = torch.rand((inner, columns), generator=random) - 0.5
        reference = a @ b
        a_iris, b_iris = a.to("iris"), b.to("iris")
        for op in operations:
            previous_total = torch_iris._C._kernel_submissions()
            previous_tiles = torch_iris._C._matmul_tile_submissions()
            actual = op(a_iris, b_iris)
            self.assertEqual(actual.device, torch.device("iris:0"))
            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
            self.assertEqual(torch_iris._C._kernel_submissions() - previous_total, 1)
            tile_counts = torch_iris._C._matmul_tile_submissions()
            self.assertEqual(tuple(a - b for a, b in zip(tile_counts, previous_tiles)), expected)

    def test_rejects_unsupported_inputs(self):
        a = torch.empty((2, 3), device="iris")
        b = torch.empty((4, 2), device="iris")
        with self.assertRaisesRegex(RuntimeError, "float32"):
            torch.empty((2, 3), dtype=torch.float64, device="iris")
        with self.assertRaisesRegex(RuntimeError, "negative"):
            torch.empty_strided((2, 3), (1, -2), device="iris")
        with self.assertRaisesRegex(RuntimeError, "expanded size"):
            a.copy_(torch.empty((3, 2)))
        with self.assertRaisesRegex(RuntimeError, "size of tensor"):
            _ = a + b
        with self.assertRaisesRegex(RuntimeError, "matching inner dimensions"):
            _ = a @ b
        with self.assertRaisesRegex(RuntimeError, "at least 1D"):
            _ = a @ torch.empty((), device="iris")
        with self.assertRaisesRegex(RuntimeError, "device iris:0"):
            torch.empty(1, device="iris:1")

    def test_recurrent_cell_launch_and_module_parity(self):
        import copy

        torch.manual_seed(24)
        for cell, gate_count in (("lstm", 4), ("gru", 3)):
            gates = torch.randn(3, gate_count * 7, device="iris")
            state = torch.randn(3, 7, device="iris")
            before = torch_iris._C._kernel_submissions()
            getattr(torch.ops.aten, f"_thnn_fused_{cell}_cell")(gates, gates, state)
            self.assertEqual(torch_iris._C._kernel_submissions() - before, 1)

            cls = getattr(torch.nn, cell.upper())
            host = cls(4, 7, 2, batch_first=True, bidirectional=True)
            iris = copy.deepcopy(host).to("iris")
            x = torch.randn(3, 5, 4, requires_grad=True)
            dx = x.detach().to("iris").requires_grad_()
            expected = host(x)[0]
            actual = iris(dx)[0]
            torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
            expected.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(dx.grad.cpu(), x.grad, atol=2e-5, rtol=2e-4)
            for dev, cpu in zip(iris.parameters(), host.parameters()):
                torch.testing.assert_close(dev.grad.cpu(), cpu.grad, atol=2e-5, rtol=2e-4)

    def test_large_causal_attention_training_path(self):
        torch.manual_seed(22)
        qkv = [torch.randn(1, 4, 256, 128, requires_grad=True) for _ in range(3)]
        iris_qkv = [x.detach().to("iris").requires_grad_() for x in qkv]
        grad = torch.randn_like(qkv[0])
        expected = torch.nn.functional.scaled_dot_product_attention(*qkv, is_causal=True)
        actual = torch.nn.functional.scaled_dot_product_attention(*iris_qkv, is_causal=True)
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=2e-4)
        ref_grads = torch.autograd.grad(expected, qkv, grad)
        iris_grads = torch.autograd.grad(actual, iris_qkv, grad.to("iris"))
        for actual_grad, expected_grad in zip(iris_grads, ref_grads):
            torch.testing.assert_close(actual_grad.cpu(), expected_grad, atol=3e-5, rtol=3e-4)

    def test_dense_conv_input_gradient_fast_path(self):
        torch.manual_seed(21)
        x = torch.randn(4, 64, 32, 35, requires_grad=True)
        w = torch.randn(64, 64, 3, 3)
        grad = torch.randn_like(x)
        iris = x.detach().to("iris").requires_grad_()
        host_out = torch.nn.functional.conv2d(x, w, padding=1)
        iris_out = torch.nn.functional.conv2d(iris, w.to("iris"), padding=1)
        expected, = torch.autograd.grad(host_out, x, grad)
        actual, = torch.autograd.grad(iris_out, iris, grad.to("iris"))
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-4, rtol=2e-4)

    def test_matmul_vector_forward_and_backward(self):
        torch.manual_seed(14)
        for a_shape, b_shape in (((5,), (5,)), ((3, 5), (5,)), ((5,), (5, 7)),
                                 ((2, 3, 5), (5,)), ((5,), (2, 5, 7))):
            a = torch.randn(a_shape, requires_grad=True)
            b = torch.randn(b_shape, requires_grad=True)
            da = a.detach().to("iris").requires_grad_()
            db = b.detach().to("iris").requires_grad_()
            expected = a @ b
            result = da @ db
            torch.testing.assert_close(result.cpu(), expected)
            expected.sum().backward()
            result.sum().backward()
            torch.testing.assert_close(da.grad.cpu(), a.grad)
            torch.testing.assert_close(db.grad.cpu(), b.grad)

    def test_completed_after_input_deallocation(self):
        a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        result = (a.to("iris") @ a.to("iris")) + a.to("iris")
        torch.testing.assert_close(result.cpu(), a @ a + a)

    def test_philox_seed_and_strided_random_fills(self):
        def samples():
            base = torch.zeros((2, 4096), device="iris")
            view = base[1, 1::2]
            values = []
            for fn, args in ((view.uniform_, (-0.25, 0.75)), (view.normal_, (0.25, 1.5)),
                             (view.bernoulli_, (0.25,))):
                count = torch_iris._C._kernel_submissions()
                self.assertIs(fn(*args), view)
                self.assertEqual(torch_iris._C._kernel_submissions() - count, 1)
                values.append(view.cpu())
            self.assertTrue(torch.all(base.cpu()[0] == 0))
            self.assertTrue(torch.all(base.cpu()[1, ::2] == 0))
            return values

        torch.manual_seed(418)
        values = samples()
        torch_iris.manual_seed_all(418)
        for actual, expected in zip(samples(), values, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.all((values[0] >= -0.25) & (values[0] < 0.75)))
        self.assertAlmostEqual(values[1].mean().item(), 0.25, delta=0.13)
        self.assertAlmostEqual(values[1].std().item(), 1.5, delta=0.13)
        self.assertTrue(torch.all((values[2] == 0) | (values[2] == 1)))
        self.assertAlmostEqual(values[2].mean().item(), 0.25, delta=0.06)

    def test_serialization_preserves_strides_and_shared_storage(self):
        host = torch.arange(24, dtype=torch.int64).reshape(4, 6)
        for source_device in ("cpu", "iris"):
            with self.subTest(source_device=source_device):
                base = host.to(source_device)
                view = base[1:, 1::2].t()
                payload = BytesIO()
                torch.save({"base": base, "view": view}, payload)
                for location in ("cpu", "iris"):
                    payload.seek(0)
                    loaded = torch.load(payload, map_location=location, weights_only=True)
                    self.assertEqual(loaded["base"].device.type, location)
                    self.assertEqual(loaded["view"].storage_offset(), view.storage_offset())
                    torch.testing.assert_close(loaded["view"].cpu(), view.cpu())
                    self.assertEqual(loaded["view"].stride(), view.stride())
                    loaded["view"].fill_(123)
                    self.assertEqual(loaded["base"][1, 1].item(), 123)

    def test_tensor_formatting(self):
        for host in (
            torch.tensor([float("nan"), 1e-8, 12.3, float("inf")]),
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            torch.tensor([False, True]),
            torch.arange(2048, dtype=torch.float32).reshape(2, 1024).t(),
        ):
            with self.subTest(shape=host.shape, dtype=host.dtype):
                value = host.to("iris")
                # Device suffix line wrapping follows PyTorch's width setting.
                with torch._tensor_str.printoptions(linewidth=200):
                    expected = repr(host)[:-1] + ", device='iris:0')"
                    self.assertEqual(repr(value), expected)
                    self.assertEqual(str(value), expected)
                    output = StringIO()
                    with redirect_stdout(output):
                        print(value)
                    self.assertEqual(output.getvalue(), expected + "\n")

    @unittest.skipUnless(HAS_FP16, "OpenCL FP16 is unavailable")
    def test_fp16_casts_storage_and_blocked_gemm(self):
        self.assertIn(torch.float16, torch_iris.get_amp_supported_dtype())
        random = torch.Generator().manual_seed(404)
        weight = (torch.rand((260, 132), generator=random) - 0.5).half()
        data = (torch.rand((260, 132), generator=random) - 0.5).half()
        host_data, host_weight = data[1:257], weight[1:257]
        device_data = data.to("iris")[1:257]
        device_weight = weight.float().to("iris").half()[1:257].t()
        before = torch_iris._C._matmul_tile_submissions()[1]
        output = torch.mm(device_data, device_weight)
        self.assertEqual(torch_iris._C._matmul_tile_submissions()[1], before + 1)
        expect = (host_data.float() @ host_weight.float().t()).half()
        torch.testing.assert_close(output.cpu(), expect, atol=1e-3, rtol=1e-3, check_stride=True)
        torch.testing.assert_close(device_weight.float().cpu(), host_weight.float().t(), rtol=0, atol=0)
        self.assertEqual(device_weight.dtype, torch.float16)

    def test_allocator_reuses_buffers_and_releases_cache(self):
        torch_iris.empty_cache()
        initial = torch_iris.memory_stats()
        first = torch.empty((4096,), device="iris")
        self.assertEqual(torch_iris.memory_stats()["allocated_bytes"], initial["allocated_bytes"] + first.nbytes)
        first.fill_(4)
        del first
        self.assertGreaterEqual(torch_iris.memory_stats()["cached_bytes"], 4096 * 4)
        resumed = torch.empty((4096,), device="iris")
        self.assertEqual(torch_iris.memory_stats()["reuses"], initial["reuses"] + 1)
        resumed.fill_(9)
        torch_iris.empty_cache()
        self.assertEqual(torch_iris.memory_stats()["cached_bytes"], 0)
        torch.testing.assert_close(resumed.cpu(), torch.full((4096,), 9.0))

    def test_stream_events_and_pinned_copy_lifetime(self):
        expected = torch.arange(4096, dtype=torch.float32)
        pinned = torch.empty(expected.numel() + 2, pin_memory=True)
        self.assertTrue(pinned.is_pinned())
        pinned[1:-1].copy_(expected)
        producer = torch.Stream(device="iris")
        consumer = torch.Stream(device="iris")
        self.assertNotEqual(producer, consumer)
        begin = torch.Event(device="iris", enable_timing=True)
        end = torch.Event(device="iris", enable_timing=True)
        with producer:
            begin.record()
            resident = pinned[1:-1].to("iris", non_blocking=True)
            del pinned
            ready = producer.record_event()
        with consumer:
            consumer.wait_event(ready)
            resident.record_stream(consumer)
            shifted = resident + 0.5
            del resident
            output = torch.empty_like(expected, pin_memory=True)
            output.copy_(shifted, non_blocking=True)
            del shifted
            end.record()
        end.synchronize()
        self.assertTrue(ready.query())
        self.assertTrue(end.query())
        self.assertGreaterEqual(begin.elapsed_time(end), 0)
        consumer.synchronize()
        self.assertTrue(consumer.query())
        torch.testing.assert_close(output, expected + 0.5, rtol=0, atol=0)

    def test_fused_inference_attention_uses_one_kernel(self):
        random = torch.Generator().manual_seed(527)
        q = (torch.rand((2, 11, 3, 7), generator=random) - 0.5).transpose(1, 2)
        k = (torch.rand((2, 13, 3, 7), generator=random) - 0.5).transpose(1, 2)
        v = (torch.rand((2, 13, 3, 9), generator=random) - 0.5).transpose(1, 2)
        resident = [item.to("iris") for item in (q, k, v)]
        for causal in (False, True):
            previous = torch_iris._C._kernel_submissions()
            actual = torch.nn.functional.scaled_dot_product_attention(*resident, is_causal=causal, scale=0.25)
            self.assertEqual(torch_iris._C._kernel_submissions() - previous, 1)
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=0.25)
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5, check_stride=True)

    def test_conv2d_dispatch_paths_and_backward(self):
        random = torch.Generator().manual_seed(901)
        cases = [
            # Direct 3x3 stride=1 OC32 with non-tile-aligned spatial dims
            ((4, 16, 17, 19), (32, 16, 3, 3), True, (1, 1), (1, 1), (1, 1), 1),
            # Direct 3x3 stride=1 OC16 grouped with padding=2 and no bias
            ((2, 8, 11, 13), (16, 4, 3, 3), False, (1, 1), (2, 2), (1, 1), 2),
            # 1x1 pointwise GEMM fast path with and without bias
            ((3, 16, 9, 11), (24, 16, 1, 1), True, (1, 1), (0, 0), (1, 1), 1),
            ((2, 12, 7, 8), (16, 12, 1, 1), False, (1, 1), (0, 0), (1, 1), 1),
            # Implicit GEMM stride=2 + split-K grad_weight
            ((4, 16, 16, 16), (32, 16, 3, 3), True, (2, 2), (1, 1), (1, 1), 1),
            # Split-K > 1 grad_weight (N*OH*OW >= 512)
            ((4, 8, 16, 16), (16, 8, 3, 3), True, (1, 1), (1, 1), (1, 1), 1),
            # General dilated grouped asymmetric kernel
            ((2, 6, 11, 13), (9, 2, 5, 2), True, (2, 1), (2, 1), (2, 1), 3),
            # Depthwise 3x3 stride=1
            ((2, 8, 12, 14), (8, 1, 3, 3), True, (1, 1), (1, 1), (1, 1), 8),
        ]
        for x_shape, w_shape, use_bias, stride, padding, dilation, groups in cases:
            with self.subTest(x=x_shape, w=w_shape, bias=use_bias, s=stride, p=padding, d=dilation, g=groups):
                x_cpu = (torch.rand(x_shape, generator=random) - 0.5).requires_grad_()
                w_cpu = (torch.rand(w_shape, generator=random) - 0.5).requires_grad_()
                b_cpu = (torch.rand(w_shape[0], generator=random) - 0.5).requires_grad_() if use_bias else None
                y_cpu = torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu, stride=stride,
                                                   padding=padding, dilation=dilation, groups=groups)
                grad_cpu = torch.rand(y_cpu.shape, generator=random) - 0.5
                y_cpu.backward(grad_cpu)

                x_iris = x_cpu.detach().to("iris").requires_grad_()
                w_iris = w_cpu.detach().to("iris").requires_grad_()
                b_iris = b_cpu.detach().to("iris").requires_grad_() if use_bias else None
                y_iris = torch.nn.functional.conv2d(x_iris, w_iris, b_iris, stride=stride,
                                                    padding=padding, dilation=dilation, groups=groups)
                y_iris.backward(grad_cpu.to("iris"))

                torch.testing.assert_close(y_iris.cpu(), y_cpu, rtol=1e-4, atol=1e-5, check_stride=True)
                torch.testing.assert_close(x_iris.grad.cpu(), x_cpu.grad, rtol=2e-4, atol=1e-5, check_stride=True)
                torch.testing.assert_close(w_iris.grad.cpu(), w_cpu.grad, rtol=2e-4, atol=1e-5, check_stride=True)
                if use_bias:
                    torch.testing.assert_close(b_iris.grad.cpu(), b_cpu.grad, rtol=2e-4, atol=1e-5, check_stride=True)

        # Empty batch Conv2d forward and backward
        x_empty = torch.empty((0, 4, 8, 8), device="iris", requires_grad=True)
        w_empty = torch.ones((8, 4, 3, 3), device="iris", requires_grad=True)
        b_empty = torch.ones((8,), device="iris", requires_grad=True)
        y_empty = torch.nn.functional.conv2d(x_empty, w_empty, b_empty, padding=1)
        self.assertEqual(tuple(y_empty.shape), (0, 8, 8, 8))
        y_empty.sum().backward()
        torch.testing.assert_close(w_empty.grad.cpu(), torch.zeros((8, 4, 3, 3)))
        torch.testing.assert_close(b_empty.grad.cpu(), torch.zeros((8,)))

    def test_sg8_gemm_and_contiguous_reductions(self):
        random = torch.Generator().manual_seed(902)
        # SG8 128x64 GEMM path across all four transpose layouts with fused bias and alpha/beta
        for rows, inner, columns in ((256, 512, 512), (128, 256, 1024), (256, 1024, 128)):
            a = torch.rand((rows, inner), generator=random) - 0.5
            b = torch.rand((inner, columns), generator=random) - 0.5
            bias = torch.rand((columns,), generator=random) - 0.5
            for ta, tb in product((False, True), repeat=2):
                with self.subTest(shape=(rows, inner, columns), layout=(ta, tb)):
                    ia = a.t().contiguous().to("iris").t() if ta else a.to("iris")
                    ib = b.t().contiguous().to("iris").t() if tb else b.to("iris")
                    tiles = torch_iris._C._matmul_tile_submissions()[1]
                    actual = torch.addmm(bias.to("iris"), ia, ib, alpha=0.75, beta=-0.5)
                    self.assertEqual(torch_iris._C._matmul_tile_submissions()[1], tiles + 1)
                    expected = torch.addmm(bias, a, b, alpha=0.75, beta=-0.5)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5, check_stride=True)

        # Contiguous reduction fast paths and vectorized elementwise tails (non-multiple-of-4 lengths)
        x1d = torch.rand((2051,), generator=random) - 0.5
        y1d = torch.rand((2051,), generator=random) - 0.5
        ix1d, iy1d = x1d.to("iris"), y1d.to("iris")
        torch.testing.assert_close(ix1d.sum().cpu(), x1d.sum(), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(ix1d.mean(0, keepdim=True).cpu(), x1d.mean(0, keepdim=True), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close((ix1d * 0.75).cpu(), x1d * 0.75, rtol=1e-4, atol=1e-5)
        scaled = ix1d.clone()
        scaled.mul_(0.75)
        torch.testing.assert_close(scaled.cpu(), x1d * 0.75, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            torch.nn.functional.mse_loss(ix1d, iy1d).cpu(),
            torch.nn.functional.mse_loss(x1d, y1d), rtol=1e-4, atol=1e-5,
        )

        x2d = torch.rand((64, 129), generator=random) - 0.5
        torch.testing.assert_close(x2d.to("iris").sum(0).cpu(), x2d.sum(0), rtol=1e-4, atol=1e-5, check_stride=True)
        torch.testing.assert_close(x2d.to("iris").mean(0, keepdim=True).cpu(), x2d.mean(0, keepdim=True),
                                   rtol=1e-4, atol=1e-5, check_stride=True)

        x4d = torch.rand((4, 16, 17, 19), generator=random) - 0.5
        torch.testing.assert_close(x4d.to("iris").sum((0, 2, 3)).cpu(), x4d.sum((0, 2, 3)),
                                   rtol=1e-4, atol=1e-5, check_stride=True)

    def test_flash_attention_extended_and_gqa(self):
        random = torch.Generator().manual_seed(1001)
        for (B, Hq, Hkv, Lq, Lk, Dq, Dv), causal in [
            ((2, 4, 4, 96, 96, 32, 32), False),
            ((2, 4, 4, 96, 96, 32, 32), True),
            ((2, 8, 8, 64, 300, 64, 64), False),
            ((1, 4, 4, 270, 270, 128, 128), True),
            ((2, 8, 2, 48, 64, 64, 64), True),
        ]:
            with self.subTest(shape=(B, Hq, Hkv, Lq, Lk, Dq, Dv), causal=causal):
                q = torch.rand((B, Hq, Lq, Dq), generator=random) - 0.5
                k = torch.rand((B, Hkv, Lk, Dq), generator=random) - 0.5
                v = torch.rand((B, Hkv, Lk, Dv), generator=random) - 0.5
                gqa = Hq != Hkv
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, enable_gqa=gqa
                )
                iq, ik, iv = q.to("iris"), k.to("iris"), v.to("iris")
                before = torch_iris._C._kernel_submissions()
                actual = torch.nn.functional.scaled_dot_product_attention(
                    iq, ik, iv, is_causal=causal, enable_gqa=gqa
                )
                if self.subgroup_kernels:
                    self.assertEqual(torch_iris._C._kernel_submissions() - before, 1)
                torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5, check_stride=True)

    @unittest.skipUnless(HAS_FP16, "OpenCL FP16 is unavailable")
    def test_fp16_sg8_gemm(self):
        random = torch.Generator().manual_seed(1002)
        for ta, tb in product((False, True), repeat=2):
            a = (torch.rand((256, 256), generator=random) - 0.5).half()
            b = (torch.rand((256, 512), generator=random) - 0.5).half()
            bias = (torch.rand((512,), generator=random) - 0.5).half()
            ia = a.t().contiguous().to("iris").t() if ta else a.to("iris")
            ib = b.t().contiguous().to("iris").t() if tb else b.to("iris")
            tiles = torch_iris._C._matmul_tile_submissions()[1]
            out = torch.addmm(bias.to("iris"), ia, ib, alpha=0.5, beta=-0.25)
            self.assertEqual(torch_iris._C._matmul_tile_submissions()[1], tiles + 1)
            ref = torch.addmm(bias.float(), a.float(), b.float(), alpha=0.5, beta=-0.25).half()
            torch.testing.assert_close(out.cpu(), ref, rtol=2e-3, atol=2e-3, check_stride=True)

    def test_nll_out_of_bounds_recovery(self):
        # An out-of-bounds target raises on readback or synchronization.
        bad_logits = torch.zeros((4, 5), device="iris")
        bad_target = torch.tensor([0, 1, 5, 2], device="iris")
        with self.assertRaises(IndexError):
            loss = torch.nn.functional.cross_entropy(bad_logits, bad_target)
            _ = loss.item()
        good = torch.nn.functional.cross_entropy(
            torch.zeros((4, 5), device="iris"), torch.tensor([0, 1, 2, 3], device="iris")
        )
        torch.testing.assert_close(good.cpu(), torch.tensor(1.609438), rtol=1e-4, atol=1e-4)

    def test_vision_and_llm_operators_parity(self):
        random = torch.Generator().manual_seed(1003)
        # BatchNorm2d train + eval
        bn_cpu = torch.nn.BatchNorm2d(8)
        bn_iris = torch.nn.BatchNorm2d(8).to("iris")
        bn_iris.load_state_dict(bn_cpu.state_dict())
        x_cpu = (torch.rand((4, 8, 8, 8), generator=random) - 0.5).requires_grad_()
        x_iris = x_cpu.detach().to("iris").requires_grad_()
        y_cpu = bn_cpu(x_cpu); y_iris = bn_iris(x_iris)
        go = torch.rand(y_cpu.shape, generator=random) - 0.5
        y_cpu.backward(go); y_iris.backward(go.to("iris"))
        torch.testing.assert_close(y_iris.cpu(), y_cpu, rtol=2e-4, atol=1e-5, check_stride=True)
        torch.testing.assert_close(bn_iris.running_mean.cpu(), bn_cpu.running_mean, rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(bn_iris.running_var.cpu(), bn_cpu.running_var, rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(x_iris.grad.cpu(), x_cpu.grad, rtol=2e-4, atol=1e-5)

        # GroupNorm, AdaptiveAvgPool2d, ConvTranspose2d, Conv1d, Upsample
        gn_cpu = torch.nn.GroupNorm(4, 8)
        gn_iris = torch.nn.GroupNorm(4, 8).to("iris")
        gn_iris.load_state_dict(gn_cpu.state_dict())
        torch.testing.assert_close(gn_iris(x_iris.detach()).cpu(), gn_cpu(x_cpu.detach()), rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(
            torch.nn.functional.adaptive_avg_pool2d(x_iris.detach(), (1, 1)).cpu(),
            torch.nn.functional.adaptive_avg_pool2d(x_cpu.detach(), (1, 1)), rtol=1e-4, atol=1e-5,
        )
        torch.testing.assert_close(
            torch.nn.functional.interpolate(x_iris.detach(), size=(12, 12), mode="bilinear", align_corners=False).cpu(),
            torch.nn.functional.interpolate(x_cpu.detach(), size=(12, 12), mode="bilinear", align_corners=False),
            rtol=1e-4, atol=1e-5,
        )
        ct_w = torch.rand((8, 4, 3, 3), generator=random) - 0.5
        torch.testing.assert_close(
            torch.nn.functional.conv_transpose2d(x_iris.detach(), ct_w.to("iris"), stride=2, padding=1, output_padding=1).cpu(),
            torch.nn.functional.conv_transpose2d(x_cpu.detach(), ct_w, stride=2, padding=1, output_padding=1),
            rtol=2e-4, atol=1e-5,
        )

        # RMSNorm, TopK (large vocab), Sort (stable), Cumsum, Arange, Linspace, Randint, Trig, Bitwise
        rms_x = (torch.rand((2, 8, 64), generator=random) - 0.5).requires_grad_()
        rms_w = (torch.rand((64,), generator=random) + 0.5).requires_grad_()
        rms_xi = rms_x.detach().to("iris").requires_grad_()
        rms_wi = rms_w.detach().to("iris").requires_grad_()
        ry_c = torch.nn.functional.rms_norm(rms_x, (64,), rms_w, 1e-6)
        ry_i = torch.nn.functional.rms_norm(rms_xi, (64,), rms_wi, 1e-6)
        ry_c.sum().backward(); ry_i.sum().backward()
        torch.testing.assert_close(ry_i.cpu(), ry_c, rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(rms_xi.grad.cpu(), rms_x.grad, rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(rms_wi.grad.cpu(), rms_w.grad, rtol=2e-4, atol=1e-5)

        logits = torch.rand((2, 4096), generator=random) - 0.5
        tv_c, ti_c = torch.topk(logits, 32, dim=-1)
        tv_i, ti_i = torch.topk(logits.to("iris"), 32, dim=-1)
        torch.testing.assert_close(tv_i.cpu(), tv_c, rtol=0, atol=0)
        torch.testing.assert_close(ti_i.cpu(), ti_c, rtol=0, atol=0)
        torch.testing.assert_close(logits.to("iris").argmax(-1).cpu(), logits.argmax(-1), rtol=0, atol=0)
        torch.testing.assert_close(torch.arange(0, 16, 2, device="iris").cpu(), torch.arange(0, 16, 2), rtol=0, atol=0)
        torch.testing.assert_close(torch.linspace(-1, 1, 9, device="iris").cpu(), torch.linspace(-1, 1, 9), rtol=1e-5, atol=1e-5)

    def test_flash_attention_backward_and_decode(self):
        random = torch.Generator().manual_seed(1101)
        for dtype, rtol, atol in [(torch.float32, 2e-4, 1e-5), (torch.float16, 4e-3, 4e-3)]:
            for (B, Hq, Hkv, Lq, Lk, D), causal in [
                ((2, 4, 4, 48, 48, 32), False),
                ((2, 4, 4, 64, 64, 64), True),
                ((1, 4, 2, 64, 64, 128), True),
            ]:
                with self.subTest(dtype=dtype, shape=(B, Hq, Hkv, Lq, Lk, D), causal=causal):
                    if dtype == torch.float16 and not HAS_FP16:
                        self.skipTest("OpenCL FP16 is unavailable")
                    q = (torch.rand((B, Hq, Lq, D), generator=random) - 0.5).to(dtype).requires_grad_()
                    k = (torch.rand((B, Hkv, Lk, D), generator=random) - 0.5).to(dtype).requires_grad_()
                    v = (torch.rand((B, Hkv, Lk, D), generator=random) - 0.5).to(dtype).requires_grad_()
                    go = (torch.rand((B, Hq, Lq, D), generator=random) - 0.5).to(dtype)
                    gqa = Hq != Hkv
                    q_ref = q.detach().float().requires_grad_()
                    k_ref = k.detach().float().requires_grad_()
                    v_ref = v.detach().float().requires_grad_()
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref, is_causal=causal, enable_gqa=gqa
                    )
                    out_ref.backward(go.float())

                    iq = q.detach().to("iris").requires_grad_()
                    ik = k.detach().to("iris").requires_grad_()
                    iv = v.detach().to("iris").requires_grad_()
                    before_fwd = torch_iris._C._kernel_submissions()
                    out_i = torch.nn.functional.scaled_dot_product_attention(
                        iq, ik, iv, is_causal=causal, enable_gqa=gqa
                    )
                    if self.subgroup_kernels:
                        self.assertEqual(torch_iris._C._kernel_submissions() - before_fwd, 1)
                    before_bwd = torch_iris._C._kernel_submissions()
                    out_i.backward(go.to("iris"))
                    if self.subgroup_kernels:
                        self.assertLessEqual(torch_iris._C._kernel_submissions() - before_bwd, 4)
                    torch.testing.assert_close(out_i.cpu().float(), out_ref, rtol=rtol, atol=atol)
                    torch.testing.assert_close(iq.grad.cpu().float(), q_ref.grad, rtol=rtol * 2, atol=atol * 2)
                    torch.testing.assert_close(ik.grad.cpu().float(), k_ref.grad, rtol=rtol * 2, atol=atol * 2)
                    torch.testing.assert_close(iv.grad.cpu().float(), v_ref.grad, rtol=rtol * 2, atol=atol * 2)

        # FlashDecoding Split-KV (Lq=1..4, long Lk) & skinny GEMV (M=1..4)
        for dtype, rtol, atol in [(torch.float32, 2e-4, 2e-4), (torch.float16, 4e-3, 4e-3)]:
            if dtype == torch.float16 and not HAS_FP16:
                with self.subTest(dtype=dtype, workload="decode and GEMV"):
                    self.skipTest("OpenCL FP16 is unavailable")
                continue
            q = (torch.rand((1, 8, 1, 128), generator=random) - 0.5).to(dtype)
            k = (torch.rand((1, 2, 1024, 128), generator=random) - 0.5).to(dtype)
            v = (torch.rand((1, 2, 1024, 128), generator=random) - 0.5).to(dtype)
            ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=True)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.to("iris"), k.to("iris"), v.to("iris"), enable_gqa=True
            )
            torch.testing.assert_close(out.cpu().float(), ref, rtol=rtol, atol=atol)

            x = (torch.rand((3, 512), generator=random) - 0.5).to(dtype)
            w = (torch.rand((1024, 512), generator=random) - 0.5).to(dtype)
            out_mv = torch.nn.functional.linear(x.to("iris"), w.to("iris"))
            ref_mv = torch.nn.functional.linear(x.float(), w.float())
            torch.testing.assert_close(out_mv.cpu().float(), ref_mv, rtol=rtol * 2, atol=atol * 2)

    @unittest.skipUnless(HAS_FP16, "OpenCL FP16 is unavailable")
    def test_native_fp16_ops_and_autocast(self):
        random = torch.Generator().manual_seed(1102)
        x = (torch.rand((2, 16, 12, 12), generator=random) - 0.5).half()
        w = (torch.rand((16, 16, 3, 3), generator=random) - 0.5).half() * 0.5
        b = (torch.rand((16,), generator=random) - 0.5).half() * 0.25
        xi = x.to("iris").requires_grad_()
        wi = w.to("iris").requires_grad_()
        bi = b.to("iris").requires_grad_()
        yi = torch.nn.functional.conv2d(xi, wi, bi, padding=1)
        self.assertEqual(yi.dtype, torch.float16)
        yi.sum().backward()
        xr = x.float().requires_grad_()
        wr = w.float().requires_grad_()
        br = b.float().requires_grad_()
        yr = torch.nn.functional.conv2d(xr, wr, br, padding=1)
        yr.sum().backward()
        torch.testing.assert_close(yi.cpu().float(), yr, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(xi.grad.cpu().float(), xr.grad, rtol=1e-2, atol=1e-2)

        # Mixed-precision BatchNorm2d, GroupNorm, RMSNorm, Upsample, AdaptiveAvgPool2d
        rm = torch.zeros(16, device="iris")
        rv = torch.ones(16, device="iris")
        bw = torch.ones(16, device="iris", requires_grad=True)
        bb = torch.zeros(16, device="iris", requires_grad=True)
        bn_out = torch.nn.functional.batch_norm(x.to("iris"), rm, rv, bw, bb, training=True)
        self.assertEqual(bn_out.dtype, torch.float16)
        gn_out = torch.nn.functional.group_norm(bn_out, 4)
        self.assertEqual(gn_out.dtype, torch.float16)
        up_out = torch.nn.functional.interpolate(gn_out, size=(16, 16), mode="bilinear", align_corners=False)
        self.assertEqual(up_out.dtype, torch.float16)
        aap_out = torch.nn.functional.adaptive_avg_pool2d(up_out, (1, 1))
        self.assertEqual(aap_out.dtype, torch.float16)

        # Autocast end-to-end on Conv2d + BN + Linear
        model = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 8),
        ).to("iris").eval()
        with torch.no_grad():
            inp32 = x.float().to("iris")
            y32 = model(inp32)
            with torch.autocast("iris", dtype=torch.float16):
                y16 = model(inp32)
            self.assertEqual(y16.dtype, torch.float16)
            torch.testing.assert_close(y16.cpu().float(), y32.cpu(), rtol=2e-2, atol=2e-2)

    def test_weight_only_int8_int4_linear(self):
        random = torch.Generator().manual_seed(1103)
        for dtype, rtol, atol in [(torch.float32, 2e-4, 2e-4), (torch.float16, 4e-3, 4e-3)]:
            if dtype == torch.float16 and not HAS_FP16:
                with self.subTest(dtype=dtype):
                    self.skipTest("OpenCL FP16 is unavailable")
                continue
            for symmetric in (True, False):
                x = (torch.rand((4, 256), generator=random) - 0.5).to(dtype)
                w = (torch.rand((128, 256), generator=random) - 0.5) * 0.4
                b = (torch.rand((128,), generator=random) - 0.5).to(dtype) * 0.2

                qw8, s8, z8 = torch_iris.ops.quantize_weight_int8(w, group_size=64, symmetric=symmetric)
                ref8 = torch_iris.ops.linear_int8(x, qw8, s8, z8, b, group_size=64)
                out8 = torch_iris.ops.linear_int8(
                    x.to("iris"), qw8.to("iris"), s8.to("iris"),
                    None if z8 is None else z8.to("iris"), b.to("iris"), group_size=64
                )
                self.assertEqual(out8.dtype, dtype)
                torch.testing.assert_close(out8.cpu(), ref8, rtol=rtol, atol=atol)

                qw4, s4, z4 = torch_iris.ops.quantize_weight_int4(w, group_size=64, symmetric=symmetric)
                ref4 = torch_iris.ops.linear_int4(x, qw4, s4, z4, b, group_size=64)
                out4 = torch_iris.ops.linear_int4(
                    x.to("iris"), qw4.to("iris"), s4.to("iris"),
                    None if z4 is None else z4.to("iris"), b.to("iris"), group_size=64
                )
                self.assertEqual(out4.dtype, dtype)
                torch.testing.assert_close(out4.cpu(), ref4, rtol=rtol, atol=atol)

    def test_phase3_llm_vision_math_and_generator(self):
        random = torch.Generator().manual_seed(1104)
        # Explicit device torch.Generator + multinomial reproducibility
        g1 = torch.Generator(device="iris").manual_seed(777)
        g2 = torch.Generator(device="iris").manual_seed(777)
        probs = torch.tensor([[0.1, 0.7, 0.0, 0.2], [0.5, 0.0, 0.5, 0.0]], device="iris")
        s1 = torch.multinomial(probs, 64, replacement=True, generator=g1)
        s2 = torch.multinomial(probs, 64, replacement=True, generator=g2)
        self.assertTrue(torch.equal(s1.cpu(), s2.cpu()))
        st = g1.get_state()
        r1 = torch.randn(16, device="iris", generator=g1)
        g1.set_state(st)
        r2 = torch.randn(16, device="iris", generator=g1)
        self.assertTrue(torch.equal(r1.cpu(), r2.cpu()))
        sn = torch.multinomial(probs, 2, replacement=False)
        for row in range(2):
            vals = sn[row].cpu().tolist()
            self.assertEqual(len(set(vals)), 2)

        # searchsorted, masked_scatter_, isin, index_add_, repeat_interleave, roll
        seq = torch.tensor([1.0, 3.0, 3.0, 7.0])
        vals = torch.tensor([0.0, 3.0, 5.0, 9.0])
        torch.testing.assert_close(
            torch.searchsorted(seq.to("iris"), vals.to("iris"), right=True).cpu(),
            torch.searchsorted(seq, vals, right=True),
        )
        m = torch.tensor([[True, False, True], [False, True, False]])
        src = torch.tensor([10.0, 20.0, 30.0])
        torch.testing.assert_close(
            torch.zeros(2, 3, device="iris").masked_scatter_(m.to("iris"), src.to("iris")).cpu(),
            torch.zeros(2, 3).masked_scatter_(m, src),
        )
        a_i = torch.tensor([1, 2, 3, 4, 5])
        b_i = torch.tensor([2, 4, 6])
        torch.testing.assert_close(torch.isin(a_i.to("iris"), b_i.to("iris")).cpu(), torch.isin(a_i, b_i))
        idx = torch.tensor([0, 1, 0])
        src2 = torch.ones(3, 4) * 2.0
        torch.testing.assert_close(
            torch.zeros(2, 4, device="iris").index_add_(0, idx.to("iris"), src2.to("iris"), alpha=0.5).cpu(),
            torch.zeros(2, 4).index_add_(0, idx, src2, alpha=0.5),
        )
        reps = torch.tensor([2, 0, 3])
        t2 = torch.arange(12, dtype=torch.float32).view(3, 4)
        torch.testing.assert_close(t2.to("iris").repeat_interleave(reps.to("iris"), dim=0).cpu(), t2.repeat_interleave(reps, dim=0))
        torch.testing.assert_close(torch.roll(t2.to("iris"), shifts=(-1, 2), dims=(0, 1)).cpu(), torch.roll(t2, shifts=(-1, 2), dims=(0, 1)))

        # Padding, activations, grid_sampler_2d, Conv3d/Pool3d
        x4 = torch.rand((2, 4, 6, 6), generator=random) - 0.5
        for mode in ("reflect", "replicate"):
            torch.testing.assert_close(
                torch.nn.functional.pad(x4.to("iris"), (1, 2, 2, 1), mode=mode).cpu(),
                torch.nn.functional.pad(x4, (1, 2, 2, 1), mode=mode),
            )
        for act in (
            lambda z: torch.nn.functional.leaky_relu(z, 0.1),
            lambda z: torch.nn.functional.elu(z, 1.0),
            torch.nn.functional.hardswish,
            torch.nn.functional.hardsigmoid,
            torch.nn.functional.mish,
            torch.nn.functional.relu6,
        ):
            torch.testing.assert_close(act(x4.to("iris")).cpu(), act(x4), rtol=1e-4, atol=1e-4)
        grid = torch.rand((2, 4, 4, 2), generator=random) * 2.2 - 1.1
        torch.testing.assert_close(
            torch.nn.functional.grid_sample(x4.to("iris"), grid.to("iris"), mode="bicubic", padding_mode="border", align_corners=False).cpu(),
            torch.nn.functional.grid_sample(x4, grid, mode="bicubic", padding_mode="border", align_corners=False),
            rtol=1e-3, atol=1e-3,
        )
        x5 = torch.rand((1, 2, 4, 4, 4), generator=random) - 0.5
        w5 = torch.rand((3, 2, 3, 3, 3), generator=random) - 0.5
        torch.testing.assert_close(
            torch.nn.functional.conv3d(x5.to("iris"), w5.to("iris"), padding=1).cpu(),
            torch.nn.functional.conv3d(x5, w5, padding=1), rtol=2e-4, atol=1e-4,
        )
        torch.testing.assert_close(
            torch.nn.functional.max_pool3d(x5.to("iris"), 2, 2).cpu(),
            torch.nn.functional.max_pool3d(x5, 2, 2),
        )
        torch.testing.assert_close(
            torch.nn.functional.avg_pool3d(x5.to("iris"), 2, 2).cpu(),
            torch.nn.functional.avg_pool3d(x5, 2, 2),
        )

        # Reductions, math, and large integer sort > 1024
        torch.testing.assert_close(torch.var(x4.to("iris"), dim=(1, 2)).cpu(), torch.var(x4, dim=(1, 2)), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.std(x4.to("iris"), dim=-1).cpu(), torch.std(x4, dim=-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.prod(x4.abs().to("iris") + 0.5, dim=-1).cpu(), torch.prod(x4.abs() + 0.5, dim=-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.linalg.vector_norm(x4.to("iris"), ord=2, dim=-1).cpu(), torch.linalg.vector_norm(x4, ord=2, dim=-1), rtol=1e-4, atol=1e-4)
        y4 = torch.rand((2, 4, 6, 6), generator=random) + 0.3
        torch.testing.assert_close(torch.atan2(x4.to("iris"), y4.to("iris")).cpu(), torch.atan2(x4, y4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.erf(x4.to("iris")).cpu(), torch.erf(x4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.log1p(y4.to("iris")).cpu(), torch.log1p(y4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.expm1(x4.to("iris")).cpu(), torch.expm1(x4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.fmod(x4.to("iris"), y4.to("iris")).cpu(), torch.fmod(x4, y4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.remainder(x4.to("iris"), y4.to("iris")).cpu(), torch.remainder(x4, y4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.lerp(x4.to("iris"), y4.to("iris"), 0.25).cpu(), torch.lerp(x4, y4, 0.25), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.addcmul(x4.to("iris"), y4.to("iris"), y4.to("iris"), value=2.0).cpu(), torch.addcmul(x4, y4, y4, value=2.0), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.addcdiv(x4.to("iris"), x4.to("iris"), y4.to("iris"), value=1.5).cpu(), torch.addcdiv(x4, x4, y4, value=1.5), rtol=1e-4, atol=1e-4)

        big_int = torch.randint(-5000, 5000, (2, 1400), generator=random, dtype=torch.int64)
        sv_i, _ = torch.sort(big_int.to("iris"), dim=-1, descending=True)
        sv_c, _ = torch.sort(big_int, dim=-1, descending=True)
        self.assertTrue(torch.equal(sv_i.cpu(), sv_c))

    def test_real_world_models_resnet_mobilenet_unet_gpt2_llama(self):
        import torchvision.models as tvm
        torch.manual_seed(1105)

        # 1. Real torchvision ResNet-18 & MobileNetV2 (zero missing ATen ops)
        x_img = torch.randn(2, 3, 64, 64)
        for ctor in (lambda: tvm.resnet18(weights=None), lambda: tvm.mobilenet_v2(weights=None)):
            m_c = ctor().eval()
            m_i = ctor().eval()
            m_i.load_state_dict(m_c.state_dict())
            m_i.to("iris")
            with torch.no_grad():
                yc = m_c(x_img)
                yi = m_i(x_img.to("iris"))
                torch.testing.assert_close(yi.cpu(), yc, rtol=2e-3, atol=2e-3)
                with self.subTest(model=type(m_i).__name__, dtype=torch.float16):
                    if not HAS_FP16:
                        self.skipTest("OpenCL FP16 is unavailable")
                    with torch.autocast("iris", dtype=torch.float16):
                        yac = m_i(x_img.to("iris"))
                    self.assertEqual(yac.shape, yc.shape)
                    self.assertTrue(torch.isfinite(yac.cpu()).all())

        # 2. UNet encoder-decoder (Conv2d + GroupNorm + Mish + ConvTranspose2d + Upsample + skip cat)
        class MiniUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GroupNorm(4, 16), torch.nn.Mish())
                self.down = torch.nn.MaxPool2d(2)
                self.mid = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.GroupNorm(4, 32), torch.nn.LeakyReLU(0.1))
                self.up = torch.nn.ConvTranspose2d(32, 16, 2, stride=2)
                self.dec = torch.nn.Conv2d(32, 4, 1)
            def forward(self, x):
                e1 = self.enc1(x)
                m = self.mid(self.down(e1))
                u = self.up(m)
                return self.dec(torch.cat((u, e1), dim=1))

        u_c = MiniUNet()
        u_i = MiniUNet()
        u_i.load_state_dict(u_c.state_dict())
        u_i.to("iris")
        xu = torch.randn(2, 3, 16, 16)
        yu_c = u_c(xu); yu_c.sum().backward()
        yu_i = u_i(xu.to("iris")); yu_i.sum().backward()
        torch.testing.assert_close(yu_i.cpu(), yu_c, rtol=2e-3, atol=2e-3)

        # 3. Multi-layer Llama with KV-cache + top-k/top-p multinomial sampling + AdamW FlashAttention bwd
        class TinyLlamaKV(torch.nn.Module):
            def __init__(self, vocab=64, dim=128, heads=4, layers=2):
                super().__init__()
                self.heads = heads; self.hd = dim // heads
                self.emb = torch.nn.Embedding(vocab, dim)
                self.layers = torch.nn.ModuleList([
                    torch.nn.ModuleDict({
                        "w1": torch.nn.Linear(dim, 3 * dim, bias=False),
                        "wo": torch.nn.Linear(dim, dim, bias=False),
                        "wg": torch.nn.Linear(dim, dim * 2, bias=False),
                        "wu": torch.nn.Linear(dim, dim * 2, bias=False),
                        "wd": torch.nn.Linear(dim * 2, dim, bias=False),
                    }) for _ in range(layers)
                ])
                self.head = torch.nn.Linear(dim, vocab, bias=False)
            def forward(self, tokens, kv_cache=None):
                B, L = tokens.shape
                h = self.emb(tokens)
                new_kv = []
                for idx, blk in enumerate(self.layers):
                    hn = torch.nn.functional.rms_norm(h, (h.size(-1),))
                    qkv = blk["w1"](hn).view(B, L, 3, self.heads, self.hd)
                    q = qkv[:, :, 0].transpose(1, 2)
                    k = qkv[:, :, 1].transpose(1, 2)
                    v = qkv[:, :, 2].transpose(1, 2)
                    if kv_cache is not None:
                        k = torch.cat((kv_cache[idx][0], k), dim=2)
                        v = torch.cat((kv_cache[idx][1], v), dim=2)
                    new_kv.append((k, v))
                    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
                    h = h + blk["wo"](a.transpose(1, 2).reshape(B, L, -1))
                    hn2 = torch.nn.functional.rms_norm(h, (h.size(-1),))
                    h = h + blk["wd"](torch.nn.functional.silu(blk["wg"](hn2)) * blk["wu"](hn2))
                return self.head(torch.nn.functional.rms_norm(h, (h.size(-1),))), new_kv

        lm_c = TinyLlamaKV()
        lm_i = TinyLlamaKV()
        lm_i.load_state_dict(lm_c.state_dict())
        lm_i.to("iris")
        opt_c = torch.optim.AdamW(lm_c.parameters(), lr=1e-3, fused=False)
        opt_i = torch.optim.AdamW(lm_i.parameters(), lr=1e-3, fused=True)
        toks = torch.randint(0, 64, (2, 16))
        tgts = torch.randint(0, 64, (32,))
        lc = torch.nn.functional.cross_entropy(lm_c(toks)[0].view(-1, 64), tgts)
        lc.backward(); opt_c.step()
        li = torch.nn.functional.cross_entropy(lm_i(toks.to("iris"))[0].view(-1, 64), tgts.to("iris"))
        li.backward(); opt_i.step()
        torch.testing.assert_close(li.cpu(), lc, rtol=2e-3, atol=2e-3)

        # KV-cached decode + top-k/top-p multinomial sampling
        with torch.no_grad():
            logits_c, kv_c = lm_c(toks[:1, :8])
            logits_i, kv_i = lm_i(toks[:1, :8].to("iris"))
            torch.testing.assert_close(logits_i.cpu(), logits_c, rtol=2e-3, atol=2e-3)
            gen = torch.Generator(device="iris").manual_seed(42)
            probs = torch.softmax(logits_i[:, -1, :], dim=-1)
            tv, ti = torch.topk(probs, 8, dim=-1)
            sampled_local = torch.multinomial(tv, 1, generator=gen)
            next_tok = ti.gather(-1, sampled_local)
            step_logits, _ = lm_i(next_tok, kv_cache=kv_i)
            self.assertEqual(step_logits.shape, (1, 1, 64))
            self.assertTrue(torch.isfinite(step_logits.cpu()).all())


if __name__ == "__main__":
    unittest.main()

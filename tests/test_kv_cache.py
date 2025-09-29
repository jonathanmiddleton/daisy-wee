import unittest
import torch

from inference.kv_cache import KVCache


class TestKVCache(unittest.TestCase):
    def test_empty_view_returns_zero_length(self):
        # L layers, B batch, H heads, W window, D dim
        L, B, H, W, D = 2, 2, 2, 3, 2
        cache = KVCache(L=L, B=B, H=H, W=W, D=D, device="cpu", dtype=torch.float32)

        # No steps advanced; view should be empty along sequence dimension
        k, v = cache.view(layer=1)
        self.assertEqual(k.shape, (B, 0, H, D))
        self.assertEqual(v.shape, (B, 0, H, D))

    def test_write_and_view_without_wrap(self):
        L, B, H, W, D = 1, 2, 2, 4, 2
        cache = KVCache(L=L, B=B, H=H, W=W, D=D, device="cpu", dtype=torch.float32)

        # Perform fewer writes than window size
        values = [10.0, 11.0, 12.0]  # n=3 < W
        for val in values:
            k_new = torch.full((B, H, 1, D), val)
            v_new = torch.full((B, H, 1, D), val + 1000.0)
            cache.write(layer=0, k_new=k_new, v_new=v_new)
            cache.advance()

        k, v = cache.view(layer=0)
        # Expected shapes: (B, n, H, D)
        self.assertEqual(k.shape, (B, len(values), H, D))
        self.assertEqual(v.shape, (B, len(values), H, D))

        # Build expected tensors in chronological order
        expected_k = torch.cat([torch.full((B, 1, H, D), val) for val in values], dim=1)
        expected_v = torch.cat([torch.full((B, 1, H, D), val + 1000.0) for val in values], dim=1)

        torch.testing.assert_close(k, expected_k)
        torch.testing.assert_close(v, expected_v)

    def test_write_and_view_with_wrap(self):
        L, B, H, W, D = 1, 2, 2, 3, 2
        cache = KVCache(L=L, B=B, H=H, W=W, D=D, device="cpu", dtype=torch.float32)

        # Perform more writes than window size to force wrap-around
        total_steps = W + 2  # 5 writes for W=3
        values = [float(20 + i) for i in range(total_steps)]
        for val in values:
            k_new = torch.full((B, H, 1, D), val)
            v_new = torch.full((B, H, 1, D), val + 1000.0)
            cache.write(layer=0, k_new=k_new, v_new=v_new)
            cache.advance()

        # View should return last W items in chronological order
        k, v = cache.view(layer=0)
        self.assertEqual(k.shape, (B, W, H, D))
        self.assertEqual(v.shape, (B, W, H, D))

        expected_vals = values[-W:]  # last W entries
        expected_k = torch.cat([torch.full((B, 1, H, D), val) for val in expected_vals], dim=1)
        expected_v = torch.cat([torch.full((B, 1, H, D), val + 1000.0) for val in expected_vals], dim=1)

        torch.testing.assert_close(k, expected_k)
        torch.testing.assert_close(v, expected_v)

    def test_write_accepts_B_1_H_D_input(self):
        # Verify write() transposes inputs of shape (B, 1, H, D) correctly
        L, B, H, W, D = 1, 2, 3, 2, 4
        cache = KVCache(L=L, B=B, H=H, W=W, D=D, device="cpu", dtype=torch.float32)

        # Provide (B, 1, H, D) input
        k_new = torch.randn(B, 1, H, D)
        v_new = torch.randn(B, 1, H, D)
        cache.write(layer=0, k_new=k_new, v_new=v_new)
        cache.advance()

        # View should return exactly the provided tensors (sequence length 1)
        k_view, v_view = cache.view(layer=0)
        self.assertEqual(k_view.shape, (B, 1, H, D))
        self.assertEqual(v_view.shape, (B, 1, H, D))
        torch.testing.assert_close(k_view, k_new)
        torch.testing.assert_close(v_view, v_new)

    def test_reset_resets_time_and_view(self):
        L, B, H, W, D = 1, 2, 2, 3, 2
        cache = KVCache(L=L, B=B, H=H, W=W, D=D, device="cpu", dtype=torch.float32)

        # Do some writes and advances
        for i in range(2):
            val = float(30 + i)
            cache.write(layer=0,
                        k_new=torch.full((B, H, 1, D), val),
                        v_new=torch.full((B, H, 1, D), val + 1000.0))
            cache.advance()

        # Reset should clear time and make view empty again
        cache.reset()
        self.assertEqual(cache.t, 0)
        k, v = cache.view(layer=0)
        self.assertEqual(k.shape, (B, 0, H, D))
        self.assertEqual(v.shape, (B, 0, H, D))


if __name__ == "__main__":
    unittest.main()

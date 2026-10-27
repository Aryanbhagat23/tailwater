"""Minimal numpy MLP with manual backprop and Adam — enough for PPO on a
12-dimensional observation without a deep-learning dependency."""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale: float = 1.0):
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = out_scale if i == len(sizes) - 2 else np.sqrt(2.0)
            q, _ = np.linalg.qr(rng.standard_normal((max(a, b), max(a, b))))
            self.W.append(q[:a, :b] * scale)
            self.b.append(np.zeros(b))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x):
        hs = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            hs.append(h)
        return h, hs

    def __call__(self, x):
        return self.forward(x)[0]

    def backward(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = dout
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - hs[i] ** 2)
        return gW + gb

    def state(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_state(cls, st):
        obj = cls.__new__(cls)
        obj.W = [np.array(w) for w in st["W"]]
        obj.b = [np.array(b) for b in st["b"]]
        return obj


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, max_norm=0.5):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, *betas, eps
        self.max_norm = max_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if self.max_norm and norm > self.max_norm:
            grads = [g * (self.max_norm / norm) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

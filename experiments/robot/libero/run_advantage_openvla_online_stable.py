#!/usr/bin/env python
"""Stable entry point that delays the cached upstream OpenVLA loader until after EGL setup."""

import torch

import run_advantage_openvla_online as online


_from_pretrained = online.AutoModelForVision2Seq.from_pretrained


def stable_from_pretrained(model_path, *args, **kwargs):
    kwargs["trust_remote_code"] = True
    kwargs["max_memory"] = {
        **{index: "7GiB" for index in range(torch.cuda.device_count())},
        "cpu": "64GiB",
    }
    return _from_pretrained(model_path, *args, **kwargs)


online.AutoModelForVision2Seq.from_pretrained = stable_from_pretrained


if __name__ == "__main__":
    online.main()

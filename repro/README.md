## Usage

```bash
python test_fwd_inv_dynamics.py --device cuda
python test_fwd_inv_dynamics.py --device cpu
```

## Description
Observed behavior is that when using cpu, L2 error is very small as expected. However, when using cuda as device, errors are much larger.

# Required code fixes before reviewer reruns

## 1. Efficient-SU2 repetitions are currently ignored

In `experiments/qurift_main.py`, make these three changes.

### Add the field to `QFCConfig`

```python
fm_eff_reps: int = 1
```

### Put the CLI value into `mapper_cfg`

```python
mapper_cfg |= dict(
    fm_eff_reps=args.fm_eff_reps,
    fm_eff_alpha=1.0,
    fm_eff_ent_kind=args.fm_eff_ent_kind,
    fm_eff_pad_mod=args.fm_eff_pad_mod,
    fm_eff_twoq_op=args.fm_eff_twoq_op,
)
```

### Pass it to the encoder builder

```python
su2_name, su2_op = build_efficient_su2_oplist_qisk_new(
    D=self.D,
    n_wires=cfg.n_wires,
    single_ops=("ry", "rz"),
    entanglement=cfg.fm_eff_ent_kind,
    twoq=cfg.fm_eff_twoq_op,
    pad_mode=cfg.fm_eff_pad_mod,
    reps=cfg.fm_eff_reps,
    alpha=cfg.fm_eff_alpha,
)
```

Add a smoke assertion before a full rerun: for the same width and data dimension,
the emitted Efficient-SU2 operation-list length must increase between reps 1 and 5.

## 2. Add a target-model seed argument

Add:

```python
parser.add_argument("--seed", type=int, default=43)
```

Replace dataset-specific hardcoded seeds with:

```python
seed = int(args.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 3. Store complete metadata in attack exports

Extend `payload["meta"]` with:

```python
"seed": int(args.seed),
"fm_kind": args.fm_kind,
"reps": int(
    args.fm_z_reps if args.fm_kind == "z"
    else args.fm_zz_reps if args.fm_kind == "zz"
    else args.fm_eff_reps
),
"pad_mode": (
    args.fm_z_pad_mode if args.fm_kind == "z"
    else args.fm_zz_pad_mode if args.fm_kind == "zz"
    else args.fm_eff_pad_mod
),
"fm_ent": (
    "NA" if args.fm_kind == "z"
    else args.fm_zz_entanglement if args.fm_kind == "zz"
    else args.fm_eff_ent_kind
),
"fm_op": "NA" if args.fm_kind in {"z", "zz"} else args.fm_eff_twoq_op,
"trainable_params": count_trainable_params(model),
```

## 4. Seed the MIA model itself

Near the start of `run_one_target` in `train_mia_attack.py`:

```python
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

## 5. Standardize architecture-comparison protocols

For any new QNN/HQNN/QCNN comparison, use the same:

- train/validation/test counts;
- epochs;
- target seed and split;
- batch size;
- stopping rule.

Report each model's learning rate, trainable parameter count, and quantum gate count.
Describe this as a comparison of complete model wrappers unless the classical
preprocessors and heads are separately matched or frozen.

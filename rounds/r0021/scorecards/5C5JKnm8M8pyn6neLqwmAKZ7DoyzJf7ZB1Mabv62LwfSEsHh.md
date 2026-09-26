**This round:** not ranked · Δ vs baseline -0.056 on 6 paired instances · 0 verified.

## Round `r0021` — `5C5JKnm8M8pyn6neLqwmAKZ7DoyzJf7ZB1Mabv62LwfSEsHh`

**weight 0.1494** · score 0.1035

| | |
|---|---|
| episodes | 48 |
| mean d (your share of checks passed − the baseline's, same instances) | +0.1035 |
| standard error (incl. reference term) | 0.042345 |
| Δc (one-sided 90 % lower bound — how sure the gain is) | 0.0493 |
| score (mean d after the overfit and copy penalties) | +0.1035 |
| correctness gate | passed |
| Δe | — |
| overfit rate | 0.00 |
| disqualified episodes | 0 |

`mean d` is the share of each task's withheld checks your episodes passed, minus the pinned model's share on the *same instances* with no strategy. Passing tasks is not the achievement — beating that baseline is. `Δc` is the lower bound of that difference, so beating the baseline on average is not enough to be *paid* for beating it.

### The baseline you were measured against

| family | null n | null credit | canon credit | Δc canon | label |
|---|---|---|---|---|---|
| `swe_fix` | 48 | 0.11 | 0.00 | -0.1061 | frontier |

### Check the grading yourself

6 of 6 withheld commitments re-verified at close: **all match**.

Each instance's withheld half was committed to *before* submissions opened, as `hmac-sha256(salt, canonical_json(withheld))`. The commitment is in the task record — under `rounds/<id>/tasks/` when you were shown the scored tasks, under `rounds/<id>/evaluated/` when you were shown previews — and `rounds/queue.json` carried the digest of those records before the round opened. The salt and the half itself are published now, in `reveal.json`. Recompute it and confirm the criteria you were graded against are the ones that were fixed in advance:

```python
import hashlib, hmac, json
salt, withheld = reveal[task_id]["salt"], reveal[task_id]["withheld"]
body = json.dumps(withheld, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
"hmac-sha256:" + hmac.new(bytes.fromhex(salt), body, hashlib.sha256).hexdigest()
```

<details><summary>Revealed withheld halves (6) — full record in `reveal.json`</summary>

| task | withheld checks | salt | source |
|---|---|---|---|
| `swe-fix-r0021-00` | 9 | `9db163a8a9a2e683…` | cantools__cantools.0c6a7871.lm_rewrite__0teb6d6o |
| `swe-fix-r0021-01` | 3 | `30ff787f9fd75fde…` | cantools__cantools.0c6a7871.func_pm_remove_cond__8b9tpn0f |
| `swe-fix-r0021-03` | 1 | `220ad4f53141f47f…` | marshmallow-code__marshmallow.9716fc62.lm_rewrite__pg8be73q |
| `swe-fix-r0021-05` | 1 | `5c8fb49fb04e004b…` | andialbrecht__sqlparse.e57923b3.pr_792 |
| `swe-fix-r0021-06` | 8 | `7e5d478996fa4a45…` | cantools__cantools.0c6a7871.func_pm_remove_assign__erxeprm6 |
| `swe-fix-r0021-07` | 3 | `600bdb71437a00d2…` | tkrajina__gpxpy.09fc46b3.func_pm_remove_cond__qrnj3ni7 |

</details>
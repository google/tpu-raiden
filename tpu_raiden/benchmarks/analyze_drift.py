#!/usr/bin/env python3
"""Offline analysis of canary_drift.json (download it from the BAP run artifacts).

Everything here comes from the ONE probe run -- no extra CI needed:

  lag sweep    : delta between rounds separated by k measurements. The wall-clock
                 gap grows with k, so this traces how drift accumulates with time
                 and tells you what a COLOCATED A/B (whose baseline->experiment gap
                 is a teardown + rerun) will actually face.
  iters sweep  : each round kept its raw per-iteration samples, so a smaller iters
                 is simulated by taking the median of the first m of them. This is
                 the knob that buys a tighter threshold.

Usage:  python3 analyze_drift.py canary_drift.json
"""
import json
import statistics
import sys


def pct(xs, q):
  xs = sorted(xs)
  if not xs:
    return float('nan')
  if len(xs) == 1:
    return xs[0]
  pos = (len(xs) - 1) * q / 100.0
    # linear interpolation, same convention as the gate's _pct
  lo = int(pos)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def deltas_at_lag(vals, lag):
  """|relative delta| in %, between rounds `lag` apart."""
  return [abs(vals[i + lag] - vals[i]) / vals[i] * 100.0
          for i in range(len(vals) - lag) if vals[i] > 0]


def main(path):
  d = json.load(open(path))
  series, iters_rec = d['series'], d['iters']
  labels = sorted({s['label'] for s in series})
  print(f'{d["rounds"]} rounds, iters={iters_rec}, configs={labels}\n')

  # ---- 1. drift vs lag (at the recorded iters) -------------------------------
  print('=' * 72)
  print('|delta| vs LAG  -- how far two same-machine measurements drift apart')
  print('  lag 1 == consecutive rounds, the closest analogue of a COLOCATED A/B')
  print('=' * 72)
  for label in labels:
    rows = [s for s in series if s['label'] == label]
    for dirn in ('d2h', 'h2d'):
      vals = [r[f'{dirn}_gbps'] for r in rows]
      print(f'\n{label} {dirn}   (n={len(vals)} rounds)')
      print(f'  {"lag":>4}{"median":>10}{"p95":>9}{"p99":>9}{"max":>9}{"n":>6}')
      for lag in (1, 2, 3, 5, 10):
        if lag >= len(vals):
          break
        ds = deltas_at_lag(vals, lag)
        print(f'  {lag:>4}{statistics.median(ds):9.2f}%{pct(ds, 95):8.2f}%'
              f'{pct(ds, 99):8.2f}%{max(ds):8.2f}%{len(ds):6}')

  # ---- 2. drift vs iters (lag 1, resampled from the raw samples) -------------
  print('\n' + '=' * 72)
  print('|delta| at LAG 1 vs ITERS  -- noise you can buy back by measuring longer')
  print('  simulated by taking the median of the first m raw samples per round')
  print('=' * 72)
  sweep = [m for m in (5, 10, 20, 30, 50, 100) if m <= iters_rec]
  for label in labels:
    rows = [s for s in series if s['label'] == label]
    for dirn in ('d2h', 'h2d'):
      print(f'\n{label} {dirn}')
      print(f'  {"iters":>6}{"median":>10}{"p95":>9}{"p99":>9}{"max":>9}')
      for m in sweep:
        vals = [statistics.median(r[f'{dirn}_gbps_all'][:m]) for r in rows]
        ds = deltas_at_lag(vals, 1)
        print(f'  {m:>6}{statistics.median(ds):9.2f}%{pct(ds, 95):8.2f}%'
              f'{pct(ds, 99):8.2f}%{max(ds):8.2f}%')

  # ---- 3. the number you actually need ---------------------------------------
  print('\n' + '=' * 72)
  print('THRESHOLD: must sit ABOVE the p99 below, or the A/B cries wolf on noise')
  print('=' * 72)
  worst = 0.0
  for label in labels:
    rows = [s for s in series if s['label'] == label]
    for dirn in ('d2h', 'h2d'):
      vals = [r[f'{dirn}_gbps'] for r in rows]
      p99 = pct(deltas_at_lag(vals, 1), 99)
      worst = max(worst, p99)
      print(f'  {label:<18}{dirn:<5}p99 = {p99:5.2f}%')
  print(f'\n  worst p99 across configs = {worst:.2f}%  '
        f'-> threshold >= ~{worst * 1.5:.1f}% (1.5x headroom) at iters={iters_rec}')
  print('  (raise iters to push this down -- see the iters sweep above)')


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'canary_drift.json')

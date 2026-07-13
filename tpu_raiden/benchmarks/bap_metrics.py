"""Writes V1 TensorBoard scalar events for BAP to ingest.

Follows BAP's documented "zero heavy dependencies" path (tensorboard protos
directly -- no tensorflow, no torch).
"""

import os
import sys
import time

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def emit(scalars):
  """Writes all scalars in one event file.

  Args:
    scalars: dict mapping tag -> float. Tags MUST match the `metrics.name`
      entries in the benchmark registry.
  """
  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tbdir:
    # Not running under BAP (e.g. local `bazel run`). Nothing to ingest.
    print('TENSORBOARD_OUTPUT_DIR not set; skipping metric emission.',
          file=sys.stderr)
    return

  writer = EventFileWriter(tbdir)
  for tag, value in scalars.items():
    writer.add_event(
        event_pb2.Event(
            step=0,
            wall_time=time.time(),
            summary=summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(
                    tag=tag, simple_value=float(value))]),
        ))
  writer.close()
  print(f'Wrote {len(scalars)} metrics -> {tbdir}')

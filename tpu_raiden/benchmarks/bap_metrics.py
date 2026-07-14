# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Emits V1 TensorBoard scalar events for BAP to ingest.

A .tfevents file is just a TFRecord stream of serialized Event protos, and
tensorflow ships both halves: the protos and TFRecordWriter's length+CRC
framing. Writing them directly is the only path open to us here:

  * tf.summary.scalar (the TF2 API) is a shim that delegates to the `tensorboard`
    package -- see tensorflow/python/summary/tb_summary.py. tensorboard is not in
    the pinned pip hub, so it raises TBNotInstalledError.
  * tf.compat.v1.summary.FileWriter refuses to construct under eager execution,
    which is on by default in TF2.
  * tensorboardX / tensorboard would each need a new pip hub; tensorflow is
    already a hub package.

tensorflow is a declared Bazel dep, so the import is unconditional and a write
failure is fatal: a silent metric drop would let a job go green while feeding
nothing to the dashboard or the A/B analyzer -- which is exactly what the old
best-effort path did.
"""

import os
import socket
import sys
import time

# Bumped per emit() call. TFRecordWriter truncates, so two calls landing in the
# same wall-clock second would otherwise clobber each other's event file.
_seq = 0


def emit(scalars):
  """Writes scalars to an event file under TENSORBOARD_OUTPUT_DIR.

  Args:
    scalars: dict mapping tag -> float, or tag -> sequence of floats. A sequence
      is written as one event per step, which lets BAP compute stats (MEAN, P99,
      ...) over the real distribution instead of a single pre-reduced number --
      the A/B analyzer needs that spread to see through sub-1% margins. Each tag
      MUST exactly match a metrics{name: ...} entry in the benchmark registry,
      or BAP ingests nothing for it.
  """
  tblog_dir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tblog_dir:
    # Not running under BAP (e.g. a local `bazel run`); nothing to ingest.
    print('TENSORBOARD_OUTPUT_DIR not set; skipping metric emission.',
          file=sys.stderr)
    return

  # Imported here rather than at module scope: tensorflow is heavy, and pulling
  # it in before the timed transfers would perturb the bandwidth being measured
  # (the tightest configs gate on <1% margin). By the time emit() runs every
  # measurement is done, and nothing below touches the accelerator.
  # pylint: disable=g-import-not-at-top
  import tensorflow as tf
  from tensorflow.core.framework import summary_pb2
  from tensorflow.core.util import event_pb2
  # pylint: enable=g-import-not-at-top

  global _seq
  _seq += 1
  # TensorBoard readers (and BAP's parser) discover event files by the
  # "events.out.tfevents." prefix; pid+seq keep concurrent or repeated writers
  # from colliding (baseline and experiment share a host in A/B COLOCATED mode).
  path = os.path.join(
      tblog_dir,
      f'events.out.tfevents.{int(time.time())}.{socket.gethostname()}'
      f'.{os.getpid()}.{_seq}')

  points = 0
  with tf.io.TFRecordWriter(path) as writer:
    for tag, value in scalars.items():
      series = value if isinstance(value, (list, tuple)) else [value]
      for step, v in enumerate(series):
        writer.write(
            event_pb2.Event(
                step=step,
                wall_time=time.time(),
                summary=summary_pb2.Summary(value=[
                    summary_pb2.Summary.Value(tag=tag, simple_value=float(v))
                ]),
            ).SerializeToString())
        points += 1

  print(f'Wrote {points} point(s) across {len(scalars)} metric(s) -> {path}',
        flush=True)

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

"""Emits TensorBoard scalar events for BAP to ingest.

Uses tf.summary -- option 1 of BAP's onboarding guide, which parses both V1 and
V2 event formats. tensorflow is the only writer library carried by the pinned
pip hub (jax_pypi), so the import is unconditional: a silent metric drop would
let a job go green while feeding nothing to the dashboard or the A/B analyzer.
"""

import os
import sys


def emit(scalars):
  """Writes every scalar into one event file under TENSORBOARD_OUTPUT_DIR.

  Args:
    scalars: dict mapping tag -> value. Each tag MUST exactly match a
      metrics{name: ...} entry in the benchmark registry, or BAP ingests
      nothing for it.
  """
  tblog_dir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tblog_dir:
    # Not running under BAP (e.g. a local `bazel run`); nothing to ingest.
    print('TENSORBOARD_OUTPUT_DIR not set; skipping metric emission.',
          file=sys.stderr)
    return

  # Imported here rather than at module scope: tensorflow is heavy, and pulling
  # it in before the timed transfers would perturb the bandwidth being measured
  # (the tightest configs gate on <1% margin). By the time emit() runs, every
  # measurement is done. tf.summary itself only writes files -- it never touches
  # the accelerator, so it does not contend with JAX/libtpu.
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  writer = tf.summary.create_file_writer(tblog_dir)
  with writer.as_default():
    for tag, value in scalars.items():
      tf.summary.scalar(tag, float(value), step=0)
  writer.flush()
  writer.close()
  print(f'Wrote {len(scalars)} metric(s) -> {tblog_dir}', flush=True)

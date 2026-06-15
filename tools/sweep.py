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

"""Optimization sweep script for JAX Raiden benchmark."""

import itertools
import os
import re
import subprocess
import time


def run_experiment(num_blocks, block_size, num_layers, parallelism):
  """Runs a single experiment trial with given parameters."""
  # Determine the IP addresses for dual-nic
  # Primary IP is TPU worker IP; secondary is from second NIC.
  # In a typical setup, we might need to query the IPs or use known patterns.
  # For now, assume PEER_IPS is set in the environment.
  peer_ips = os.environ.get("PEER_IPS", "")

  sender_cmd = [
      "/mnt/disks/persistent/amylin/.venv/bin/python3",
      "tools/perf_test_runner.py",
      "--role=sender",
      f"--num_blocks={num_blocks}",
      f"--block_size={block_size}",
      f"--num_layers={num_layers}",
      f"--parallelism={parallelism}",
  ]
  if peer_ips:
    sender_cmd.append(f"--peer_ips={peer_ips}")

  receiver_cmd = [
      "/mnt/disks/persistent/amylin/.venv/bin/python3",
      "tools/perf_test_runner.py",
      "--role=receiver",
      "--peer=localhost",  # Since both run on the same VM for loopback test
      f"--num_blocks={num_blocks}",
      f"--block_size={block_size}",
      f"--num_layers={num_layers}",
      f"--parallelism={parallelism}",
  ]
  if peer_ips:
    receiver_cmd.append(f"--peer_ips={peer_ips}")

  print(
      f"Running sweep: blocks={num_blocks}, block_size={block_size},"
      f" layers={num_layers}, parallelism={parallelism}"
  )

  cwd = "/mnt/disks/persistent/amylin/tpu_raiden_oss"
  env = os.environ.copy()
  env["PYTHONPATH"] = cwd
  env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

  env_sender = env.copy()
  env_sender["TPU_VISIBLE_CHIPS"] = "0"
  env_receiver = env.copy()
  env_receiver["TPU_VISIBLE_CHIPS"] = "1"
  env["PYTHONUNBUFFERED"] = "1"

  # Remove lockfile to avoid multi-process conflicts
  subprocess.call(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"])

  # Start sender
  sender_proc = subprocess.Popen(
      sender_cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      cwd=cwd,
      env=env_sender,
  )
  time.sleep(30)  # Give sender time to initialize and populate 16GB of cache!

  # Start receiver
  receiver_proc = subprocess.Popen(
      receiver_cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      cwd=cwd,
      env=env_receiver,
  )

  try:
    out, _ = receiver_proc.communicate(timeout=60)
  except subprocess.TimeoutExpired:
    out = "Receiver timed out."
    receiver_proc.kill()

  try:
    sender_out, _ = sender_proc.communicate(timeout=10)
  except subprocess.TimeoutExpired:
    sender_proc.kill()
    sender_out, _ = sender_proc.communicate()
  finally:
    sender_proc.kill()
    receiver_proc.kill()
    subprocess.call(["sudo", "pkill", "-9", "-f", "perf_test_runner.py"])
    subprocess.call(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"])
    time.sleep(15)  # Wait for kernel to finish vfio-pci reset

  # Parse bandwidth
  match = re.search(r"Effective Bandwidth:\s+([\d\.]+)\s+Gbps", out)
  if match:
    bw = float(match.group(1))
    print(f"Result: {bw} Gbps")
    return bw
  else:
    print("Failed to parse bandwidth. Receiver Output:")
    print(out)
    print("Sender Output:")
    print(sender_out)
    return 0.0


def main():
  blocks = [256, 512, 1024]
  block_sizes = [2, 4, 8]
  layers = [8, 16]
  parallelisms = [1, 2, 4]

  best_bw = 0.0
  best_params = None

  for num_blocks, block_size, num_layers, parallelism in itertools.product(
      blocks, block_sizes, layers, parallelisms
  ):
    bw = run_experiment(num_blocks, block_size, num_layers, parallelism)
    if bw > best_bw:
      best_bw = bw
      best_params = (num_blocks, block_size, num_layers, parallelism)

    if bw > 250.0:
      print(f"SUCCESS! Exceeded 250 Gbps: {bw} Gbps with params {best_params}")
      break

  print(f"Best Bandwidth: {best_bw} Gbps with params {best_params}")


if __name__ == "__main__":
  main()

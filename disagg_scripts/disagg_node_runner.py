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

"""Two-node real-disaggregation harness for ``DisaggKVCacheManager``.

Runs exactly ONE role of a cross-host KV push:

    prefill (LOCAL):  device KV --D2H--> host staging --H2H push--> decode host
    decode  (REMOTE): host staging --H2D--> device KV, then verify the bytes

Unlike ``api/jax/disagg_kv_cache_manager_test.py`` (both managers in one
process on 127.0.0.1), here prefill and decode are separate processes on
different physical TPU hosts. They rendezvous through the ZMQ discovery proxy
(``kv_cache/disagg_proxy.py``) because the manager's control/data ports are
ephemeral: the decode REGISTERs its ``{ip, zmq_port, transport_port}`` *after*
it has queued its receive request, and the prefill RESOLVEs it before pushing.

Verification uses a deterministic data pattern (int32 arange + layer index) so
each side can independently compute the expected result without shipping the
payload out of band.

This is a *baseline* that passes a single PREFILL_D2H->H2H->DECODE_H2D transfer
end to end. To write your own scenarios, extend ``build_reference_layer`` /
``verify_decode`` or add transfer plans; the orchestration plumbing is reused.
"""

import argparse
import sys
import threading
import time

import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Discovery-proxy client (talks to kv_cache/disagg_proxy.py).
# ---------------------------------------------------------------------------
def _proxy_call(endpoint: str, msg: str, timeout_ms: int = 10000) -> str:
  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.REQ)
  sock.setsockopt(zmq.LINGER, 0)
  sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
  sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
  sock.connect(endpoint)
  try:
    sock.send_string(msg)
    return sock.recv_string()
  finally:
    sock.close()


def proxy_register(endpoint, name, ip, zmq_port, trans_port):
  reply = _proxy_call(endpoint, f"REGISTER:{name}:{ip}:{zmq_port}:{trans_port}")
  if reply != "OK":
    raise RuntimeError(f"proxy REGISTER failed: {reply}")


def proxy_resolve(endpoint, name, timeout_s=120.0):
  """Block until ``name`` is registered, then return (ip, zmq_port, trans_port)."""
  deadline = time.time() + timeout_s
  while time.time() < deadline:
    reply = _proxy_call(endpoint, f"RESOLVE:{name}")
    if reply.startswith("OK:"):
      _, ip, zmq_port, trans_port = reply.split(":")
      return ip, int(zmq_port), int(trans_port)
    time.sleep(0.5)
  raise TimeoutError(f"proxy RESOLVE for '{name}' timed out after {timeout_s}s")


# ---------------------------------------------------------------------------
# Deterministic payload (shared formula => both hosts agree without shipping
# the data). Override these two hooks to write a different test.
# ---------------------------------------------------------------------------
def build_reference_layer(shape, np_dtype, layer_idx):
  """The full per-layer reference array, identical on prefill and decode."""
  n = int(np.prod(shape))
  flat = (np.arange(n, dtype=np.int64) + layer_idx).reshape(shape)
  return flat.astype(np_dtype)


def verify_decode(decode_arrays, plan, shape, np_dtype):
  """Check that each pushed chunk landed at its destination on every layer.

  ``plan`` is a list of (src_off, dst_off, size) in major-dim units.
  Returns (ok: bool, message: str).
  """
  for layer_idx, dev_arr in enumerate(decode_arrays):
    got = np.asarray(dev_arr)
    ref = build_reference_layer(shape, np_dtype, layer_idx)
    for src_off, dst_off, size in plan:
      want = ref[src_off:src_off + size]
      have = got[dst_off:dst_off + size]
      # Compare raw bytes: the transfer is a byte copy, and byte equality is
      # exact for every dtype (and unlike value equality is robust to NaN/inf
      # that low-precision dtypes like fp8 can produce from the test pattern).
      if want.tobytes() != have.tobytes():
        return False, (
            f"layer {layer_idx}: dst[{dst_off}:{dst_off+size}] != "
            f"src[{src_off}:{src_off+size}] (bytes differ)"
        )
  return True, "all chunks verified"


# ---------------------------------------------------------------------------
# JAX device-array setup (one sharded array per layer, model-parallel over all
# local devices). Imports are deferred so --help works without TPU.
# ---------------------------------------------------------------------------
def make_device_arrays(shape, dtype_name, n_layers, device_type, fill):
  import jax
  import jax.numpy as jnp
  import ml_dtypes

  dtype_map = {
      "int32": jnp.int32,
      "bf16": jnp.bfloat16,
      "fp32": jnp.float32,
      "fp8": jnp.float8_e4m3fn,
  }
  jnp_dtype = dtype_map[dtype_name]
  # numpy dtype used to BUILD the payload and to reconstruct the expected bytes
  # in verify_decode. Must be the real dtype (bf16/fp8 via ml_dtypes), never
  # None, or the reference bytes won't match the on-device bytes.
  np_dtype = {"int32": np.int32, "bf16": ml_dtypes.bfloat16,
              "fp32": np.float32, "fp8": ml_dtypes.float8_e4m3fn}[dtype_name]

  devices = jax.devices(device_type)
  if not devices:
    raise RuntimeError(f"no {device_type} devices found")
  num = len(devices)
  mesh = jax.sharding.Mesh(
      np.array(devices).reshape((1, num)), ("data", "model"))
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(None, None, "model"))

  arrays = []
  for i in range(n_layers):
    if fill == "reference":
      # Build directly in the target dtype so the on-device bytes are identical
      # to what verify_decode reconstructs from the same formula.
      host = jnp.asarray(build_reference_layer(shape, np_dtype, i))
    else:  # "zeros"
      host = jnp.zeros(shape, dtype=jnp_dtype)
    arrays.append(jax.device_put(host, sharding))
  jax.block_until_ready(arrays)
  return arrays, num, np_dtype


def parse_offsets(s):
  return [int(x) for x in s.split(",") if x != ""]


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--role", required=True, choices=["prefill", "decode"])
  ap.add_argument("--proxy-endpoint", required=True,
                  help="tcp://<local_ip>:<proxy_port>")
  ap.add_argument("--self-name", required=True)
  ap.add_argument("--self-ip", required=True,
                  help="address the peer dials this engine on")
  ap.add_argument("--peer-name", default="decode",
                  help="prefill: name of the decode peer to resolve")
  ap.add_argument("--mode", default="push", choices=["push", "pull"],
                  help="push: prefill pushes to decode; pull: decode pulls "
                       "from prefill")
  ap.add_argument("--n-layers", type=int, default=2)
  ap.add_argument("--block-size", type=int, default=2)
  ap.add_argument("--dtype", default="int32",
                  choices=["int32", "bf16", "fp32", "fp8"])
  ap.add_argument("--device", default="tpu", choices=["tpu", "cpu"])
  ap.add_argument("--transport-parallelism", type=int, default=1,
                  help="parallel TCP streams per H2H Push/Pull (BlockTransport)")
  ap.add_argument("--worker-parallelism", type=int, default=1,
                  help="concurrent H2H worker threads (transfers in flight)")
  ap.add_argument("--request-id", type=int, default=1001,
                  help="base request id; request i uses request-id + i")
  ap.add_argument("--num-requests", type=int, default=1,
                  help="number of concurrent requests to fire")
  ap.add_argument("--shape", default="8,128,8,8,128",
                  help="per-layer KV array shape (comma-separated); the major "
                       "dim auto-grows to fit all requests")
  ap.add_argument("--src-offsets", default="4,6",
                  help="per-request src chunks; subsequent requests are shifted")
  ap.add_argument("--dst-offsets", default="0,2")
  ap.add_argument("--sizes", default="2,2")
  ap.add_argument("--timeout", type=float, default=30.0)
  args = ap.parse_args()

  # In CPU mode, pin JAX to the CPU backend so importing jax doesn't probe (and
  # fail on) a TPU that another job may be holding. Must precede `import jax`.
  if args.device == "cpu":
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

  shape = tuple(parse_offsets(args.shape))
  base_src = parse_offsets(args.src_offsets)
  base_dst = parse_offsets(args.dst_offsets)
  sizes = parse_offsets(args.sizes)
  span = sum(sizes)  # per-request major-dim footprint; tile requests by this

  # Build N non-overlapping request plans by shifting both src and dst by i*span,
  # so concurrent requests touch disjoint regions (and stay independently
  # verifiable). Each entry: (request_id, src_offsets, dst_offsets, sizes).
  requests = []
  for i in range(args.num_requests):
    requests.append((
        args.request_id + i,
        [s + i * span for s in base_src],
        [d + i * span for d in base_dst],
        list(sizes),
    ))

  # Flat (src, dst, size) chunk list across all requests, used for verification.
  plan = [(s, d, sz)
          for _, src_i, dst_i, szs in requests
          for s, d, sz in zip(src_i, dst_i, szs)]

  # Grow the major dim so every chunk fits; honor a larger user --shape.
  major = shape[0]
  for _, src_i, dst_i, szs in requests:
    for offs in (src_i, dst_i):
      major = max(major, max(o + sz for o, sz in zip(offs, szs)))
  shape = (major,) + shape[1:]
  host_blocks = max(1, (major + args.block_size - 1) // args.block_size)

  from api.jax import disagg_kv_cache_manager as dm

  fill = "reference" if args.role == "prefill" else "zeros"
  arrays, num_dev, np_dtype = make_device_arrays(
      shape, args.dtype, args.n_layers, args.device, fill)
  print(f"[{args.role}] {num_dev} {args.device} devices; "
        f"{args.n_layers} layers shape={shape} block_size={args.block_size}",
        flush=True)

  manager = dm.DisaggKVCacheManager(
      device_arrays=arrays,
      block_size=args.block_size,
      local_port=0,
      host_blocks_to_allocate=host_blocks,
      unsafe_skip_buffer_lock=(args.device == "cpu"),
      transport_parallelism=args.transport_parallelism,
      worker_parallelism=args.worker_parallelism,
  )
  manager.start()
  time.sleep(0.2)  # let ports bind
  my_zmq = manager.zmq_control_port()
  my_trans = manager.local_port()
  print(f"[{args.role}] zmq_port={my_zmq} transport_port={my_trans} "
        f"num_requests={len(requests)}", flush=True)

  # Latch over all request ids; all_done fires once every callback has landed.
  pending = {rid for rid, *_ in requests}
  lock = threading.Lock()
  all_done = threading.Event()
  errors = []

  def make_cb(rid):
    def cb(status):
      if status is not None:
        errors.append(f"req {rid}: {status}")
      with lock:
        pending.discard(rid)
        if not pending:
          all_done.set()
    return cb

  n = len(requests)
  pull = (args.mode == "pull")
  peer_name = "prefill" if args.role == "decode" else "decode"

  def register_self():
    proxy_register(args.proxy_endpoint, args.self_name, args.self_ip,
                   my_zmq, my_trans)

  def connect_to(name):
    ip, pz, pt = proxy_resolve(args.proxy_endpoint, name)
    manager.register_peer(name, ip, pz, pt)
    print(f"[{args.role}] connected to '{name}' -> {ip} (zmq={pz}, trans={pt})",
          flush=True)

  rc = 0
  try:
    if args.role == "decode":
      if pull:
        # PULL: the prefill registers up front; resolve+connect to it, queue
        # the pull receives, THEN advertise readiness so the prefill only
        # stages (and sends NOTIFY_READY) once our receives are pending.
        connect_to(peer_name)
      for rid, _src_i, dst_i, szs in requests:
        manager.submit_request(
            request_id=rid,
            req_type=dm.DisaggTransferRequestType.DECODE_H2D,
            dst_offsets=dst_i, sizes=szs,
            peer=(peer_name if pull else ""), pull=pull,
            callback=make_cb(rid))
      register_self()  # readiness signal (prefill's RESOLVE unblocks on this)
      print(f"[decode] {n} request(s) queued ({args.mode}); waiting...",
            flush=True)
      if not all_done.wait(timeout=args.timeout):
        raise TimeoutError(f"decode incomplete; pending={sorted(pending)}")
      if errors:
        raise RuntimeError("decode callback error: " + "; ".join(errors))
      ok, msg = verify_decode(arrays, plan, shape, np_dtype)
      print(f"[decode] verify ({n} req, {len(plan)} chunks): {msg}", flush=True)
      rc = 0 if ok else 1

    else:  # prefill
      if pull:
        register_self()  # decode must be able to resolve+pull from us
      connect_to(peer_name)  # blocks until the decode has queued its receives
      for rid, src_i, dst_i, szs in requests:
        manager.submit_request(
            request_id=rid,
            req_type=dm.DisaggTransferRequestType.PREFILL_D2H,
            src_offsets=src_i, dst_offsets=dst_i, sizes=szs,
            peer=peer_name, pull=pull,
            callback=make_cb(rid))
      print(f"[prefill] submitted {n} PREFILL_D2H ({args.mode}); waiting...",
            flush=True)
      if not all_done.wait(timeout=args.timeout):
        raise TimeoutError(f"prefill incomplete; pending={sorted(pending)}")
      if errors:
        raise RuntimeError("prefill callback error: " + "; ".join(errors))
      print(f"[prefill] all {n} transfers complete.", flush=True)
      rc = 0
  finally:
    manager.stop()

  print(f"[{args.role}] {'PASS' if rc == 0 else 'FAIL'}", flush=True)
  return rc


if __name__ == "__main__":
  sys.exit(main())

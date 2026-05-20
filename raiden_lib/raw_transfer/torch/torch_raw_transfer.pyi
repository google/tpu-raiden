class PjRtCopyFuture:
  def Await(self) -> None: ...
  def wait(self) -> None: ...
  def IsReady(self) -> bool: ...
  def is_ready(self) -> bool: ...

class RawHostBuffer:
  def __init__(self, size_bytes: int) -> None: ...
  @property
  def size_bytes(self) -> int: ...
  @property
  def data_ptr(self) -> int: ...
  @property
  def is_pjrt_backed(self) -> bool: ...

class PreparedTorchRawTransfer:
  def __init__(
      self,
      tpu_tensor,
      host_buffer: RawHostBuffer,
      unsafe_skip_buffer_lock: bool = ...,
  ) -> None: ...
  @property
  def physical_size_bytes(self) -> int: ...
  @property
  def host_buffer(self) -> RawHostBuffer: ...
  def d2h_async(self) -> PjRtCopyFuture: ...
  def h2d_async(self) -> PjRtCopyFuture: ...
  def d2h(self) -> None: ...
  def h2d(self) -> None: ...

def await_all(futures: PjRtCopyFuture | list[PjRtCopyFuture]) -> None: ...
def is_ready(futures: PjRtCopyFuture | list[PjRtCopyFuture]) -> bool: ...

def transfer_d2h_async(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_h2d_async(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_d2h(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None: ...

def transfer_h2d(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None: ...

def transfer_d2h_batch_async(
    src_arrs: list,
    dst_arrs: list,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_h2d_batch_async(
    src_arrs: list,
    dst_arrs: list,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> PjRtCopyFuture: ...

def transfer_d2h_batch(
    src_arrs: list,
    dst_arrs: list,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None: ...

def transfer_h2d_batch(
    src_arrs: list,
    dst_arrs: list,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None: ...

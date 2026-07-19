/* Localize the engine's embedded abseil + protobuf symbols so they do NOT enter
 * the global dynamic-symbol scope. Otherwise libtpu/sdk.so (built against abseil
 * lts_20250814) binds its protobuf arena to the engine's abseil (lts_20260107) at
 * a mismatched struct layout -> SerialArenaChunkHeader alignment SIGABRT at TPU
 * init. jaxlib's libjax_common.so already exports 0 absl/protobuf; this matches
 * that. xla / pjrt / nanobind / PyInit_* stay exported (global: *), so this does
 * NOT break the engine<->jaxlib integration the way a blanket RTLD_LOCAL does.
 */
{
  global: *;
  local:
    _ZN4absl*;
    _ZNK4absl*;
    _ZNO4absl*;
    _ZNR4absl*;
    _ZZN4absl*;
    _ZTVN4absl*;
    _ZTIN4absl*;
    _ZTSN4absl*;
    _ZTTN4absl*;
    _ZTSZN4absl*;
    _ZN6google8protobuf*;
    _ZNK6google8protobuf*;
    _ZNO6google8protobuf*;
    _ZZN6google8protobuf*;
    _ZTVN6google8protobuf*;
    _ZTIN6google8protobuf*;
    _ZTSN6google8protobuf*;
    _ZTSZN6google8protobuf*;
    _ZTIZN6google8protobuf*;
    _ZGVZN6google8protobuf*;
};

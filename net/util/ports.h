// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NET_UTIL_PORTS_H_
#define NET_UTIL_PORTS_H_

// OSS shim for google3 //net/util:ports. Only PickUnusedPortOrDie() is used by
// the tpu-raiden tests.
namespace net_util {

// Returns an unused TCP port (binds to port 0, reads the kernel-assigned port
// via getsockname, then closes). Aborts on failure. There is an inherent
// TOCTOU race between returning the port and a caller binding it; this matches
// the intended test-only usage of the google3 original.
int PickUnusedPortOrDie();

}  // namespace net_util

#endif  // NET_UTIL_PORTS_H_

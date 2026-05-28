# OSS shim for //third_party/zeromq:cppzmq.
# Vendors zmq.h (from system/conda libzmq 4.3.5) and zmq.hpp (cppzmq v4.10.0)
# under include/, and links against system libzmq.so. The C++ sources include
# the header via the workspace-relative path "third_party/zeromq/include/zmq.hpp",
# so no -I flag is needed.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "cppzmq",
    hdrs = [
        "include/zmq.h",
        "include/zmq.hpp",
    ],
    includes = ["include"],
    linkopts = [
        "-L/mnt/disks/jcgu/anaconda3/lib",
        "-lzmq",
        "-Wl,-rpath,/mnt/disks/jcgu/anaconda3/lib",
    ],
    visibility = ["//visibility:public"],
)

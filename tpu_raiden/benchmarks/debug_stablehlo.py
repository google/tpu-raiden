import os, sys, glob, subprocess, traceback, ctypes

print("=== python ===", sys.version)

def dump_chain(e):
    cur, seen = e, set()
    while cur and id(cur) not in seen:
        seen.add(id(cur))
        print("  CAUSE:", repr(cur))
        cur = cur.__cause__ or cur.__context__

try:
    import jaxlib
    root = os.path.dirname(jaxlib.__file__)
    print("=== jaxlib at:", root)
except BaseException as e:
    print("!!! cannot even import jaxlib package:"); traceback.print_exc(); sys.exit(1)

sos = sorted(set(
    glob.glob(root + "/**/*stablehlo*.so", recursive=True) +
    glob.glob(root + "/**/libMLIRPythonCAPI*", recursive=True) +
    glob.glob(root + "/**/_mlir*.so", recursive=True)
))
print("=== native libs found:", sos)
for so in sos:
    print("=== ldd", so, "===")
    subprocess.run(["ldd", so])
    try:
        ctypes.CDLL(so)                     
        print("    dlopen OK:", so)
    except OSError as e:
        print("    dlopen FAILED:", so, "->", e)  


print("=== try import jaxlib._stablehlo ===")
try:
    import jaxlib._stablehlo
    print("OK: stablehlo import succeeded")
except BaseException as e:
    traceback.print_exc()
    dump_chain(e)
    sys.exit(1)

print("=== try import jax ===")
try:
    import jax
    print("OK: import jax succeeded, backend =", jax.default_backend())
except BaseException as e:
    traceback.print_exc()
    dump_chain(e)
    sys.exit(1)

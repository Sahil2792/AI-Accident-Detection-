"""Quick syntax verification for modified files."""
import py_compile
import sys

files = [
    "routes/main.py",
    "detector/detect.py",
    "detector/video.py",
    "detector/webcam.py",
    "detector/image.py",
]

all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except py_compile.PyCompileError as e:
        print(f"ERROR: {f}: {e}")
        all_ok = False

if all_ok:
    print("ALL FILES COMPILE SUCCESSFULLY")
    sys.exit(0)
else:
    print("SOME FILES FAILED TO COMPILE")
    sys.exit(1)
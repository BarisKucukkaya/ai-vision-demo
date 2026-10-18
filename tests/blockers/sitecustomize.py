# Imported at interpreter start via PYTHONPATH: makes `import cv2` fail in a test worker.
import sys

sys.modules["cv2"] = None

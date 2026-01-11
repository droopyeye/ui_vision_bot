import sys
import numpy, torch, easyocr

try:
    from PyQt6.QtWidgets import QApplication, QLabel
    print("PyQt6 import OK")
except Exception as e:
    print("PyQt6 import FAILED")
    raise

app = QApplication(sys.argv)

labelString = "PyQt6 is working! NumPy: {} Torch: {} EasyOCR OK".format(
    numpy.__version__, torch.__version__)

label = QLabel(labelString)
label.setWindowTitle("PyQt6 Test Window")
label.resize(400, 100)
label.show()

print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("EasyOCR OK")

sys.exit(app.exec())

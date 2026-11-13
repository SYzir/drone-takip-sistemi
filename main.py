import sys
from pathlib import Path

from PyQt5 import QtWidgets

from TracingUI import Ui_MainWindow
from tracking_ui_controller import TrackingUIController


def main():
    project_dir = Path(__file__).resolve().parent
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)

    controller = TrackingUIController(
        main_window=main_window,
        ui=ui,
        model_path=project_dir / "yolo11n.pt",
        default_video_path=project_dir / "droneDeneme.mp4",
    )
    app.aboutToQuit.connect(controller.close)
    main_window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
from ui.main_window import VideoProcessingApp
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoProcessingApp()
    main_window.show()
    
    # Ensure cleanup is called on close
    main_window.closeEvent = main_window.on_close
    
    sys.exit(app.exec_())
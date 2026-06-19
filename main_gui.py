
import sys
import os
import cv2
import time
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer, QRectF, QPropertyAnimation, QEasingCurve, Property, QPoint, QSequentialAnimationGroup, QPauseAnimation, QParallelAnimationGroup, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPainterPath, QRadialGradient, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QProgressBar, QFrame, QGraphicsDropShadowEffect, QSizePolicy,
    QStackedWidget, QGraphicsOpacityEffect, QInputDialog, QMessageBox
)
from camera import Camera
from body_tracker import BodyTracker
from celebration_detector import CelebrationDetector
import database
CELEBRATION_TO_FILE = {
    "MBAPPE_CROSS":    "mbappe.webp",
    "MESSI_SKYPOINT":  "messi.webp",
    "RONALDO_SIUUU":   "ronaldo.jpeg",
    "SON_CAMERA":      "sonnn.jpeg",
    "DIAZ_OPEN":       "diazz.jpg",
    "NEYMAR_EARS":     "naymar.jpg",
}
CELEBRATION_LABELS = {
    "MBAPPE_CROSS":    "Kylian Mbappe",
    "MESSI_SKYPOINT":  "Lionel Messi",
    "RONALDO_SIUUU":   "Cristiano Ronaldo",
    "SON_CAMERA":      "Heung-min Son",
    "DIAZ_OPEN":       "Brahim Diaz",
    "NEYMAR_EARS":     "Neymar Jr",
    "NONE":            "None",
}
CELEBRATION_MOVES = {
    "MBAPPE_CROSS":    "Arms Crossed Pose",
    "MESSI_SKYPOINT":  "Pointing to the Sky",
    "RONALDO_SIUUU":   "SIUUU Power Pose",
    "SON_CAMERA":      "Camera Frame Gesture",
    "DIAZ_OPEN":       "Open Palms 'What?' Pose",
    "NEYMAR_EARS":     "Playful Ears Gesture",
    "NONE":            "No pose detected",
}
CELEBRATION_DESCRIPTIONS = {
    "MBAPPE_CROSS":    "Mbappe's signature cross-armed stance with thumbs up. (Requires full body view)",
    "MESSI_SKYPOINT":  "Messi raising both index fingers up, dedicating goals to his grandmother. (Head + Hands)",
    "RONALDO_SIUUU":   "Ronaldo jumping, spinning, and landing with arms spread wide and low. (Requires full body view)",
    "SON_CAMERA":      "Son framing his face with hands to capture the golden memory. (Head + Hands)",
    "DIAZ_OPEN":       "Diaz spreading his open palms wide in a relaxed, welcoming gesture. (Requires full body view)",
    "NEYMAR_EARS":     "Neymar sticking his tongue out and placing open hands next to his ears. (Head + Hands)",
    "NONE":            "Make one of the player's gestures to trigger their legendary celebration card!",
}
PANINI_CARD_FILES = {
    "MBAPPE_CROSS":    "card_mbappe.jpg",
    "MESSI_SKYPOINT":  "card_messi.jpg",
    "RONALDO_SIUUU":   "card_ronaldo.jpg",
    "SON_CAMERA":      "card_son.png",
    "DIAZ_OPEN":       "card_diaz.jpg",
    "NEYMAR_EARS":     "card_neymar.png",
}
PLAYER_GLOW_COLORS = {
    "MESSI_SKYPOINT":  QColor(255, 30, 30),     
    "RONALDO_SIUUU":   QColor(30, 200, 60),     
    "MBAPPE_CROSS":    QColor(30, 80, 255),     
    "NEYMAR_EARS":     QColor(30, 100, 255),    
    "SON_CAMERA":      QColor(255, 30, 80),     
    "DIAZ_OPEN":       QColor(30, 180, 60),     
}
class CameraWorker(QThread):
    """Worker thread that handles camera frame capture, tracking, and detection."""
    frame_ready = Signal(np.ndarray, str, list, object, object)  
    fps_updated = Signal(float)
    status_msg = Signal(str)
    def __init__(self, source=0, width=640, height=480):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.running = False
        self.camera = None
        self.tracker = None
        self.detector = None
    def set_source(self, source):
        self.source = source
    def run(self):
        self.running = True
        self.status_msg.emit("Initializing camera & AI models...")
        try:
            self.camera = Camera(self.source, self.width, self.height)
            self.tracker = BodyTracker(detection_conf=0.6, tracking_conf=0.6)
            self.detector = CelebrationDetector()
            self.status_msg.emit("System Active - Show your moves!")
        except Exception as e:
            self.status_msg.emit(f"Initialization Error: {e}")
            self.running = False
            return
        last_time = time.time()
        frame_count = 0
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                pose_lm, hand_lm_list, face_lm, annotated_frame = self.tracker.process(frame)
                celebration = self.detector.detect(pose_lm, hand_lm_list)
            except Exception as e:
                annotated_frame = frame.copy()
                celebration = "NONE"
                pose_lm, hand_lm_list, face_lm = None, [], None
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                self.fps_updated.emit(fps)
                frame_count = 0
                last_time = now
            self.frame_ready.emit(annotated_frame, celebration, hand_lm_list, pose_lm, face_lm)
        if self.camera:
            self.camera.release()
        if self.tracker:
            self.tracker.close()
from PySide6.QtWidgets import QWidget
class AnimatedProgressBar(QWidget):
    """Custom animated progress bar featuring a rolling soccer ball and fluid wave dynamics."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimum = 0
        self.maximum = 6
        self.value = 0
        self.target_value = 0
        self.current_value = 0.0  
        self.ball_pixmap = QPixmap("ball.png")
        self.wave_phase = 0.0
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animate_step)
        self.anim_timer.start(30)  
        self.setFixedHeight(50)
    def setMaximum(self, maximum):
        self.maximum = maximum
        self.update()
    def setValue(self, value):
        self.target_value = value
        self.update()
    def animate_step(self):
        self.wave_phase += 0.12
        if self.wave_phase > 2 * np.pi:
            self.wave_phase -= 2 * np.pi
        diff = self.target_value - self.current_value
        if abs(diff) > 0.005:
            self.current_value += diff * 0.08
        else:
            self.current_value = self.target_value
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        ball_size = 38
        track_height = 26
        track_y = (self.height() - track_height) // 2
        track_rect = QRectF(ball_size / 2, track_y, self.width() - ball_size, track_height)
        track_path = QPainterPath()
        track_path.addRoundedRect(track_rect, track_height / 2, track_height / 2)
        painter.fillPath(track_path, QColor(20, 15, 20, 200))
        painter.setPen(QPen(QColor(212, 175, 55, 120), 2))
        painter.drawPath(track_path)
        ratio = self.current_value / max(self.maximum, 1)
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio > 0:
            fill_width = track_rect.width() * ratio
            fill_rect = QRectF(track_rect.left(), track_rect.top(), fill_width, track_rect.height())
            painter.save()
            painter.setClipPath(track_path)
            liquid_grad = QLinearGradient(track_rect.left(), 0, track_rect.left() + fill_width, 0)
            liquid_grad.setColorAt(0.0, QColor(220, 35, 35, 230))   
            liquid_grad.setColorAt(0.5, QColor(230, 120, 15, 230))  
            liquid_grad.setColorAt(1.0, QColor(35, 220, 75, 230))   
            painter.fillRect(fill_rect, QBrush(liquid_grad))
            wave_path = QPainterPath()
            wave_path.moveTo(fill_rect.left(), fill_rect.bottom())
            steps = 40
            for i in range(steps + 1):
                x = fill_rect.left() + (fill_width * i / steps)
                wave_offset = np.sin((i / steps) * 3 * np.pi + self.wave_phase) * 5
                y = fill_rect.top() + (fill_rect.height() / 2) + wave_offset
                wave_path.lineTo(x, y)
            wave_path.lineTo(fill_rect.right(), fill_rect.bottom())
            wave_path.closeSubpath()
            painter.fillPath(wave_path, QBrush(QColor(255, 255, 255, 50)))
            painter.restore()
        ball_x = track_rect.left() + (track_rect.width() * ratio) - (ball_size / 2)
        ball_y = (self.height() - ball_size) // 2
        ball_rect = QRectF(ball_x, ball_y, ball_size, ball_size)
        painter.save()
        painter.translate(ball_rect.center())
        painter.rotate(self.current_value * 180.0)
        clip_path = QPainterPath()
        clip_path.addEllipse(QRectF(-ball_size / 2, -ball_size / 2, ball_size, ball_size))
        painter.setClipPath(clip_path)
        if not self.ball_pixmap.isNull():
            painter.drawPixmap(QRectF(-ball_size / 2, -ball_size / 2, ball_size, ball_size).toRect(), self.ball_pixmap)
        else:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(QRectF(-ball_size / 2, -ball_size / 2, ball_size, ball_size))
        painter.restore()
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Trebuchet MS", 10, QFont.Bold)
        painter.setFont(font)
        text_str = f"UNLOCKED: {self.target_value} / {self.maximum}"
        painter.drawText(self.rect(), Qt.AlignCenter, text_str)
class VictoryScreen(QWidget):
    """Full-screen victory overlay: ball transition, then Panini cards with glowing lights."""
    retry_clicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.setMouseTracking(True)
        self.mouse_pos = QPoint(-1, -1)
        self.ball_phase = 0.0       
        self.cards_visible = False
        self.glow_phase = 0.0       
        self.ball_pixmap = QPixmap("ball.png")
        self.bg_pixmap = QPixmap("background.png")
        self.trophy_pixmap = QPixmap("trophy.png")
        self.card_order = [
            "MESSI_SKYPOINT", "RONALDO_SIUUU", "NEYMAR_EARS", 
            "SON_CAMERA", "DIAZ_OPEN", "MBAPPE_CROSS"
        ]
        self.card_pixmaps = {}
        for code in self.card_order:
            path = PANINI_CARD_FILES.get(code, "")
            pix = QPixmap(path)
            if pix.isNull():
                pix = QPixmap(CELEBRATION_TO_FILE.get(code, ""))
            self.card_pixmaps[code] = pix
        self.card_opacities = [0.0] * 6
        self.card_y_offsets = [80.0] * 6  
        self.card_scales = [1.0] * 6
        self.card_glow_alphas = [0.0] * 6
        self.card_rects = [QRectF()] * 6
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(16)  
        self.ball_speed = 0.008
        self.card_reveal_started = False
        self.card_reveal_index = 0
        self.card_reveal_timer = QTimer(self)
        self.card_reveal_timer.timeout.connect(self._reveal_next_card)
        self.btn_retry = QPushButton("Try Again", self)
        self.btn_retry.setObjectName("roundButton")
        self.btn_retry.setStyleSheet("""
            QPushButton#roundButton {
                border-radius: 20px;
                padding: 10px 30px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(220, 35, 35, 230), stop:0.5 rgba(230, 120, 15, 230), stop:1 rgba(35, 220, 75, 230));
                border: 2px solid #000000;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton#roundButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(240, 55, 55, 230), stop:0.5 rgba(250, 140, 35, 230), stop:1 rgba(55, 240, 95, 230));
                border-color: #FFDF00;
            }
        """)
        self.btn_retry.clicked.connect(self.retry_clicked.emit)
        self.btn_retry.hide()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        btn_w, btn_h = 160, 45
        self.btn_retry.setGeometry(self.width() // 2 - btn_w // 2, self.height() - 65, btn_w, btn_h)
    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
    def leaveEvent(self, event):
        self.mouse_pos = QPoint(-1, -1)
    def start_animation(self):
        """Start the ball-roll → card-reveal sequence."""
        self.ball_phase = 0.0
        self.cards_visible = False
        self.card_reveal_started = False
        self.card_reveal_index = 0
        self.card_opacities = [0.0] * 6
        self.card_y_offsets = [80.0] * 6
        self.card_scales = [1.0] * 6
        self.card_glow_alphas = [0.0] * 6
        self.btn_retry.hide()
        self.show()
        self.raise_()
        self.update()
    def _animate(self):
        """Called every frame (~60 FPS)."""
        if self.ball_phase < 1.0:
            self.ball_phase += self.ball_speed
            if self.ball_phase >= 1.0:
                self.ball_phase = 1.0
                QTimer.singleShot(400, self._start_card_reveal)
        self.glow_phase += 0.02
        if self.glow_phase > 6.2832:  
            self.glow_phase -= 6.2832
        for i in range(6):
            target_opacity = 1.0 if (self.card_reveal_started and i < self.card_reveal_index) else 0.0
            self.card_opacities[i] += (target_opacity - self.card_opacities[i]) * 0.08
            target_y = 0.0 if (self.card_reveal_started and i < self.card_reveal_index) else 80.0
            self.card_y_offsets[i] += (target_y - self.card_y_offsets[i]) * 0.08
            is_hovered = self.cards_visible and target_opacity > 0.5 and self.card_rects[i].contains(self.mouse_pos)
            target_scale = 1.08 if is_hovered else 1.0
            self.card_scales[i] += (target_scale - self.card_scales[i]) * 0.15
            target_glow = 1.0 if is_hovered else 0.0
            self.card_glow_alphas[i] += (target_glow - self.card_glow_alphas[i]) * 0.15
        self.update()
    def _start_card_reveal(self):
        self.card_reveal_started = True
        self.cards_visible = True
        self.card_reveal_timer.start(350)  
        self._reveal_next_card()
    def _reveal_next_card(self):
        if self.card_reveal_index < 6:
            self.card_reveal_index += 1
        else:
            self.card_reveal_timer.stop()
            self.btn_retry.show()
            self.btn_retry.raise_()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        if not self.bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.bg_pixmap)
        else:
            painter.fillRect(self.rect(), QColor(5, 5, 15))
        painter.fillRect(self.rect(), QColor(0, 0, 0, 200))
        g_top = QLinearGradient(0, 0, 0, 120)
        g_top.setColorAt(0.0, QColor(0, 0, 0, 240))
        g_top.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, w, 120, g_top)
        g_bot = QLinearGradient(0, h - 120, 0, h)
        g_bot.setColorAt(0.0, QColor(0, 0, 0, 0))
        g_bot.setColorAt(1.0, QColor(0, 0, 0, 240))
        painter.fillRect(0, h - 120, w, 120, g_bot)
        if self.ball_phase < 1.0:
            ball_size = 80
            ball_x = int(-ball_size + (w + ball_size) * self.ball_phase)
            ball_y = int(h / 2 - ball_size / 2)
            spin = self.ball_phase * 720
            painter.save()
            painter.translate(ball_x + ball_size / 2, ball_y + ball_size / 2)
            painter.rotate(spin)
            if not self.ball_pixmap.isNull():
                scaled = self.ball_pixmap.scaled(ball_size, ball_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(-ball_size // 2, -ball_size // 2, scaled)
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(QRectF(-ball_size / 2, -ball_size / 2, ball_size, ball_size))
            painter.restore()
        if self.cards_visible:
            title_rect = QRectF(0, 20, w, 60)
            trophy_w = 40
            if not self.trophy_pixmap.isNull():
                t_scaled = self.trophy_pixmap.scaled(trophy_w, trophy_w, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.setPen(QColor(255, 255, 255))
                font = QFont("Impact", 32, QFont.Bold)
                painter.setFont(font)
                fm = painter.fontMetrics()
                text_w = fm.horizontalAdvance("YOUR COLLECTION")
                tx1 = int(w / 2 - text_w / 2 - trophy_w - 20)
                tx2 = int(w / 2 + text_w / 2 + 20)
                ty = 30
                painter.drawPixmap(tx1, ty, t_scaled)
                painter.drawPixmap(tx2, ty, t_scaled)
                painter.drawText(title_rect, Qt.AlignCenter, "YOUR COLLECTION")
            else:
                painter.setPen(QColor(255, 255, 255))
                font = QFont("Impact", 32, QFont.Bold)
                painter.setFont(font)
                painter.drawText(title_rect, Qt.AlignCenter, "🏆 YOUR COLLECTION 🏆")
            card_w, card_h = 160, 220
            margin_x = 40
            margin_y = 30
            total_grid_w = 3 * card_w + 2 * margin_x
            total_grid_h = 2 * card_h + margin_y
            start_x = (w - total_grid_w) // 2
            start_y = (h - total_grid_h) // 2 - 20
            for i, code in enumerate(self.card_order):
                row = i // 3
                col = i % 3
                cx = start_x + col * (card_w + margin_x)
                cy = start_y + row * (card_h + margin_y) + int(self.card_y_offsets[i])
                opacity = self.card_opacities[i]
                if opacity < 0.01:
                    continue
                self.card_rects[i] = QRectF(cx, cy, card_w, card_h)
                painter.save()
                painter.setOpacity(opacity)
                scale = self.card_scales[i]
                glow_alpha = self.card_glow_alphas[i]
                center_x = cx + card_w / 2
                center_y = cy + card_h / 2
                if glow_alpha > 0.01:
                    base_color = PLAYER_GLOW_COLORS.get(code, QColor(255, 255, 255))
                    glow_color = QColor(base_color)
                    glow_color.setAlpha(int(255 * glow_alpha))
                    glow_grad = QRadialGradient(center_x, center_y, card_w * 0.9)
                    glow_grad.setColorAt(0.0, glow_color)
                    glow_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
                    painter.fillRect(int(center_x - card_w*1.5), int(center_y - card_h*1.5), int(card_w*3), int(card_h*3), QBrush(glow_grad))
                painter.translate(center_x, center_y)
                painter.scale(scale, scale)
                painter.translate(-center_x, -center_y)
                pix = self.card_pixmaps.get(code)
                if pix and not pix.isNull():
                    scaled = pix.scaled(card_w, card_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    px = cx + (card_w - scaled.width()) // 2
                    py = cy + (card_h - scaled.height()) // 2
                    painter.drawPixmap(int(px), int(py), scaled)
                else:
                    painter.fillRect(int(cx), int(cy), card_w, card_h, QColor(40, 40, 40))
                    painter.setPen(QColor(255, 255, 255))
                    f = QFont("Impact", 12)
                    painter.setFont(f)
                    player_name = CELEBRATION_LABELS.get(code, "?")
                    painter.drawText(QRectF(cx, cy, card_w, card_h), Qt.AlignCenter, player_name)
                painter.setPen(QColor(255, 255, 255))
                name_font = QFont("Impact", 11, QFont.Bold)
                painter.setFont(name_font)
                name_text = CELEBRATION_LABELS.get(code, "")
                painter.drawText(QRectF(cx, cy + card_h + 5, card_w, 24), Qt.AlignCenter, name_text)
                painter.restore()
        painter.end()
class WorldCupCelebrationApp(QMainWindow):
    """Main window representing the elegant World Cup themed GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏆 FIFA WORLD CUP CELEBRATION DETECTOR")
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        self.current_celebration = "NONE"
        self.last_trigger_time = 0.0
        self.cooldown_duration = 2.0  
        self.unlocked_celebrations = set()
        self.current_user_id = None
        self.camera_source = 0
        self.worker = None
        self.init_ui()
        self.victory_screen = VictoryScreen(self)
        self.victory_screen.hide()
        self.victory_screen.retry_clicked.connect(self.reset_game)
        self.victory_triggered = False
        
        # Load initial users
        self.load_users()
    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
            }
            QWidget#centralWidget {
                background: transparent;
            }
            QWidget {
                color: #FFFFFF;
                font-family: 'Impact', 'Arial Black', sans-serif;
            }
            QFrame#cardFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgba(220, 35, 35, 230), 
                                            stop:0.5 rgba(230, 120, 15, 230), 
                                            stop:1 rgba(35, 220, 75, 230));
                border: 2px solid #000000;
                border-radius: 14px;
            }
            QLabel#headerTitle {
                color: #000000;
                font-family: 'Impact', 'Arial Black', sans-serif;
                font-size: 46px;
                font-weight: 900;
                letter-spacing: 2px;
            }
            QLabel#sectionLabel {
                color: #D4AF37;
                font-family: 'Impact', 'Arial Black', sans-serif;
                font-size: 14px;
                font-weight: 900;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #4A0E17, stop:1 #32080E);
                border: 1px solid #D4AF37;
                border-radius: 6px;
                color: #FFFFFF;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton#roundButton {
                border-radius: 18px;
                padding: 8px 20px;
                border: 2px solid #D4AF37;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #5C121D, stop:1 #3E0B11);
                border-color: #FFDF00;
            }
            QPushButton:pressed {
                background: #25050A;
            }
            QComboBox {
                background-color: #2D2D2D;
                border: 1px solid rgba(212, 175, 55, 0.4);
                border-radius: 6px;
                padding: 6px;
                min-width: 100px;
                color: #FFFFFF;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QLineEdit {
                background-color: #2D2D2D;
                border: 1px solid rgba(212, 175, 55, 0.4);
                border-radius: 6px;
                padding: 6px;
                color: #FFFFFF;
            }
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(212, 175, 55, 0.3);
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #B38F24, stop:1 #F5D061);
                border-radius: 9px;
            }
        """)
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        logo_lbl = QLabel()
        logo_pixmap = QPixmap("logo.png")
        if not logo_pixmap.isNull():
            logo_lbl.setPixmap(logo_pixmap.scaledToHeight(130, Qt.SmoothTransformation))
        else:
            logo_lbl.setText("🏆")
            logo_lbl.setStyleSheet("font-size: 28px;")
        header_layout.addWidget(logo_lbl)
        header_layout.setAlignment(Qt.AlignVCenter)
        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        title_container.setAlignment(Qt.AlignVCenter)
        title_lbl = QLabel("WORLD CUP CELEBRATI<img src='ball.png' width='40' height='40'>N")
        title_lbl.setTextFormat(Qt.RichText)
        title_lbl.setObjectName("headerTitle")
        sub_title_lbl = QLabel("Show your moves")
        sub_title_lbl.setStyleSheet("color: #000000; font-size: 18px; font-weight: bold; letter-spacing: 1px;")
        title_container.addWidget(title_lbl)
        title_container.addWidget(sub_title_lbl)
        header_layout.addLayout(title_container)
        header_layout.addStretch()
        
        # --- USER SELECTION UI ---
        user_container = QVBoxLayout()
        user_container.setSpacing(2)
        
        user_row = QHBoxLayout()
        user_lbl = QLabel("Player:")
        user_lbl.setStyleSheet("color: #000000; font-weight: bold; font-size: 14px;")
        self.combo_users = QComboBox()
        self.combo_users.setStyleSheet("background-color: #2D2D2D; color: #FFFFFF; border: 1px solid #D4AF37; font-size: 12px;")
        self.combo_users.currentIndexChanged.connect(self.on_user_changed)
        
        self.btn_add_user = QPushButton("+ New")
        self.btn_add_user.setStyleSheet("""
            QPushButton { background-color: #2D2D2D; color: #D4AF37; border: 1px solid #D4AF37; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
            QPushButton:hover { background-color: #D4AF37; color: #000000; }
        """)
        self.btn_add_user.clicked.connect(self.add_new_user)
        
        user_row.addWidget(user_lbl)
        user_row.addWidget(self.combo_users)
        user_row.addWidget(self.btn_add_user)
        user_container.addLayout(user_row)
        header_layout.addLayout(user_container)
        
        header_layout.addSpacing(20)
        # --- END USER SELECTION UI ---

        self.status_badge = QLabel("Status: Idle")
        self.status_badge.setStyleSheet("""
            background-color: rgba(212, 175, 55, 0.15);
            color: #D4AF37;
            border: 1px solid #D4AF37;
            border-radius: 15px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: bold;
        """)
        header_layout.addWidget(self.status_badge)
        main_layout.addWidget(header_widget)
        split_layout = QHBoxLayout()
        split_layout.setSpacing(15)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        cam_card = QFrame()
        cam_card.setObjectName("cardFrame")
        cam_card_layout = QVBoxLayout(cam_card)
        cam_card_layout.setContentsMargins(10, 10, 10, 10)
        cam_header = QHBoxLayout()
        cam_title = QLabel("LIVE CAMERA")
        cam_title.setObjectName("sectionLabel")
        cam_header.addWidget(cam_title)
        cam_header.addStretch()
        self.fps_lbl = QLabel("FPS: --")
        self.fps_lbl.setStyleSheet("color: #00FF88; font-weight: bold; font-size: 11px;")
        cam_header.addWidget(self.fps_lbl)
        cam_card_layout.addLayout(cam_header)
        self.video_lbl = QLabel("No active video feed")
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("""
            background-color: #08080C;
            border: 2px solid rgba(255,255,255,0.05);
            border-radius: 10px;
            font-size: 16px;
            color: rgba(255,255,255,0.4);
        """)
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_lbl.setMinimumSize(640, 480)
        cam_card_layout.addWidget(self.video_lbl)
        control_panel = QHBoxLayout()
        control_panel.setSpacing(8)
        self.btn_camera = QPushButton(" Start Camera")
        self.btn_camera.setObjectName("roundButton")
        self.btn_camera.setIcon(QIcon("icon_camera.png"))
        self.btn_camera.setIconSize(QSize(18, 18))
        self.btn_camera.clicked.connect(self.toggle_camera)
        control_panel.addWidget(self.btn_camera)
        self.btn_screenshot = QPushButton(" Snapshot")
        self.btn_screenshot.setObjectName("roundButton")
        self.btn_screenshot.setIcon(QIcon("icon_snapshot.png"))
        self.btn_screenshot.setIconSize(QSize(18, 18))
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        control_panel.addWidget(self.btn_screenshot)
        control_panel.addStretch()
        lbl_source = QLabel("Source:")
        lbl_source.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 12px;")
        control_panel.addWidget(lbl_source)
        self.combo_source = QComboBox()
        self.combo_source.addItems(["Camera 0", "Camera 1", "Camera 2", "Custom IP / URL"])
        self.combo_source.currentIndexChanged.connect(self.source_changed)
        control_panel.addWidget(self.combo_source)
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("http://192.168.1.X:4747/video")
        self.txt_url.setHidden(True)
        self.txt_url.textChanged.connect(self.url_changed)
        control_panel.addWidget(self.txt_url)
        cam_card_layout.addLayout(control_panel)
        left_panel.addWidget(cam_card)
        metrics_card = QFrame()
        metrics_card.setObjectName("cardFrame")
        metrics_layout = QGridLayout(metrics_card)
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_title = QLabel("TRACKING METRICS")
        metrics_title.setObjectName("sectionLabel")
        metrics_layout.addWidget(metrics_title, 0, 0, 1, 4)
        self.metric_pose = QLabel("Pose landmarks: --")
        self.metric_pose.setStyleSheet("color: rgba(255,255,255,0.85); font-size: 12px;")
        metrics_layout.addWidget(self.metric_pose, 1, 0)
        self.metric_hands = QLabel("Hands detected: 0")
        self.metric_hands.setStyleSheet("color: rgba(255,255,255,0.85); font-size: 12px;")
        metrics_layout.addWidget(self.metric_hands, 1, 1)
        self.metric_face = QLabel("Face Mesh: --")
        self.metric_face.setStyleSheet("color: rgba(255,255,255,0.85); font-size: 12px;")
        metrics_layout.addWidget(self.metric_face, 1, 2)
        self.metric_body_status = QLabel("Body visible: No")
        self.metric_body_status.setStyleSheet("color: #FF5555; font-weight: bold; font-size: 12px;")
        metrics_layout.addWidget(self.metric_body_status, 1, 3)
        left_panel.addWidget(metrics_card)
        split_layout.addLayout(left_panel, 2)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        showcase_card = QFrame()
        showcase_card.setObjectName("cardFrame")
        showcase_layout = QVBoxLayout(showcase_card)
        showcase_layout.setContentsMargins(15, 15, 15, 15)
        showcase_layout.setSpacing(10)
        self.showcase_title = QLabel("WORLD CUP CELEBRATION CARD")
        self.showcase_title.setObjectName("sectionLabel")
        showcase_layout.addWidget(self.showcase_title)
        self.showcase_img = QLabel("No active celebration")
        self.showcase_img.setAlignment(Qt.AlignCenter)
        self.showcase_img.setFixedSize(300, 300)
        self.showcase_img.setStyleSheet("""
            background-color: #08080C;
            border: 2px solid rgba(212, 175, 55, 0.4);
            border-radius: 12px;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.4);
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(212, 175, 55, 80))
        shadow.setOffset(0, 0)
        self.showcase_img.setGraphicsEffect(shadow)
        showcase_layout.addWidget(self.showcase_img, 0, Qt.AlignCenter)
        right_panel.addWidget(showcase_card)
        quest_card = QFrame()
        quest_card.setObjectName("cardFrame")
        quest_layout = QVBoxLayout(quest_card)
        quest_layout.setContentsMargins(15, 12, 15, 12)
        quest_layout.setSpacing(8)
        quest_title = QLabel("CELEBRATION UNLOCK QUEST")
        quest_title.setObjectName("sectionLabel")
        quest_layout.addWidget(quest_title)
        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setMaximum(6)
        self.progress_bar.setValue(0)
        quest_layout.addWidget(self.progress_bar)
        self.player_checkmarks = {}
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)
        players_list = [
            ("MBAPPE_CROSS", "Mbappe"),
            ("MESSI_SKYPOINT", "Messi"),
            ("RONALDO_SIUUU", "Ronaldo"),
            ("SON_CAMERA", "Son"),
            ("DIAZ_OPEN", "Diaz"),
            ("NEYMAR_EARS", "Neymar"),
        ]
        for idx, (code, name) in enumerate(players_list):
            row = idx // 2
            col = idx % 2
            item = QFrame()
            item.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
            """)
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(8, 6, 8, 6)
            indicator = QWidget()
            indicator.setFixedSize(12, 12)
            indicator.setStyleSheet("background-color: #404040; border-radius: 6px;")
            item_layout.addWidget(indicator)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: rgba(255,255,255,0.7);")
            item_layout.addWidget(name_lbl)
            item_layout.addStretch()
            grid_layout.addWidget(item, row, col)
            self.player_checkmarks[code] = (indicator, name_lbl, item)
        quest_layout.addWidget(grid_widget)
        right_panel.addWidget(quest_card)
        split_layout.addLayout(right_panel, 1)
        main_layout.addStretch(1)
        main_layout.addLayout(split_layout)
        main_layout.addStretch(1)
        self.statusBar_lbl = QLabel("System Initializing...")
        self.statusBar_lbl.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 11px;")
        self.statusBar_lbl.setHidden(True)
        main_layout.addWidget(self.statusBar_lbl)
    @Slot(int)
    def source_changed(self, idx):
        if idx == 3:  
            self.txt_url.setHidden(False)
            self.camera_source = self.txt_url.text()
        else:
            self.txt_url.setHidden(True)
            self.camera_source = idx
    @Slot(str)
    def url_changed(self, text):
        if self.combo_source.currentIndex() == 3:
            self.camera_source = text
    def toggle_camera(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.running = False
            self.worker.quit()
            self.worker.wait()
            self.worker = None
            self.btn_camera.setText(" Start Camera")
            self.video_lbl.setText("Camera paused")
            self.status_badge.setText("Status: Paused")
            self.fps_lbl.setText("FPS: --")
        else:
            source = self.camera_source
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            self.worker = CameraWorker(source=source)
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.fps_updated.connect(self.update_fps)
            self.worker.status_msg.connect(self.update_status)
            self.worker.start()
            self.btn_camera.setText(" Pause Camera")
            self.status_badge.setText("Status: Active")
    def update_status(self, text):
        self.statusBar_lbl.setText(text)
        if "System Active" in text or "Show your moves" in text:
            self.status_badge.setText("Status: Active")
            self.status_badge.setStyleSheet("""
                background-color: rgba(0, 255, 136, 0.15);
                color: #00FF88;
                border: 1px solid #00FF88;
                border-radius: 12px;
                padding: 3px 10px;
                font-size: 10px;
                font-weight: bold;
            """)
        elif "Error" in text:
            self.status_badge.setText("Status: Error")
            self.status_badge.setStyleSheet("""
                background-color: rgba(255, 85, 85, 0.15);
                color: #FF5555;
                border: 1px solid #FF5555;
                border-radius: 12px;
                padding: 3px 10px;
                font-size: 10px;
                font-weight: bold;
            """)
    def update_fps(self, fps):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")
    @Slot(np.ndarray, str, list, object, object)
    def update_frame(self, frame, celebration, hand_lms, pose_lms, face_lms):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_lbl.setPixmap(scaled_pixmap)
        self.metric_pose.setText(f"Pose landmarks: {len(pose_lms) if pose_lms else '--'}")
        self.metric_hands.setText(f"Hands detected: {len(hand_lms)}")
        self.metric_face.setText(f"Face Mesh: {'Yes' if face_lms else '--'}")
        if pose_lms and len(pose_lms) >= 25:
            left_hip = pose_lms[23]
            right_hip = pose_lms[24]
            if left_hip[3] > 0.5 and right_hip[3] > 0.5:
                self.metric_body_status.setText("Body visible: Yes")
                self.metric_body_status.setStyleSheet("color: #00FF88; font-weight: bold; font-size: 12px;")
            else:
                self.metric_body_status.setText("Body visible: Partially")
                self.metric_body_status.setStyleSheet("color: #FFAA00; font-weight: bold; font-size: 12px;")
        else:
            self.metric_body_status.setText("Body visible: No")
            self.metric_body_status.setStyleSheet("color: #FF5555; font-weight: bold; font-size: 12px;")
        now = time.time()
        if celebration != "NONE":
            self.current_celebration = celebration
            self.last_trigger_time = now
            if celebration not in self.unlocked_celebrations:
                self.unlocked_celebrations.add(celebration)
                if self.current_user_id is not None:
                    database.save_progress(self.current_user_id, celebration)
                self.unlock_celebration_ui(celebration)
        if now - self.last_trigger_time > self.cooldown_duration:
            self.current_celebration = "NONE"
        self.update_showcase_panel(self.current_celebration)
    def unlock_celebration_ui(self, code):
        """Update checklist indicators when a user unlocks a celebration."""
        if code in self.player_checkmarks:
            indicator, label, frame = self.player_checkmarks[code]
            indicator.setStyleSheet("background-color: #D4AF37; border-radius: 6px;")
            label.setStyleSheet("font-size: 11px; font-weight: bold; color: #FFFFFF;")
            frame.setStyleSheet("""
                background-color: rgba(212, 175, 55, 0.1);
                border: 1px solid rgba(212, 175, 55, 0.5);
                border-radius: 6px;
            """)
            self.progress_bar.setValue(len(self.unlocked_celebrations))
            if len(self.unlocked_celebrations) >= 6 and not self.victory_triggered:
                self.victory_triggered = True
                QTimer.singleShot(2000, self._launch_victory_screen)
    def update_showcase_panel(self, code):
        """Update the player showcase card dynamically."""
        if code == "NONE":
            self.showcase_title.setText("WORLD CUP CELEBRATION CARD")
            self.showcase_img.clear()
            self.showcase_img.setText("")
            self.showcase_img.setStyleSheet("""
                background-color: #08080C;
                border: 2px solid rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                font-size: 14px;
                color: rgba(255, 255, 255, 0.4);
            """)
        else:
            player_name = CELEBRATION_LABELS.get(code, "Unknown Player")
            move_name = CELEBRATION_MOVES.get(code, "")
            self.showcase_title.setText(f"{player_name.upper()} - {move_name.upper()}")
            image_file = CELEBRATION_TO_FILE.get(code, "")
            if image_file and os.path.exists(image_file):
                pixmap = QPixmap(image_file)
                scaled_pix = pixmap.scaled(self.showcase_img.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                crop_x = (scaled_pix.width() - self.showcase_img.width()) // 2
                crop_y = (scaled_pix.height() - self.showcase_img.height()) // 2
                cropped_pix = scaled_pix.copy(crop_x, crop_y, self.showcase_img.width(), self.showcase_img.height())
                self.showcase_img.setPixmap(cropped_pix)
                self.showcase_img.setStyleSheet("""
                    background-color: transparent;
                    border: 2px solid #D4AF37;
                    border-radius: 12px;
                """)
            else:
                self.showcase_img.clear()
                self.showcase_img.setText(f"[Image missing: {image_file}]")
                self.showcase_img.setStyleSheet("""
                    background-color: #08080C;
                    border: 2px solid #FF5555;
                    border-radius: 12px;
                    color: #FF5555;
                """)
    def take_screenshot(self):
        """Saves a screenshot and flashes the screen as visual feedback."""
        if self.worker is None or not self.worker.running:
            return
        filename = f"snapshot_{int(time.time())}.png"
        pixmap = self.video_lbl.pixmap()
        if pixmap and not pixmap.isNull():
            pixmap.save(filename)
            self.statusBar_lbl.setText(f"📸 Snapshot saved to: {filename}")
            self.flash_overlay()
    def flash_overlay(self):
        original_style = self.video_lbl.styleSheet()
        self.video_lbl.setStyleSheet("""
            background-color: #FFFFFF;
            border: 5px solid #00FFFF;
            border-radius: 10px;
        """)
        QTimer.singleShot(150, lambda: self.video_lbl.setStyleSheet(original_style))
    def reset_game(self):
        """Resets the state to allow the user to try again."""
        self.victory_screen.hide()
        self.victory_triggered = False
        self.unlocked_celebrations.clear()
        if self.current_user_id is not None:
            database.reset_progress(self.current_user_id)
        self.progress_bar.setValue(0)
        for code, (indicator, label, frame) in self.player_checkmarks.items():
            indicator.setStyleSheet("background-color: #404040; border-radius: 6px;")
            label.setStyleSheet("font-size: 11px; font-weight: bold; color: rgba(255,255,255,0.7);")
            frame.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
            """)
        self.update_showcase_panel("NONE")
        if self.worker is None:
            self.toggle_camera()
    def _launch_victory_screen(self):
        """Stop the camera and show the animated victory screen."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.running = False
            self.worker.quit()
            self.worker.wait()
            self.worker = None
            self.btn_camera.setText(" Start Camera")
        self.victory_screen.setGeometry(self.centralWidget().geometry())
        self.victory_screen.start_animation()
    def resizeEvent(self, event):
        """Keep the victory screen overlay sized to fill the window."""
        super().resizeEvent(event)
        if hasattr(self, 'victory_screen') and self.victory_screen.isVisible():
            self.victory_screen.setGeometry(self.centralWidget().geometry())

    def load_users(self):
        self.combo_users.blockSignals(True)
        self.combo_users.clear()
        users = database.get_users()
        for u in users:
            self.combo_users.addItem(u["username"], u["id"])
        self.combo_users.blockSignals(False)
        if users:
            self.current_user_id = users[0]["id"]
            self.load_user_progress()
            
    def on_user_changed(self, index):
        if index >= 0:
            self.current_user_id = self.combo_users.currentData()
            self.load_user_progress()
            
    def load_user_progress(self):
        if self.current_user_id is None:
            return
        
        self.unlocked_celebrations.clear()
        self.progress_bar.setValue(0)
        self.victory_triggered = False
        
        # Reset visual indicators first
        for code, (indicator, label, frame) in self.player_checkmarks.items():
            indicator.setStyleSheet("background-color: #404040; border-radius: 6px;")
            label.setStyleSheet("font-size: 11px; font-weight: bold; color: rgba(255,255,255,0.7);")
            frame.setStyleSheet("background-color: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px;")
            
        unlocked = database.get_user_progress(self.current_user_id)
        for code in unlocked:
            self.unlocked_celebrations.add(code)
            self.unlock_celebration_ui(code)
            
    def add_new_user(self):
        name, ok = QInputDialog.getText(self, "New Player", "Enter player name:")
        if ok and name.strip():
            uid = database.add_user(name.strip())
            if uid is None:
                QMessageBox.warning(self, "Error", "Player already exists!")
            else:
                self.load_users()
                # Select the new user
                index = self.combo_users.findData(uid)
                if index >= 0:
                    self.combo_users.setCurrentIndex(index)
    def closeEvent(self, event):
        """Make sure threads cleanly shutdown when GUI window is closed."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.running = False
            self.worker.quit()
            if not self.worker.wait(3000):  # Wait max 3 seconds
                self.worker.terminate()  # Force terminate if it doesn't exit
                self.worker.wait()
        event.accept()
    def paintEvent(self, event):
        """Paint background.png stretched across the window."""
        painter = QPainter(self)
        bg_pixmap = QPixmap("background.png")
        if not bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), bg_pixmap)
        else:
            painter.fillRect(self.rect(), QColor("#120005"))
        gradient_top = QLinearGradient(0, 0, 0, 150)
        gradient_top.setColorAt(0.0, QColor(0, 0, 0, 200))
        gradient_top.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, self.width(), 150, gradient_top)
        gradient_bottom = QLinearGradient(0, self.height() - 150, 0, self.height())
        gradient_bottom.setColorAt(0.0, QColor(0, 0, 0, 0))
        gradient_bottom.setColorAt(1.0, QColor(0, 0, 0, 200))
        painter.fillRect(0, self.height() - 150, self.width(), 150, gradient_bottom)
def main():
    app = QApplication(sys.argv)
    window = WorldCupCelebrationApp()
    window.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
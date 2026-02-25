"""
Project Name / 專案名稱：Image Perspective Correction Tool (圖片梯形矯正與透視處理工具)
File Name / 檔案名稱：image_fix_09.py
Author / 作者：毛恩 MaoEn (Ben)
License / 授權條款：MIT License (See project README for details / 詳見專案 README)

Declarations / 聲明：
1. Parts of the code in this project were written and optimized with the assistance of AI (Claude / Gemini).
   本專案程式碼部分由 AI 輔助編寫與優化 (Claude/ Gemini)。
2. You are welcome to freely share, modify, and use this code, but please retain the original author's attribution.
   歡迎自由分享、修改與使用本代碼，但請保留原作者署名。
3. My favorite V-Tuber, please remember to subscribe to her: Umi (白星優米) https://youtube.com/@umitw46?si=YAS3N6vEymUu1UDW
   最喜歡的V-Tuber，記得訂閱她: 白星優米 https://youtube.com/@umitw46?si=YAS3N6vEymUu1UDW
"""


import sys
import os
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF（已隨 exe 打包，直接匯入）

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QPainterPath
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsItem,
    QFileDialog, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QProgressBar, QMessageBox,
    QGroupBox, QGridLayout
)


# ==================== 影像處理模組 ====================
# DocumentScanner：負責影像讀取、角點偵測與透視矯正

class DocumentScanner:
    """文件影像處理器：提供自動角點偵測、透視矯正等核心功能。"""

    def __init__(self, max_height: int = 800):
        # max_height: 偵測用縮放上限，避免大圖造成運算過慢
        self.max_height = max_height

    def load_image(self, path: str, page_idx: int = None) -> np.ndarray:
        """
        讀取圖片或 PDF 頁面。
        :param path: 檔案路徑
        :param page_idx: PDF 頁碼（0-based），None 表示讀取一般圖片
        :return: BGR 格式的 numpy 陣列
        """
        if page_idx is not None:
            # 讀取 PDF 指定頁面
            doc = fitz.open(path)
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            # 根據色彩通道數決定轉換方式（RGBA=4 通道，RGB=3 通道）
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
            doc.close()  # [優化] 確保 PDF 文件資源被正確釋放
            return img
        else:
            # 讀取一般圖片
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"無法讀取影像：{path}")
            return img

    def _resize_for_detection(self, image: np.ndarray):
        """
        [內部] 將圖片等比例縮小到 max_height 以內以加速偵測。
        :return: (縮小後圖片, 縮放比例)
        """
        h, w = image.shape[:2]
        if h <= self.max_height:
            return image.copy(), 1.0
        ratio = self.max_height / float(h)
        resized = cv2.resize(image, (int(w * ratio), self.max_height))
        return resized, ratio

    def auto_detect_corners(self, image: np.ndarray):
        """
        自動偵測文件的四個角點。
        流程：縮圖 → 灰階 → 高斯模糊 → Canny 邊緣 → 輪廓偵測 → 找四邊形
        :return: 有序的 4x2 角點陣列 (float32)，偵測失敗回傳 None
        """
        resized, ratio = self._resize_for_detection(image)
        total_area = resized.shape[0] * resized.shape[1]

        # 邊緣偵測前處理
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # 找出所有輪廓並依面積由大到小排序
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area = cv2.contourArea(c)
            # 排除面積過小的輪廓（小於圖片總面積 10%）
            if area < (total_area * 0.1):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 找到四邊形即視為文件邊界
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                pts /= ratio  # 座標還原至原始圖片比例
                return self._order_points(pts)
        return None

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        [內部] 將四個角點依序排列為：左上、右上、右下、左下。
        利用座標和/差的特性快速排序。
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]    # 左上（x+y 最小）
        rect[2] = pts[np.argmax(s)]    # 右下（x+y 最大）
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # 右上（y-x 最小）
        rect[3] = pts[np.argmax(diff)] # 左下（y-x 最大）
        return rect

    def default_inset_corners(self, image: np.ndarray, inset_ratio: float = 0.05) -> np.ndarray:
        """
        當自動偵測失敗時，產生預設內縮角點（距邊緣 inset_ratio 比例處）。
        :return: 有序的 4x2 角點陣列 (float32)
        """
        h, w = image.shape[:2]
        dx, dy = w * inset_ratio, h * inset_ratio
        return np.array([
            (dx,     dy),       # 左上
            (w - dx, dy),       # 右上
            (w - dx, h - dy),   # 右下
            (dx,     h - dy),   # 左下
        ], dtype="float32")

    def warp_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        依據四個角點進行透視矯正，輸出平整的矩形文件影像。
        :param corners: 有序的 4x2 角點陣列（左上、右上、右下、左下）
        :return: 透視矯正後的影像
        """
        rect = np.array(corners, dtype="float32")
        tl, tr, br, bl = rect

        # 計算輸出尺寸（取兩組對邊長度的最大值，保留最多細節）
        maxWidth  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([
            [0,            0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0,            maxHeight - 1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# ==================== UI 元件：角點拖曳控制點 ====================

class CornerHandle(QGraphicsEllipseItem):
    """
    可拖曳的圓形控制點，代表文件的一個角點。
    idx: 角點索引（0=左上, 1=右上, 2=右下, 3=左下）
    """

    def __init__(self, idx: int, radius: int = 25, parent=None):
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setBrush(Qt.red)
        self.setPen(QPen(Qt.white, 3))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(2)  # 確保控制點顯示於圖片與框線之上
        self.idx = idx
        self._view = None  # 父 ImageView 的參考，由外部在建立時設定

    def itemChange(self, change, value):
        """拖曳時將控制點限制在圖片範圍內，並即時通知父 View 更新框線。"""
        if change == QGraphicsItem.ItemPositionChange and self._view is not None:
            new_pos = value
            if self._view._image_size is not None:
                w, h = self._view._image_size
                # 將座標夾緊在圖片邊界內（防止控制點拖到圖片外）
                x = max(0.0, min(new_pos.x(), float(w)))
                y = max(0.0, min(new_pos.y(), float(h)))
                new_pos = QPointF(x, y)
            self._view.handle_corner_moved(self.idx, new_pos)
            return new_pos
        return super().itemChange(change, value)


# ==================== UI 元件：影像顯示視窗 ====================

class ImageView(QGraphicsView):
    """
    支援縮放、平移與角點拖曳的影像顯示元件。
    corners_changed Signal: 角點位置變更時發出，供主視窗即時儲存狀態。
    """
    corners_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 圖片層（Z 軸 = 0，最底層）
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)

        # 四邊形框線層（Z 軸 = 1，位於圖片之上）
        self._polygon_item = QGraphicsPathItem()
        self._polygon_item.setPen(QPen(Qt.green, 4))
        self._polygon_item.setZValue(1)
        self._scene.addItem(self._polygon_item)

        # 四個角點控制點（Z 軸 = 2，最頂層）
        self._corner_items = []
        for i in range(4):
            h = CornerHandle(i)
            h._view = self
            h.setVisible(False)  # 預設隱藏，載入圖片後再顯示
            self._scene.addItem(h)
            self._corner_items.append(h)

        self._image_size = None  # (width, height) 快取，供邊界限制使用

        # 設定視圖互動行為
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_image(self, image: np.ndarray, show_corners: bool = False):
        """
        顯示新影像，傳入 None 則清空視圖。
        :param image: BGR 格式的 numpy 陣列
        :param show_corners: 是否顯示角點控制點（左側原圖用 True，右側結果用 False）
        """
        if image is None:
            self._photo.setPixmap(QPixmap())
            self._image_size = None
            for h in self._corner_items:
                h.setVisible(False)
            return

        h, w = image.shape[:2]
        # BGR → RGB → QImage → QPixmap（PySide6 需要 RGB 格式）
        rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self._photo.setPixmap(QPixmap.fromImage(qimg))
        self._image_size = (w, h)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        if not show_corners:
            for h in self._corner_items:
                h.setVisible(False)

    def set_corners(self, corners: np.ndarray):
        """
        設定並顯示四個角點的位置。傳入 None 則隱藏所有角點與框線。
        """
        if self._image_size is None:
            return

        if corners is None:
            for h in self._corner_items:
                h.setVisible(False)
            self._polygon_item.setPath(QPainterPath())  # 清除框線
            return

        for i, (x, y) in enumerate(corners):
            self._corner_items[i].setPos(QPointF(x, y))
            self._corner_items[i].setVisible(True)
        self._update_polygon_path()

    def get_corners(self):
        """
        取得目前四個角點的座標。
        :return: 4x2 float32 numpy 陣列；若角點未完全顯示則回傳 None
        """
        if self._image_size is None:
            return None
        pts = []
        for h in self._corner_items:
            if not h.isVisible():
                return None
            p = h.pos()
            pts.append([p.x(), p.y()])
        return np.array(pts, dtype="float32")

    def _update_polygon_path(self):
        """[內部] 依照目前四個角點位置更新綠色四邊形框線，並觸發 corners_changed Signal。"""
        pts = [h.pos() for h in self._corner_items if h.isVisible()]
        if len(pts) != 4:
            self._polygon_item.setPath(QPainterPath())
            return

        path = QPainterPath()
        path.moveTo(pts[0])
        path.lineTo(pts[1])
        path.lineTo(pts[2])
        path.lineTo(pts[3])
        path.closeSubpath()
        self._polygon_item.setPath(path)
        self.corners_changed.emit()  # 通知主視窗角點已變更

    def handle_corner_moved(self, idx: int, new_pos: QPointF):
        """角點被拖曳時的回呼函式，由 CornerHandle.itemChange 呼叫。"""
        self._update_polygon_path()

    def zoom(self, factor: float):
        """縮放視圖（僅在有圖片時有效）。"""
        if self._photo.pixmap().isNull():
            return
        self.scale(factor, factor)

    def wheelEvent(self, event):
        """滾輪縮放：向上放大（×1.25），向下縮小（×0.8）。"""
        if self._photo.pixmap().isNull():
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        self.scale(1.25 if angle > 0 else 0.8, 1.25 if angle > 0 else 0.8)


# ==================== 主視窗 ====================

class MainWindow(QMainWindow):
    """
    應用程式主視窗，整合影像載入、角點編輯、透視矯正與批次匯出功能。
    左側：原圖 + 可拖曳的角點選取框
    右側：透視矯正後的預覽結果
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("圖片梯形矯正")
        self.scanner = DocumentScanner()

        # --- 目前顯示中的狀態 ---
        self.current_image   = None  # 已套用旋轉的原始影像（numpy BGR）
        self.current_warped  = None  # 透視矯正後的影像（numpy BGR）
        self.current_path    = None  # 目前檔案的絕對路徑
        self.current_page    = None  # 目前 PDF 頁碼（None = 一般圖片）

        # --- 批次處理清單 ---
        # 格式: list of (path: str, page_idx: int | None)
        self.batch_files = []
        self.batch_index = -1

        # --- 每個檔案的持久狀態 ---
        # 格式: state_key -> {'corners': np.ndarray | None, 'rotation': int (0-3)}
        # state_key: "path" 或 "path::page_N"（PDF 頁面需包含頁碼以避免衝突）
        self.file_data = {}

        self._build_ui()

    # ==================== UI 建構 ====================

    def _build_ui(self):
        """建立主視窗的所有 UI 元件與佈局。"""

        # ---- 上方：左右分割的影像視窗 ----
        splitter = QSplitter(self)
        self.view_original = ImageView()
        self.view_result   = ImageView()
        splitter.addWidget(self.view_original)
        splitter.addWidget(self.view_result)
        splitter.setStretchFactor(0, 3)  # 左側（原圖）佔較多空間
        splitter.setStretchFactor(1, 2)  # 右側（結果）佔較少空間

        # 角點拖曳後即時儲存狀態到 file_data
        self.view_original.corners_changed.connect(self.on_corners_changed)

        # ---- 底部控制面板 ----
        control_panel = QWidget()
        panel_layout  = QHBoxLayout(control_panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)

        # -- 群組 1：檔案操作 --
        grp_file = QGroupBox("檔案 / 批次")
        layout_file = QGridLayout(grp_file)
        btn_open_file       = QPushButton("開啟檔案")
        btn_open_folder     = QPushButton("開啟資料夾")
        self.btn_export_all = QPushButton("全部匯出")
        self.btn_export_all.setStyleSheet("background-color: #d0f0c0;")
        self.btn_prev = QPushButton("◀ 上一張")
        self.btn_next = QPushButton("下一張 ▶")
        layout_file.addWidget(btn_open_file,       0, 0)
        layout_file.addWidget(btn_open_folder,     0, 1)
        layout_file.addWidget(self.btn_prev,       1, 0)
        layout_file.addWidget(self.btn_next,       1, 1)
        layout_file.addWidget(self.btn_export_all, 2, 0, 1, 2)  # 跨兩欄

        # -- 群組 2：編輯與檢視 --
        grp_edit = QGroupBox("編輯 / 檢視")
        layout_edit = QGridLayout(grp_edit)
        btn_rot_l    = QPushButton("↺ 向左旋轉")
        btn_rot_r    = QPushButton("向右旋轉 ↻")
        btn_reset    = QPushButton("重設選取範圍")
        btn_zoom_in  = QPushButton("放大 (+)")
        btn_zoom_out = QPushButton("縮小 (-)")
        layout_edit.addWidget(btn_rot_l,    0, 0)
        layout_edit.addWidget(btn_rot_r,    0, 1)
        layout_edit.addWidget(btn_reset,    1, 0, 1, 2)  # 跨兩欄
        layout_edit.addWidget(btn_zoom_in,  2, 0)
        layout_edit.addWidget(btn_zoom_out, 2, 1)

        # -- 群組 3：輸出設定 --
        grp_out = QGroupBox("輸出與操作")
        layout_out = QVBoxLayout(grp_out)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("輸出資料夾路徑")
        btn_browse     = QPushButton("瀏覽...")
        btn_apply      = QPushButton("預覽 / 套用")
        btn_export_one = QPushButton("匯出目前檔案")
        h_path = QHBoxLayout()
        h_path.addWidget(self.output_edit)
        h_path.addWidget(btn_browse)
        layout_out.addLayout(h_path)
        layout_out.addWidget(btn_apply)
        layout_out.addWidget(btn_export_one)

        panel_layout.addWidget(grp_file)
        panel_layout.addWidget(grp_edit)
        panel_layout.addWidget(grp_out)

        # ---- 進度條與狀態列 ----
        self.progress     = QProgressBar()
        self.progress.setValue(0)
        self.status_label = QLabel("就緒")

        # ---- 整合主佈局 ----
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.addWidget(splitter,       stretch=1)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.status_label)
        self.setCentralWidget(central)

        # ---- 連接所有按鈕訊號 ----
        btn_open_file.clicked.connect(self.on_open_file)
        btn_open_folder.clicked.connect(self.on_open_folder)
        self.btn_export_all.clicked.connect(self.on_batch_export_all)
        self.btn_prev.clicked.connect(self.on_prev_image)
        self.btn_next.clicked.connect(self.on_next_image)
        btn_rot_l.clicked.connect(lambda: self.on_rotate("left"))
        btn_rot_r.clicked.connect(lambda: self.on_rotate("right"))
        btn_reset.clicked.connect(self.on_reset_selection)
        btn_zoom_in.clicked.connect(lambda: self.view_original.zoom(1.25))
        btn_zoom_out.clicked.connect(lambda: self.view_original.zoom(0.8))
        btn_browse.clicked.connect(self.on_choose_output)
        btn_apply.clicked.connect(self.on_apply)
        btn_export_one.clicked.connect(self.on_export_current)

        self._update_nav_buttons()  # 初始化導覽按鈕狀態

    # ==================== 狀態管理 ====================

    def _get_current_state_key(self) -> str:
        """
        取得目前檔案的唯一狀態識別碼。
        PDF 頁面使用 "path::page_N" 格式，確保不同頁碼的狀態不互相衝突。
        """
        if self.current_page is not None:
            return f"{self.current_path}::page_{self.current_page}"
        return self.current_path

    def _update_file_data(self, state_key: str, corners=None, rotation_delta: int = 0):
        """
        安全地建立或更新指定 state_key 的檔案狀態。
        :param corners: 新的角點陣列（None 表示不更新角點）
        :param rotation_delta: 旋轉增量（1=CW90°, 3=CCW90°），0 表示不更新旋轉
        """
        if state_key not in self.file_data:
            self.file_data[state_key] = {'corners': None, 'rotation': 0}
        if corners is not None:
            self.file_data[state_key]['corners'] = corners
        if rotation_delta != 0:
            # rotation 值：0=0°, 1=90°CW, 2=180°, 3=270°CW（逆時針90°）
            current_rot = self.file_data[state_key]['rotation']
            self.file_data[state_key]['rotation'] = (current_rot + rotation_delta) % 4

    def on_corners_changed(self):
        """當使用者拖曳角點時，即時將最新座標儲存到 file_data（避免切換檔案後遺失）。"""
        if self.current_path:
            corners = self.view_original.get_corners()
            self._update_file_data(self._get_current_state_key(), corners=corners)

    def _apply_rotation(self, image: np.ndarray, k: int) -> np.ndarray:
        """
        依照 k 值旋轉圖片。
        :param k: 0=不旋轉, 1=順時針90°, 2=180°, 3=逆時針90°
        """
        # [優化] 使用字典取代 if-elif 鏈，更簡潔易讀
        rotations = {
            1: cv2.ROTATE_90_CLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        if k in rotations:
            return cv2.rotate(image, rotations[k])
        return image  # k=0，不旋轉

    # ==================== 導覽控制 ====================

    def _update_nav_buttons(self):
        """根據目前批次清單的位置，動態啟用或停用上一張/下一張按鈕。"""
        has_files = len(self.batch_files) > 0
        self.btn_prev.setEnabled(has_files and self.batch_index > 0)
        self.btn_next.setEnabled(has_files and self.batch_index < len(self.batch_files) - 1)

    def on_prev_image(self):
        """載入批次清單中的上一個檔案。"""
        if self.batch_index > 0:
            self.batch_index -= 1
            self.load_current_file()

    def on_next_image(self):
        """載入批次清單中的下一個檔案。"""
        if self.batch_index < len(self.batch_files) - 1:
            self.batch_index += 1
            self.load_current_file()

    # ==================== 開檔邏輯 ====================

    def _build_batch_list(self, files: list) -> list:
        """
        處理匯入的檔案清單，將 PDF 拆分為多個 (path, page_index) 項目。
        一般圖片的 page_index 為 None。
        :return: list of (path: str, page_idx: int | None)
        """
        batch = []
        for path in files:
            if path.lower().endswith('.pdf'):
                try:
                    doc = fitz.open(path)
                    for i in range(len(doc)):
                        batch.append((path, i))
                    doc.close()
                except Exception as e:
                    QMessageBox.warning(self, "錯誤", f"無法讀取 PDF {path}：\n{e}")
            else:
                batch.append((path, None))
        return batch

    def on_open_file(self):
        """開啟單一圖片或 PDF 檔案。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖片或 PDF", "",
            "圖片與 PDF (*.png *.jpg *.jpeg *.tif *.tiff *.pdf)"
        )
        if not path:
            return
        self.batch_files = self._build_batch_list([path])
        self.batch_index = 0
        self.file_data.clear()  # 清空舊有的狀態資料
        if self.batch_files:
            self.load_current_file()

    def on_open_folder(self):
        """開啟資料夾，自動掃描並載入所有支援格式的圖片與 PDF。"""
        folder = QFileDialog.getExistingDirectory(self, "開啟資料夾", "")
        if not folder:
            return
        exts  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}
        files = [str(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]
        if not files:
            QMessageBox.warning(self, "無檔案", "此資料夾中沒有支援的圖片或 PDF。")
            return
        self.batch_files = self._build_batch_list(files)
        self.batch_index = 0
        self.file_data.clear()  # 清空舊有的狀態資料
        if self.batch_files:
            self.load_current_file()

    def load_current_file(self):
        """
        依據 batch_index 載入並顯示目前的檔案。
        - 若 file_data 已有此檔案的狀態，套用已儲存的旋轉與角點
        - 若為首次載入，執行自動角點偵測並初始化狀態
        """
        if not (0 <= self.batch_index < len(self.batch_files)):
            return

        path, page_idx = self.batch_files[self.batch_index]

        # 讀取原始影像（尚未套用旋轉）
        try:
            img = self.scanner.load_image(path, page_idx)
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))
            return

        # [優化] 使用與 _get_current_state_key 相同邏輯計算 state_key
        # （此時 current_path/current_page 尚未更新，不能直接呼叫該方法）
        state_key = f"{path}::page_{page_idx}" if page_idx is not None else path

        # 首次載入此檔案：執行自動角點偵測
        if state_key not in self.file_data:
            corners = self.scanner.auto_detect_corners(img)
            if corners is None:
                corners = self.scanner.default_inset_corners(img, inset_ratio=0.05)
            self.file_data[state_key] = {'corners': corners, 'rotation': 0}

        # 套用已儲存的旋轉角度
        rotations = self.file_data[state_key]['rotation']
        img = self._apply_rotation(img, rotations)

        # 更新目前狀態變數
        self.current_image  = img
        self.current_path   = path
        self.current_page   = page_idx
        self.current_warped = None

        # 更新狀態列顯示
        name = os.path.basename(path)
        if page_idx is not None:
            name += f"（第 {page_idx + 1} 頁）"
        self.status_label.setText(f"{name}（{self.batch_index + 1}/{len(self.batch_files)}）")
        self.progress.setValue(int((self.batch_index + 1) / len(self.batch_files) * 100))

        # 顯示影像與角點
        self.view_original.set_image(img, show_corners=True)
        self.view_result.set_image(None)
        corners = self.file_data[state_key]['corners']
        self.auto_corners = corners.copy()
        self.view_original.set_corners(corners)
        self._update_nav_buttons()

    # ==================== 編輯功能 ====================

    def on_reset_selection(self):
        """對目前（已旋轉）影像重新執行自動角點偵測，並重設框線位置。"""
        if self.current_image is None:
            return
        corners = self.scanner.auto_detect_corners(self.current_image)
        if corners is None:
            corners = self.scanner.default_inset_corners(self.current_image, inset_ratio=0.05)
        self.view_original.set_corners(corners)
        self.status_label.setText("已重設為自動偵測角點")
        # on_corners_changed 會被 corners_changed Signal 自動觸發並儲存狀態

    def on_rotate(self, direction: str):
        """
        將目前影像旋轉 90°，同步轉換角點座標。
        :param direction: "left"（逆時針）或 "right"（順時針）
        """
        if self.current_image is None:
            return

        curr_corners = self.view_original.get_corners()
        h, w = self.current_image.shape[:2]

        if direction == "left":
            rot_delta   = 3  # 逆時針 90° = +270° CW
            rotated_img = cv2.rotate(self.current_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if curr_corners is not None:
                # 逆時針 90° 座標轉換公式：x' = y, y' = (w-1) - x
                new_corners = np.zeros_like(curr_corners)
                new_corners[:, 0] = curr_corners[:, 1]
                new_corners[:, 1] = w - curr_corners[:, 0]
                curr_corners = self.scanner._order_points(new_corners)

        elif direction == "right":
            rot_delta   = 1  # 順時針 90°
            rotated_img = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
            if curr_corners is not None:
                # 順時針 90° 座標轉換公式：x' = (h-1) - y, y' = x
                new_corners = np.zeros_like(curr_corners)
                new_corners[:, 0] = h - curr_corners[:, 1]
                new_corners[:, 1] = curr_corners[:, 0]
                curr_corners = self.scanner._order_points(new_corners)
        else:
            return

        # 更新狀態與視圖
        self.current_image  = rotated_img
        self.current_warped = None
        direction_text = "左" if direction == "left" else "右"
        self._update_file_data(
            self._get_current_state_key(),
            corners=curr_corners,
            rotation_delta=rot_delta
        )
        self.view_original.set_image(rotated_img, show_corners=True)
        self.view_result.set_image(None)
        if curr_corners is not None:
            self.view_original.set_corners(curr_corners)
        self.status_label.setText(f"已向{direction_text}旋轉（選取範圍已保留）")

    # ==================== 輸出功能 ====================

    def on_choose_output(self):
        """開啟資料夾選擇對話框，讓使用者指定匯出目的地。"""
        folder = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", "")
        if folder:
            self.output_edit.setText(folder)

    def _get_output_folder(self):
        """
        取得輸出資料夾路徑，若不存在則自動建立。
        優先使用使用者指定的路徑；若未指定，則在來源檔案旁建立 output 子資料夾。
        :return: Path 物件，或 None（路徑無法確定時）
        """
        user_out = self.output_edit.text().strip()
        if user_out:
            path = Path(user_out)
        elif self.current_path:
            path = Path(self.current_path).parent / "output"
        else:
            return None
        path.mkdir(parents=True, exist_ok=True)
        return path

    def on_apply(self):
        """套用透視矯正並在右側顯示預覽結果。"""
        if self.current_image is None:
            return
        corners = self.view_original.get_corners()
        if corners is None or len(corners) != 4:
            QMessageBox.warning(self, "警告", "請先確認四個角點已正確設定。")
            return
        warped = self.scanner.warp_perspective(self.current_image, corners)
        self.current_warped = warped
        self.view_result.set_image(warped)
        self.status_label.setText("已套用透視矯正，請至右側確認預覽")

    def _generate_output_path(self, out_dir: Path, stem: str) -> Path:
        """
        [工具方法] 產生不重複的輸出檔名，防止覆蓋已存在的檔案。
        命名規則：stem_scanned.png → stem_scanned_1.png → stem_scanned_2.png ...
        """
        out_path = out_dir / f"{stem}_scanned.png"
        idx = 1
        while out_path.exists():
            out_path = out_dir / f"{stem}_scanned_{idx}.png"
            idx += 1
        return out_path

    def on_export_current(self):
        """匯出目前檔案的透視矯正結果，匯出後自動前進至下一張。"""
        if self.current_image is None:
            return
        # 若尚未套用矯正，自動執行
        if self.current_warped is None:
            self.on_apply()
        if self.current_warped is None:
            return

        out_dir = self._get_output_folder()
        if not out_dir:
            return

        stem = Path(self.current_path).stem
        if self.current_page is not None:
            stem += f"_page_{self.current_page + 1}"

        out_path = self._generate_output_path(out_dir, stem)
        cv2.imwrite(str(out_path), self.current_warped)
        self.status_label.setText(f"已匯出：{out_path.name}")

        # 自動前進到下一張
        if 0 <= self.batch_index < len(self.batch_files) - 1:
            self.batch_index += 1
            self.load_current_file()

    def on_batch_export_all(self):
        """
        批次匯出所有批次清單中的檔案。
        - 已手動調整的頁面：使用儲存的角點與旋轉
        - 未調整的頁面：自動偵測角點
        """
        if not self.batch_files:
            QMessageBox.warning(self, "無檔案", "尚未載入任何檔案。")
            return

        out_dir = self._get_output_folder()
        if not out_dir:
            QMessageBox.warning(self, "警告", "請先選擇輸出資料夾。")
            return

        reply = QMessageBox.question(
            self, "批次匯出",
            f"即將處理並匯出 {len(self.batch_files)} 頁。\n"
            "未手動調整的頁面將自動偵測角點，是否繼續？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        self.status_label.setText("開始批次匯出...")
        count = 0
        self.setEnabled(False)
        QApplication.processEvents()

        try:
            for i, (str_path, page_idx) in enumerate(self.batch_files):
                p         = Path(str_path)
                state_key = f"{str_path}::page_{page_idx}" if page_idx is not None else str_path

                # 目前正在預覽且已有矯正結果，直接使用以節省重複運算
                if i == self.batch_index and self.current_warped is not None:
                    final_img = self.current_warped
                else:
                    try:
                        img = self.scanner.load_image(str_path, page_idx)
                    except Exception:
                        continue  # 跳過無法讀取的檔案，繼續處理其他

                    rotations = 0
                    corners   = None

                    # 套用已儲存的手動調整狀態
                    if state_key in self.file_data:
                        data      = self.file_data[state_key]
                        rotations = data['rotation']
                        corners   = data['corners']

                    img = self._apply_rotation(img, rotations)

                    # 無儲存角點則執行自動偵測
                    if corners is None:
                        corners = self.scanner.auto_detect_corners(img)
                        if corners is None:
                            corners = self.scanner.default_inset_corners(img, inset_ratio=0.05)
                        self._update_file_data(state_key, corners=corners)

                    final_img = self.scanner.warp_perspective(img, corners)

                stem = p.stem
                if page_idx is not None:
                    stem += f"_page_{page_idx + 1}"
                out_path = self._generate_output_path(out_dir, stem)

                cv2.imwrite(str(out_path), final_img)
                count += 1

                # 更新進度顯示
                self.progress.setValue(int((i + 1) / len(self.batch_files) * 100))
                self.status_label.setText(f"匯出中 {i + 1}/{len(self.batch_files)}：{out_path.name}")
                QApplication.processEvents()

            QMessageBox.information(self, "批次完成", f"成功匯出 {count} 個檔案。")
            self.status_label.setText("批次匯出完成。")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"批次處理失敗：\n{e}")
        finally:
            self.setEnabled(True)  # 無論成功或失敗都重新啟用 UI


# ==================== 程式進入點 ====================

def main():
    """應用程式主函式，建立 QApplication 並啟動主視窗。"""
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

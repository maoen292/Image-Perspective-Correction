# Image Perspective Correction Tool / 圖片梯形矯正與透視處理工具

[English](#english) | [繁體中文](#繁體中文)

---

<h2 id="english">English</h2>

This is a Python-based GUI tool (PySide6 + OpenCV) designed to correct keystone distortion and perspective in documents and images. It supports both single image and PDF processing, and features batch export capabilities, making it ideal for users who need to process a large number of scanned or photographed documents.

### Core Features
* **Automatic Corner Detection**: Smartly identifies document edges, saving you the time of manual cropping. *(Note: This feature may occasionally be inaccurate and can be fine-tuned by dragging the points with your mouse).*
* **Real-time Preview & Dragging**: The original image on the left provides four control points for fine-tuning, while the right side displays the corrected result in real-time.
* **Multiple Format Support**: Can read JPG, PNG, TIFF, and PDF files.
* **Batch Processing Capability**: Load an entire folder, quickly apply correction settings, and batch export with a single click.

### How to Use
1. Install required packages: `opencv-python`, `numpy`, `PySide6`, `PyMuPDF`
2. Run the main script: `image_fix_09.py`
3. Click "開啟檔案" (Open File) or "開啟資料夾" (Open Folder) to load the images or PDFs you want to process.
4. If the automatically detected boundaries are slightly off, simply drag the red dots on the left screen to fine-tune.
5. Click "預覽 / 套用" (Preview / Apply) to confirm the result, then export your images.

### Disclaimer & License
* This project is open-sourced under the **[MIT License](https://opensource.org/licenses/MIT)**. You are welcome to freely share, modify, and use this code, but please retain the original author's attribution (Author: 毛恩 MaoEn).
* Most of the code in this project was written and optimized with the assistance of AI (Claude / Gemini).

---

### Highly Recommand You to read following
Since you've scrolled down this far, besides writing code, I'd like to use this little space to promote my absolute favorite V-Tuber! 
If you find this tool has saved you valuable time, I hope you can check out her channel. **Don't forget to subscribe to her: 白星優米 (Umi)**!
[白星優米 Umi YouTube Channel](https://youtube.com/@umitw46?si=YAS3N6vEymUu1UDW)

<br>

---

<h2 id="繁體中文">繁體中文</h2>

這是一個基於 Python (PySide6 + OpenCV) 開發的圖形化介面工具，專門用來處理文件、圖片的「梯形變形」與「透視矯正」。支援單張圖片與 PDF 處理，並具備批次匯出的功能，非常適合需要大量處理掃描檔或翻拍文件的使用者。

### 核心功能
* **自動角點偵測**：智慧辨識文件邊緣，省去手動拉框的時間 (此功能有時會辨識不正確，可透過滑鼠直接拖曳微調)
* **即時預覽與拖曳**：左側原圖提供四個控制點供微調，右側即時顯示矯正結果
* **支援多種格式**：可讀取 JPG, PNG, TIFF 以及 PDF 檔案
* **批次處理能力**：載入整個資料夾，快速套用矯正設定並一鍵批次匯出

### 如何使用
1. 安裝必要的套件：`opencv-python`, `numpy`, `PySide6`, `PyMuPDF`
2. 執行主程式：`image_fix_09.py`
3. 點擊「開啟檔案」或「開啟資料夾」載入需要處理的圖片或 PDF。
4. 如自動偵測的邊界有偏差，直接用滑鼠拖曳左側畫面中的紅點進行微調。
5. 點選「預覽 / 套用」確認效果後，即可匯出圖片。

### 聲明與授權條款 (License)
* 本專案採用 **[MIT License](https://opensource.org/licenses/MIT)** 開源授權。歡迎自由分享、修改與使用本代碼，但請保留原作者署名（作者：毛恩 MaoEn）。
* 本專案的大部分程式碼是在 AI（Claude / Gemini）的輔助下編寫與優化完成的。

---

### 私心推薦
既然你都滑到這裡了，除了寫代碼，我也想藉這個小空間推廣一下我最喜歡的 V-Tuber！
如果你覺得這個工具幫你省下了寶貴的時間，希望你可以去看看她的頻道，**記得訂閱她：白星優米**！
[白星優米 YouTube 頻道連結](https://youtube.com/@umitw46?si=YAS3N6vEymUu1UDW)

- - -

# Semantic Segmentation (UNet / ResNet)

本專案實現了基於深度學習的像素級語意分割（Semantic Segmentation）架構，核心聚焦於 **UNet** 與 **ResNet** 架構的實現、優化與對比實驗。
專案從底層的 Data Pipeline 的建立、image enhancement、損失函數優化，到解決語意分割常見的類別不平衡問題，均進行了模組化設計與效能調適，可適用於高效能的影像辨識與邊緣切割任務 。

## 架構設計 (Architecture & Methodology)

為平衡高階語意特徵捕捉與低階空間細節還原，本專案主要實現並探討了以下兩種核心架構：

1. UNet (Encoder-Decoder with Skip Connections) 

* **收縮路徑 (Contracting Path)**：透過連續的卷積與池化操作，逐層提取影像的上下文特徵（Context）。
* **擴張路徑 (Expanding Path)**：利用轉置卷積（Transposed Convolution）進行上採樣，恢復影像的解析度與空間維度。
* **特徵拼接 (Skip Connections)**：將編碼器層的特徵圖（Feature Maps）直接拼接到對應的解碼器層，有效彌補下採樣過程中的低階幾何資訊損失，精確鎖定邊界細節。

2. ResNet Backbone 整合 

* 引入殘差結構（Residual Blocks）作為特徵提取網絡（Encoder），藉由殘差映射（Residual Mapping）解決深層網路梯度消失（Gradient Vanishing）的問題。
* 藉由預訓練權重（Pre-trained Weights）加速模型收斂，並大幅提升在有限資料集下的泛化能力與 Agent 的 robustness。

---

## 開發核心與效能優化 (Key Challenges & Solutions)

本專案在實作過程中，針對語意分割進行了調校：

* **類別不平衡（Class Imbalance）優化**：
在語意分割中，背景通常佔據絕大部分像素，易導致模型預測偏向背景。本專案捨棄單一的交叉熵損失，採用 **Dice Loss** 與 **Focal Loss** 的加權組合（Combined Loss Function），強制模型專注於稀有或難分類的邊緣目標像素。
* **資料管道與增強（Data Augmentation Pipeline）**：
為防範模型過擬合，建立彈性的影像增強管道，包含隨機旋轉（Rotation）、翻轉（Flipping）、亮度與對比度調整（Brightness/Contrast adjustment），並確保影像與分割遮罩（Masks）在幾何變換時保持絕對同步。
* **學習率調控策略（Learning Rate Scheduling）**：
採用動態學習率衰減（如 ReduceLROnPlateau 或 Cosine Annealing），在訓練初期維持高學習率以快速探索解空間，後期精細微調以確保收斂至最優解。

---

##  評估指標 (Evaluation Metrics)

本專案採用語意分割領域標準的量化指標進行模型評估，非簡化的 Accuracy 數據：

* **mIoU (Mean Intersection over Union)**：評估預測遮罩與真實標籤交集與聯集的比例，為衡量分割精準度的核心指標。
* **Dice Coefficient (F1-Score at Pixel Level)**：評估預測結果與真實邊界的重疊程度。
* **Pixel Accuracy**：計算影像中被正確分類的像素百分比。

---

##  環境配置與安裝 (Installation)

### 1. 依賴套件

環境建議使用 Python 3.10+，核心依賴套件如下：

```bash
pip install torch torchvision numpy opencv-python matplotlib scikit-learn tqdm

```

*(備註：亦可根據實作框架更換為 `tensorflow` / `keras` 相關依賴。)*

### 2. 專案結構

```text
├── data/               # 訓練與驗證資料集 (Images & Masks)
├── models/             # 模型架構定義 (unet.py, resnet_backbone.py)
├── utils/              # 資料載入、影像增強與指標計算模組
├── train.py            # 訓練與驗證主程式
├── evaluate.py         # 模型評估與指標產出
└── inference.py        # 單張/批次影像預測與遮罩可視化

```

---

## Usage

### 模型訓練

執行 `train.py` 啟動訓練，可透過參數指定架構與超參數：

```bash
python train.py --model unet --epoch 50 --batch_size 16 --lr 1e-4 --loss combined

```

### 影像推論與可視化

使用訓練完成的權重進行預測，並輸出對比圖（原圖、真實遮罩、預測遮罩）：

```bash
python inference.py --weight_path ./checkpoints/best_model.pth --image_path ./data/test/img.png

```

---

## 開發者與專案資訊

* **開發者**：歐靜嬡 (Skylar Ou) 

* **領域分類**：電腦視覺 (Computer Vision) / 深度學習 (Deep Learning) / 語意分割 

* **Repository**：[Semantic-Segmentation](https://github.com/SkylarOu9005/Semantic-Segmentation)

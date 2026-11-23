"""回測設定。可用環境變數覆寫部分參數。"""
import os
from pathlib import Path

ROOT = Path(os.getenv("BT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

# 資料起點（含動能所需的前置資料）與回測起點
DATA_START = os.getenv("DATA_START", "2015-01-05")
BACKTEST_START = os.getenv("BACKTEST_START", "2016-01")

# 選股規則
TOP_N = int(os.getenv("TOP_N", "10"))
MIN_PRICE = 10.0                 # 股價下限
MIN_VALUE_20 = 20_000_000        # 20 日平均成交金額下限（元）
MIN_HISTORY_DAYS = 60            # 上市至少 60 個交易日才納入
REBALANCE_DAY = 11               # 每月第一個 >= 11 日的交易日做決策（月營收 10 日截止）
# 策略 A 排除營收年增率超過 100% 的股票：多為建案完工入帳、一次性營收等基期效應，
# 2016-2021 開發期這群表現明顯較差，2022 以後保留期也確認改善（但仍未通過穩健度檢查）
A_MAX_REV_YOY = 1.0

# 交易成本
FEE_RATE = 0.001425
FEE_DISCOUNT = float(os.getenv("FEE_DISCOUNT", "1.0"))   # 券商手續費折數，例如 0.6
TAX_RATE = 0.003
ROUND_TRIP_COST = FEE_RATE * FEE_DISCOUNT * 2 + TAX_RATE

# 基準與目標
BENCH = "0050"
TARGET_HIT = 0.60

# LightGBM walk-forward
MIN_TRAIN_MONTHS = 36

# 抓資料
REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "3.5"))  # 證交所建議每次間隔數秒，避免被封鎖
MAX_CONSECUTIVE_ERRORS = 5

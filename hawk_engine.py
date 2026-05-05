#!/usr/bin/env python3
"""
HAWK ENGINE v28 — Major overhaul: SV graduated penalty, dynamic retest, regime-adaptive weights, IMS fractional.
"""
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OI_SNAP_INTERVAL    = 180
OI_LOOKBACK_SNAPS   = 2
OI_DELTA_THRESHOLD  = 0.015
OI_PRICE_CONFIRM_PCT= 0.001

PCR_EXTREME_BULL    = 1.4
PCR_EXTREME_BEAR    = 0.75
PCR_SHIFT_TRIGGER   = 0.12

FUTURES_VWAP_MIN_SECS = 120
RSI_PERIOD          = 14
RSI_BULL_THRESHOLD  = 65   # v17.3: was 58 — too noisy, RSI 58 is not a real signal
RSI_BEAR_THRESHOLD  = 35   # v17.3: was 42 — need genuine oversold, not mild dip

STRIKES_AROUND_ATM  = 4
STRIKE_STEP         = 50

MIN_SIGNALS_REQUIRED = 4   # v19-LIVE: was 6 — too many signals needed before entry
MIN_SIGNALS_HIGH_VOL = 6   # v19-LIVE: was 8
WARMUP_SNAPS        = 1
MIN_CANDLES_BEFORE_ENTRY = 10  # v19-LIVE: was 15 — 15min warmup too long, misses first moves

SIGNAL_WEIGHTS = {
    # v29: Empirically recalibrated — weights now reflect ACTUAL win rates from 199 live trades
    # (hawk_learning.json). Signals below 43% win rate (break-even) are penalised; above 50%
    # are boosted. First-principles rule: a sub-coin-flip signal is a CONTRA-indicator and
    # must not be given positive weight in the vote pool.
    #
    # Empirical win rates (199 trades):
    #   RSI=68.8%, BB_SQUEEZE=57.3%, STRUCTURE=56.4%, STRADDLE_GAMMA=58.8%  → HIGH weight
    #   VWAP=53.9%, MOMENTUM=52.2%, OPT_OI=52.9%, PREM_VEL=53.8%           → MEDIUM weight
    #   BREAKOUT15=47.8%, PCP=48.8%, COI_PCR=43.8%                          → LOW weight
    #   SUPERTREND=37.9%, MAX_PAIN=37.8%, FUT_VEL=38.6%, ORB=39.4%         → NEAR-ZERO (penalty)
    #   MACD_HIST=42.5%, ADX=44.1%                                           → REDUCED

    # TREND signals
    "VWAP":        1.4,    # 53.9% WR — structural, keep high
    "VWAP_RETEST": 0.8,    # derivative of VWAP, low independence
    "STRUCTURE":   1.5,    # v29: raised 1.3→1.5 — 56.4% WR, truly independent
    "SUPERTREND":  0.3,    # v29: slashed 1.0→0.3 — 37.9% WR = contra-indicator at high weight
    "BREAKOUT15":  0.9,    # v29: reduced 1.2→0.9 — 47.8% WR, marginally below break-even
    "ORB":         0.3,    # v29: slashed 1.1→0.3 — 39.4% WR = near contra-indicator
    "AVWAP":       0.3,    # 33.3% WR — contra-indicator, near-zero

    # MOMENTUM signals
    "RSI":         1.6,    # v29: raised 0.3→1.6 — 68.8% WR = BEST empirical signal, was criminally under-weighted
    "MACD_HIST":   0.3,    # v29: reduced 0.4→0.3 — 42.5% WR, below break-even
    "MOMENTUM":    0.8,    # v29: raised 0.1→0.8 — 52.2% WR, was near-zero for no reason
    "BB_SQUEEZE":  1.2,    # v29: raised 0.8→1.2 — 57.3% WR, independent vol signal
    "ADX":         0.6,    # v29: reduced 1.0→0.6 — 44.1% WR, below break-even

    # OPTIONS signals — leading indicators
    "PREM_VEL":    2.0,    # 53.8% WR — keeps highest weight, confirmed leading indicator
    "OPT_OI":      1.2,    # v29: raised 1.0→1.2 — 52.9% WR with small sample, promising
    "STRADDLE_GAMMA": 2.0, # 58.8% WR — keeps highest weight, confirmed winner
    "MKT_OI":      0.4,    # futures OI is slow, minor
    "COI_PCR":     0.5,    # v29: reduced 0.7→0.5 — 43.8% WR, marginally below break-even
    "MAX_PAIN":    0.2,    # v29: slashed 0.5→0.2 — 37.8% WR = near contra-indicator
    "PCP":         1.0,    # 48.8% WR — neutral, keep as confirmation

    # ENERGY signal
    "FUT_VEL":     0.2,    # v29: slashed 0.8→0.2 — 38.6% WR = contra-indicator at high weight
}
WEIGHTED_VOTE_THRESHOLD = 6.0   # v29: raised 4.5→6.0 — 4.5 was too permissive; 34% win rate proves gates were too easy
WEIGHTED_CONVICTION_MIN = 0.62  # v29: raised 0.56→0.62 — 0.56 (barely > coin flip) produced 34% win rate; need genuine conviction

CANDLE_INTERVAL_SEC = 60
CANDLE_5MIN_SEC     = 300
ATR_PERIOD          = 14
ATR_MULTIPLIER      = 1.5
ATR_HIGH_MULT       = 1.5

SIGNAL_FRESHNESS_SEC = 30.0
CONVICTION_STALE_TICKS = 900

MAX_HOLD_SECONDS    = 1800
NO_ENTRY_AFTER_MINS = 45

# v17.2: Market regime thresholds
ADX_TRENDING_THRESHOLD = 22     # ADX > 22 = trending (research: >25 is strong, but 22 catches early)
ADX_STRONG_TREND = 30           # ADX > 30 = strong trend — trust signals more
RANGE_CONVICTION_PENALTY = 0.08 # v17.4: raised from 0.06 — range trades are 0% WR
PULLBACK_DEPTH_MIN = 0.25       # minimum retracement (25% of recent swing)
PULLBACK_DEPTH_MAX = 0.65       # maximum retracement (65% — beyond this, trend may be broken)

# ─────────────────────────────────────────────────────────────────────────────
# VIX REGIME
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VIXRegime:
    LOW  = "LOW"
    MID  = "MID"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class VIXTracker:
    def __init__(self):
        self._vix_values: deque = deque(maxlen=100)
        self._last_vix: float = 18.0
        self._regime: str = VIXRegime.MID
        self._straddle_history: deque = deque(maxlen=50)

    def reset(self):
        self._vix_values.clear()
        self._last_vix = 18.0
        self._regime = VIXRegime.MID
        self._straddle_history.clear()

    def update(self, vix_value: float = 0.0, spot: float = 0.0,
               ce_ltp: float = 0.0, pe_ltp: float = 0.0):
        if vix_value > 0:
            self._last_vix = vix_value
        elif spot > 0 and ce_ltp > 0 and pe_ltp > 0:
            straddle_pct = (ce_ltp + pe_ltp) / spot
            self._straddle_history.append(straddle_pct)
            self._last_vix = straddle_pct * 1800
            self._last_vix = max(8, min(50, self._last_vix))

        self._vix_values.append(self._last_vix)
        v = self._last_vix
        if v < 14: self._regime = VIXRegime.LOW
        elif v < 20: self._regime = VIXRegime.MID
        elif v < 28: self._regime = VIXRegime.HIGH
        else: self._regime = VIXRegime.EXTREME

    @property
    def vix(self) -> float: return self._last_vix
    @property
    def regime(self) -> str: return self._regime

    def conviction_floor(self) -> float:
        # v23: loosened — retest confirmation is the real quality gate now
        floors = {VIXRegime.LOW:0.48, VIXRegime.MID:0.50, VIXRegime.HIGH:0.52, VIXRegime.EXTREME:0.55}
        return floors.get(self._regime, 0.50)

    def sl_multiplier(self) -> float:
        mults = {VIXRegime.LOW:0.85, VIXRegime.MID:1.0, VIXRegime.HIGH:1.3, VIXRegime.EXTREME:1.6}
        return mults.get(self._regime, 1.0)

    def min_signals_required(self) -> int:
        # v23: loosened — retest confirmation filters bad entries, engine just generates candidates
        counts = {VIXRegime.LOW:4, VIXRegime.MID:4, VIXRegime.HIGH:5, VIXRegime.EXTREME:6}
        return counts.get(self._regime, 4)

    def weight_threshold(self) -> float:
        # v29: raised to match empirically-derived WEIGHTED_VOTE_THRESHOLD=6.0
        # Sub-thresholds scale with VIX — higher VIX = noisier signals = demand more weight
        thresholds = {VIXRegime.LOW:5.5, VIXRegime.MID:6.0, VIXRegime.HIGH:7.0, VIXRegime.EXTREME:8.5}
        return thresholds.get(self._regime, 6.0)

    def is_vix_spike(self) -> bool:
        if len(self._vix_values) < 10: return False
        recent = list(self._vix_values)[-5:]
        older = list(self._vix_values)[-10:-5]
        if not older: return False
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        return avg_recent > avg_older * 1.25

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OISnapshot:
    ts: float
    futures_price: float
    total_ce_oi: float
    total_pe_oi: float
    pcr: float
    max_ce_strike: float
    max_pe_strike: float
    ce_oi_by_strike: Dict[float, float]
    pe_oi_by_strike: Dict[float, float]
    atm: float

@dataclass
class SignalVote:
    name: str
    direction: str
    score: float
    reason: str
    ts: float = 0.0
    def __post_init__(self):
        if self.ts == 0.0:
            self.ts = time.monotonic()

@dataclass
class EngineResult:
    direction: str
    score: float
    votes_for: int
    vote_detail: List[str]
    entry_allowed: bool
    blocked_reason: str
    suggested_sl_pts: float
    suggested_sl_pct: float
    suggested_tgt_pts: float
    smart_money_bias: str
    reversal_warnings: List[str] = field(default_factory=list)
    vix_regime: str = "MID"
    market_regime: str = "UNKNOWN"
    market_phase: str = "UNKNOWN"   # v28 Innovation A: current phase from MarketPhaseTracker
    entry_thesis: Dict = field(default_factory=dict)
    probability: float = 0.0
    imm_momentum: float = 0.0
    gate_diagnostics: Dict = field(default_factory=dict)  # v25-LIVE: all gate values for dashboard

# ─────────────────────────────────────────────────────────────────────────────
# THESIS TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class ThesisTracker:
    def __init__(self):
        self._thesis: Dict = {}
        self._active = False
    def reset(self):
        self._thesis = {}
        self._active = False
    def record_entry(self, signals, direction, vwap, structure_dir, supertrend_dir):
        self._thesis = {
            "direction": direction,
            "entry_vwap_side": "above" if direction == "CE" else "below",
            "entry_structure": structure_dir,
            "entry_supertrend": supertrend_dir,
            "agreeing_signals": [s.name for s in signals if s.direction == direction],
            "entry_ts": time.monotonic(),
        }
        self._active = True
    def check_thesis(self, signals, direction, fut_price, vwap):
        if not self._active:
            return True, 0, ""
        invalidations = 0
        reasons = []
        if direction == "CE" and fut_price < vwap * 0.9995:
            invalidations += 1; reasons.append("VWAP_CROSS")
        elif direction == "PE" and fut_price > vwap * 1.0005:
            invalidations += 1; reasons.append("VWAP_CROSS")
        original_agreeing = set(self._thesis.get("agreeing_signals", []))
        opposite = "PE" if direction == "CE" else "CE"
        flipped = 0
        for s in signals:
            if s.name in original_agreeing and s.direction == opposite:
                flipped += 1
        if flipped >= 3:
            invalidations += 1; reasons.append(f"SIGNALS_FLIPPED({flipped})")
        return invalidations == 0, invalidations, "|".join(reasons)
    @property
    def is_active(self): return self._active

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE AGGREGATORS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Candle:
    ts: float
    open: float
    high: float
    low: float
    close: float
    ticks: int = 1

class CandleAggregator:
    def __init__(self, interval_sec: float = CANDLE_INTERVAL_SEC, max_candles: int = 120):
        self._interval  = interval_sec
        self._candles: deque = deque(maxlen=max_candles)
        self._current: Optional[Candle] = None
        self._candle_start: float = 0.0
    def reset(self):
        self._candles.clear(); self._current = None; self._candle_start = 0.0
    def update(self, price: float) -> Optional[Candle]:
        now = time.monotonic()
        if self._current is None:
            self._current = Candle(ts=now, open=price, high=price, low=price, close=price, ticks=1)
            self._candle_start = now
            return None
        self._current.high  = max(self._current.high, price)
        self._current.low   = min(self._current.low, price)
        self._current.close = price
        self._current.ticks += 1
        if now - self._candle_start >= self._interval:
            completed = self._current
            self._candles.append(completed)
            self._current = Candle(ts=now, open=price, high=price, low=price, close=price, ticks=1)
            self._candle_start = now
            return completed
        return None
    @property
    def closed_candles(self) -> List[Candle]: return list(self._candles)
    @property
    def n_candles(self) -> int: return len(self._candles)
    @property
    def current(self) -> Optional[Candle]: return self._current

class FiveMinCandleAggregator:
    def __init__(self):
        self._buf: List[Candle] = []
        self._completed: deque = deque(maxlen=30)
        self._trend: str = "NEUTRAL"
        self._ema5: float = float('nan')
        self._ema10: float = float('nan')
    def reset(self):
        self._buf.clear(); self._completed.clear(); self._trend = "NEUTRAL"; self._ema5 = self._ema10 = float('nan')
    def on_1min_candle(self, candle: Candle) -> Optional[Candle]:
        self._buf.append(candle)
        if len(self._buf) >= 5:
            bar = Candle(
                ts=self._buf[0].ts,
                open=self._buf[0].open,
                high=max(c.high for c in self._buf),
                low=min(c.low for c in self._buf),
                close=self._buf[-1].close,
                ticks=sum(c.ticks for c in self._buf)
            )
            self._completed.append(bar)
            self._buf.clear()
            self._update_trend(bar)
            return bar
        return None
    def _update_trend(self, bar: Candle):
        a5 = 2 / (5 + 1); a10 = 2 / (10 + 1)
        c = bar.close
        if math.isnan(self._ema5):
            self._ema5 = self._ema10 = c
        else:
            self._ema5 = a5 * c + (1 - a5) * self._ema5
            self._ema10 = a10 * c + (1 - a10) * self._ema10
        if self._ema5 > self._ema10 * 1.0002: self._trend = "CE"
        elif self._ema5 < self._ema10 * 0.9998: self._trend = "PE"
        else: self._trend = "NEUTRAL"
    @property
    def trend(self) -> str: return self._trend
    @property
    def bars(self) -> List[Candle]: return list(self._completed)

# ─────────────────────────────────────────────────────────────────────────────
# ATR CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
class ATRCalculator:
    def __init__(self, period: int = ATR_PERIOD):
        self._period = period
        self._trs = deque(maxlen=period * 2)
        self._prev_close = float('nan')
        self.atr = float('nan')
        self._atr_history = deque(maxlen=60)
    def reset(self):
        self._trs.clear(); self._prev_close = float('nan'); self.atr = float('nan'); self._atr_history.clear()
    def on_candle(self, candle: Candle):
        if not math.isnan(self._prev_close):
            tr = max(candle.high - candle.low, abs(candle.high - self._prev_close), abs(candle.low - self._prev_close))
            self._trs.append(tr)
            if len(self._trs) >= self._period:
                self.atr = float(np.mean(list(self._trs)[-self._period:]))
                self._atr_history.append(self.atr)
        self._prev_close = candle.close
    def sl_points(self, vix_mult: float = 1.0) -> float:
        if math.isnan(self.atr): return 10.0
        return max(round(self.atr * ATR_MULTIPLIER * vix_mult, 1), 6.0)
    def sl_pct_of_premium(self, vix_mult: float = 1.0) -> float:
        base_pct = 0.15
        adjusted = base_pct * vix_mult
        return max(0.08, min(0.30, adjusted))
    def is_high_volatility(self) -> bool:
        if math.isnan(self.atr) or len(self._atr_history) < 5: return False
        median_atr = float(np.median(list(self._atr_history)))
        if median_atr <= 0: return False
        return self.atr > median_atr * ATR_HIGH_MULT
    def is_low_volatility(self) -> bool:
        if math.isnan(self.atr) or len(self._atr_history) < 10: return False
        median_atr = float(np.median(list(self._atr_history)))
        if median_atr <= 0: return False
        return self.atr < median_atr * 0.6

# ─────────────────────────────────────────────────────────────────────────────
# PRICE VELOCITY FILTER
# ─────────────────────────────────────────────────────────────────────────────
class PriceVelocityFilter:
    WINDOW = 20
    def __init__(self):
        self._prices: deque = deque(maxlen=self.WINDOW + 2)
        self._ts: deque = deque(maxlen=self.WINDOW + 2)
    def reset(self):
        self._prices.clear(); self._ts.clear()
    def update(self, price: float):
        self._prices.append(price); self._ts.append(time.monotonic())
    def velocity_confirms(self, direction: str) -> Tuple[bool, float]:
        if len(self._prices) < self.WINDOW:
            return True, 0.0
        prices = list(self._prices); ts = list(self._ts)
        dt = ts[-1] - ts[-self.WINDOW]
        if dt < 1.0:
            return True, 0.0
        vel = (prices[-1] - prices[-self.WINDOW]) / dt
        if direction == "CE":
            if vel < -0.5: return False, vel
            return True, vel
        elif direction == "PE":
            if vel > 0.5: return False, vel
            return True, vel
        return True, vel


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 20: FUTURES VELOCITY — Is the underlying ACTUALLY moving right now?
# ─────────────────────────────────────────────────────────────────────────────
class FuturesVelocitySignal:
    """Measures real-time futures price movement to answer two questions:

    1. Is price moving AT ALL? (anti-stall: votes NEUTRAL when flat → fewer votes → no entry)
    2. Has the move already happened? (anti-chase: votes NEUTRAL when exhausted)

    This is NOT a filter/threshold. It's a voting signal that participates in
    consensus. When price is flat, it votes NEUTRAL — reducing the total vote
    count naturally. When price is moving with acceleration, it votes strongly.
    When the move looks exhausted (large displacement + decelerating), it
    votes NEUTRAL — pulling consensus down.

    Key design: uses absolute price displacement + velocity + acceleration.
    None of the existing 19 signals measure whether futures price is moving.
    - MOMENTUM (signal 3): EMA crossover — lagging, shows trend not movement
    - PriceVelocityFilter: only used as penalty, not a voter
    - WinProbabilityCalculator.immediate_momentum: post-entry penalty only
    - STRADDLE_GAMMA: measures option straddle, not futures

    This signal fills that gap.

    v28 Innovation E: Thresholds are now ATR-NORMALIZED instead of fixed points.
    Fixed thresholds (1.5/3.0/12.0 pts) fail on different volatility regimes:
      - High-vol day (ATR=10): 1.5pt is noise, yet old code treated it as signal
      - Low-vol day (ATR=3): 12pt is 4× ATR, yet old code used same exhaustion threshold
    With ATR normalization, each threshold scales with actual market volatility.
    Fallback to fixed values during ATR warmup (first ~14 candles).
    """

    # Windows in ticks (~3.6 ticks/sec)
    SHORT_WINDOW = 18      # ~5 seconds: is price moving RIGHT NOW?
    LONG_WINDOW = 54       # ~15 seconds: has a significant move already happened?
    MIN_TICKS = 20         # warmup before voting

    # v28 Innovation E: ATR-NORMALIZED thresholds (replace fixed point values)
    # These multipliers are applied to the current session ATR to get dynamic thresholds.
    # Fallback fixed values used only during warmup before ATR is available.
    FLAT_ATR_MULT = 0.30       # flat = displacement < 0.30 × ATR
    MOVE_ATR_MULT = 0.60       # real move = displacement > 0.60 × ATR
    EXHAUSTION_ATR_MULT = 2.5  # exhaustion = displacement > 2.5 × ATR
    EXHAUSTION_DECEL = -0.02   # acceleration negative = decelerating (fading move)

    # Fixed fallbacks for warmup (before ATR is available — ~first 14 candles)
    _FALLBACK_FLAT = 1.5
    _FALLBACK_MOVE = 3.0
    _FALLBACK_EXHAUSTION = 12.0

    def __init__(self):
        self._prices: deque = deque(maxlen=self.LONG_WINDOW + 5)
        self._n = 0
        self._last_vote = SignalVote("FUT_VEL", "NEUTRAL", 0.0, "cold")
        self._atr_ref = None  # v28 Innovation E: reference to ATRCalculator for dynamic thresholds

    def set_atr_ref(self, atr_calc):
        """Wire ATR reference for volatility-normalized thresholds."""
        self._atr_ref = atr_calc

    def _thresholds(self) -> tuple:
        """Compute dynamic thresholds from ATR, or use fixed fallbacks.
        Returns: (flat_threshold, move_threshold, exhaustion_threshold)
        """
        if self._atr_ref and not math.isnan(self._atr_ref.atr) and self._atr_ref.atr > 0.5:
            atr = self._atr_ref.atr
            return (atr * self.FLAT_ATR_MULT,
                    atr * self.MOVE_ATR_MULT,
                    atr * self.EXHAUSTION_ATR_MULT)
        return (self._FALLBACK_FLAT, self._FALLBACK_MOVE, self._FALLBACK_EXHAUSTION)

    def reset(self):
        self._prices.clear()
        self._n = 0
        self._last_vote = SignalVote("FUT_VEL", "NEUTRAL", 0.0, "cold")

    def update(self, price: float) -> SignalVote:
        if price <= 0:
            return SignalVote("FUT_VEL", "NEUTRAL", 0.0, "no_price")

        self._prices.append(price)
        self._n += 1

        if self._n < self.MIN_TICKS:
            return SignalVote("FUT_VEL", "NEUTRAL", 0.0, f"warmup({self._n})")

        prices = list(self._prices)
        n = len(prices)

        # ── Short-term velocity: is price moving RIGHT NOW? ──
        short_lb = min(self.SHORT_WINDOW, n - 1)
        if short_lb < 5:
            return SignalVote("FUT_VEL", "NEUTRAL", 0.0, "insufficient")

        short_displacement = prices[-1] - prices[-short_lb]
        abs_short = abs(short_displacement)
        short_vel = short_displacement / short_lb  # pts per tick

        # ── Short-term acceleration: is the move building or fading? ──
        half = short_lb // 2
        if half >= 3:
            vel_recent = (prices[-1] - prices[-half]) / half
            vel_older = (prices[-half] - prices[-short_lb]) / (short_lb - half)
            accel = vel_recent - vel_older
        else:
            accel = 0.0

        # ── Long-term displacement: has the bulk of the move happened? ──
        long_lb = min(self.LONG_WINDOW, n - 1)
        long_displacement = abs(prices[-1] - prices[-long_lb]) if long_lb >= 10 else 0.0

        # ── Decision Logic ──
        # v28 Innovation E: Use ATR-normalized thresholds instead of fixed points.
        flat_thresh, move_thresh, exhaust_thresh = self._thresholds()

        # CASE 1: FLAT — market not moving. Vote NEUTRAL.
        # This naturally reduces the total CE/PE vote count → harder to reach threshold.
        # Stalls happen when 19 directional signals agree but price is flat.
        # This 20th signal voting NEUTRAL breaks that false consensus.
        if abs_short < flat_thresh:
            self._last_vote = SignalVote(
                "FUT_VEL", "NEUTRAL", 0.2,
                f"FLAT: |{short_displacement:+.1f}|<{flat_thresh:.1f} in {short_lb}T"
            )
            return self._last_vote

        # CASE 2: EXHAUSTION — big move already happened AND decelerating.
        # The user's exact requirement: "Make sure you are not entering when the move has happened."
        # Vote NEUTRAL when displacement is large and momentum is fading.
        # This isn't a hard block — it just removes one vote from consensus.
        if long_displacement > exhaust_thresh and accel < self.EXHAUSTION_DECEL:
            self._last_vote = SignalVote(
                "FUT_VEL", "NEUTRAL", 0.15,
                f"EXHAUSTED: {long_displacement:.1f}pts>{exhaust_thresh:.1f}/{long_lb}T, a={accel:.3f}"
            )
            return self._last_vote

        # CASE 3: REAL MOVEMENT — price is moving with conviction.
        # Vote in the direction of movement. Score scales with velocity + acceleration.
        direction = "CE" if short_displacement > 0 else "PE"

        # v28 Innovation E: Score normalized by ATR — displacement relative to volatility.
        # Old formula: base_score = 0.40 + abs_short * 0.04 (fixed scaling)
        # New formula: base_score = 0.40 + (abs_short / move_thresh) * 0.25
        # At exactly move_threshold: score = 0.65. At 2× move_threshold: score = 0.85.
        # This makes the signal equally sensitive regardless of ATR regime.
        move_ratio = abs_short / move_thresh if move_thresh > 0 else 1.0
        base_score = min(0.40 + move_ratio * 0.25, 0.85)

        # Acceleration bonus: building move gets boosted, fading move gets dampened
        if accel > 0.01:
            # Move is accelerating — this is the IDEAL entry moment
            accel_bonus = min(accel * 3.0, 0.10)
            score = min(base_score + accel_bonus, 0.90)
            reason = f"ACCEL: {direction} {abs_short:.1f}pts/{short_lb}T, a={accel:+.3f}"
        elif accel < -0.01:
            # Move is decelerating but still significant — dampened score
            score = max(base_score - 0.10, 0.40)
            reason = f"DECEL: {direction} {abs_short:.1f}pts/{short_lb}T, a={accel:+.3f}"
        else:
            # Steady movement — good, use base score
            score = base_score
            reason = f"MOVING: {direction} {abs_short:.1f}pts/{short_lb}T"

        self._last_vote = SignalVote("FUT_VEL", direction, score, reason)
        return self._last_vote

    @property
    def displacement(self) -> float:
        """Short-term displacement for external use."""
        if len(self._prices) < 10:
            return 0.0
        lb = min(self.SHORT_WINDOW, len(self._prices) - 1)
        return self._prices[-1] - self._prices[-lb]

    @property
    def is_moving(self) -> bool:
        """Is price moving above flat threshold? v28 Innovation E: ATR-normalized."""
        flat_thresh, _, _ = self._thresholds()
        return abs(self.displacement) >= flat_thresh

    @property
    def is_exhausted(self) -> bool:
        """Has a big move already happened? v28 Innovation E: ATR-normalized."""
        if len(self._prices) < 15:
            return False
        _, _, exhaust_thresh = self._thresholds()
        lb = min(self.LONG_WINDOW, len(self._prices) - 1)
        return abs(self._prices[-1] - self._prices[-lb]) > exhaust_thresh

    def current_vote(self) -> SignalVote:
        return self._last_vote


class MomentumBurstDetector:
    """
    CUSUM-based momentum burst detector — detects the START of a real directional move.

    The key insight: stall kills happen when signals fire during NOISE, not during
    a genuine momentum burst. CUSUM (Cumulative Sum Control Chart) is the gold-standard
    algorithm for sequential change-point detection (Page, 1954).

    How it works:
      - Tracks cumulative deviations of price returns from a neutral reference (k)
      - When cumulative sum exceeds threshold (h), a momentum burst is detected
      - Resets when the opposite direction takes over
      - Direction of burst tells you which way the REAL money is flowing

    Parameters tuned for NIFTY 50 at ~0.3s tick intervals (~3.6 ticks/sec):
      - k (allowance): 0.15 pts/tick — movements smaller than this are noise
      - h (threshold): 2.0 pts cumulative — need sustained 2+ pts of directional pressure
      - These are CONSERVATIVE: only fires on genuine moves, not noise

    Research: Page (1954), Lucas & Crosier (1982), Hawkins & Olwell (1998).
    Used in industrial quality control and adapted for financial change detection.
    """
    # CUSUM parameters — tuned for NIFTY at 3.6Hz
    K_ALLOWANCE = 0.15   # minimum per-tick change to count (noise filter)
    H_THRESHOLD = 2.0    # cumulative threshold to declare burst
    DECAY = 0.97         # slight decay to prevent infinite accumulation during slow trends

    def __init__(self):
        self._cusum_up = 0.0     # cumulative upward pressure
        self._cusum_dn = 0.0     # cumulative downward pressure
        self._last_price = 0.0
        self._burst_dir = "NEUTRAL"
        self._burst_strength = 0.0
        self._tick_count = 0

    def reset(self):
        self._cusum_up = 0.0
        self._cusum_dn = 0.0
        self._last_price = 0.0
        self._burst_dir = "NEUTRAL"
        self._burst_strength = 0.0
        self._tick_count = 0

    def update(self, price: float):
        """Feed a new price tick. Call every tick (~0.3s)."""
        if self._last_price <= 0:
            self._last_price = price
            return

        self._tick_count += 1
        delta = price - self._last_price
        self._last_price = price

        # Apply decay to prevent infinite accumulation
        self._cusum_up *= self.DECAY
        self._cusum_dn *= self.DECAY

        # Accumulate deviations above/below noise threshold
        self._cusum_up = max(0.0, self._cusum_up + delta - self.K_ALLOWANCE)
        self._cusum_dn = max(0.0, self._cusum_dn - delta - self.K_ALLOWANCE)

        # Detect burst
        if self._cusum_up >= self.H_THRESHOLD:
            self._burst_dir = "CE"
            self._burst_strength = min(self._cusum_up / (self.H_THRESHOLD * 2), 1.0)
        elif self._cusum_dn >= self.H_THRESHOLD:
            self._burst_dir = "PE"
            self._burst_strength = min(self._cusum_dn / (self.H_THRESHOLD * 2), 1.0)
        else:
            self._burst_dir = "NEUTRAL"
            self._burst_strength = max(self._cusum_up, self._cusum_dn) / self.H_THRESHOLD

    @property
    def direction(self) -> str:
        """Current burst direction: "CE", "PE", or "NEUTRAL"."""
        return self._burst_dir

    @property
    def strength(self) -> float:
        """0.0 to 1.0 — how strong the current burst is."""
        return self._burst_strength

    def confirms(self, proposed_dir: str) -> bool:
        """Does the CUSUM burst confirm the proposed entry direction?

        Returns True if:
          - Burst is in same direction, OR
          - No burst detected (neutral — don't block on absence of signal)
        Returns False if:
          - Active burst in OPPOSITE direction (strongest rejection signal)
        """
        if self._burst_dir == "NEUTRAL":
            return True  # no burst either way — don't block
        return self._burst_dir == proposed_dir

class KaufmanEfficiencyRatio:
    """
    Kaufman's Efficiency Ratio (ER) — the cleanest measure of "is this move real?"

    ER = |net displacement| / |total path traveled|
      - ER → 1.0: perfectly directional (every tick moves the same way)
      - ER → 0.0: pure noise (random walk, total path >> net displacement)

    This is THE number that distinguishes real momentum from noise.
    Research: Kaufman (1995), adapted for tick-level data.

    For NIFTY at 3.6Hz:
      - ER < 0.15: pure noise, signals are unreliable
      - ER 0.15-0.40: transition zone, momentum may be developing
      - ER > 0.40: directional move confirmed

    Implementation: O(1) per tick using rolling window.
    """
    PERIOD = 60        # 60 ticks ~ 17 seconds of context
    NOISE_THRESHOLD = 0.12  # below this = noise, don't trust any signal
    BURST_THRESHOLD = 0.35  # above this = confirmed directional move

    def __init__(self):
        self._prices: deque = deque(maxlen=self.PERIOD + 1)
        self._abs_changes: deque = deque(maxlen=self.PERIOD)
        self._running_sum = 0.0  # O(1) running total of |changes|
        self._er = 0.0

    def reset(self):
        self._prices.clear()
        self._abs_changes.clear()
        self._running_sum = 0.0
        self._er = 0.0

    def update(self, price: float):
        """Feed a tick price. Call every tick. O(1) per tick."""
        if self._prices:
            abs_change = abs(price - self._prices[-1])
            # If deque is full, the oldest element will be evicted — subtract it
            if len(self._abs_changes) == self.PERIOD:
                self._running_sum -= self._abs_changes[0]
            self._abs_changes.append(abs_change)
            self._running_sum += abs_change
        self._prices.append(price)

        if len(self._prices) < self.PERIOD + 1:
            self._er = 0.5  # insufficient data — neutral
            return

        # Net displacement: |current - PERIOD ago|
        net_displacement = abs(self._prices[-1] - self._prices[-1 - self.PERIOD])

        if self._running_sum > 0:
            self._er = net_displacement / self._running_sum
        else:
            self._er = 0.0

    @property
    def er(self) -> float:
        """Current Efficiency Ratio (0.0 to 1.0)."""
        return self._er

    @property
    def is_noise(self) -> bool:
        """Is the market currently in noise mode? If so, don't trust signals."""
        return self._er < self.NOISE_THRESHOLD

    @property
    def is_directional(self) -> bool:
        """Is there a confirmed directional move?"""
        return self._er > self.BURST_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
def apply_freshness_decay(signals: List[SignalVote]) -> List[SignalVote]:
    now = time.monotonic(); result = []
    for s in signals:
        age = now - s.ts
        if age > SIGNAL_FRESHNESS_SEC and s.direction != "NEUTRAL":
            decay_factor = max(0.5, 1.0 - (age - SIGNAL_FRESHNESS_SEC) / 300.0)
            decayed = SignalVote(name=s.name, direction=s.direction, score=s.score * decay_factor, reason=f"{s.reason} [stale]", ts=s.ts)
            result.append(decayed)
        else:
            result.append(s)
    return result

class ConvictionTransitionDetector:
    """v17.3: Added post-exit "base pattern" requirement.

    After any trade exit, the engine can't immediately re-enter the same direction.
    It must see conviction DROP below 0.55 ("base") and then RISE back above floor
    ("confirmation"). This replaces arbitrary win cooldown with a signal-quality gate.

    Flow after exit:
      1. _post_exit = True, _exit_dir = direction of exited trade
      2. While _post_exit:
         - If direction CHANGES → clear post_exit (genuinely new signal)
         - If score drops below 0.55 → _base_seen = True (signal weakened)
         - If score rises above floor AND _base_seen → clear post_exit (base pattern done)
         - Otherwise → return STALE (block re-entry)
    """
    BASE_THRESHOLD = 0.55  # score must drop below this before re-entry

    def __init__(self):
        self._last_direction = "NEUTRAL"; self._sustained_ticks = 0; self._last_transition_tick = 0
        self._tick = 0; self._last_score = 0.0
        self._post_exit = False; self._exit_dir = "NEUTRAL"; self._base_seen = False

    def reset(self):
        self._last_direction = "NEUTRAL"; self._sustained_ticks = 0; self._last_transition_tick = 0
        self._tick = 0; self._last_score = 0.0
        self._post_exit = False; self._exit_dir = "NEUTRAL"; self._base_seen = False

    def soft_reset(self):
        self._last_transition_tick = self._tick; self._sustained_ticks = 0

    def mark_exit(self, direction: str):
        """Called after any trade exit. Requires base pattern before same-direction re-entry."""
        self._post_exit = True; self._exit_dir = direction; self._base_seen = False

    def check(self, direction: str, score: float) -> Tuple[bool, str]:
        self._tick += 1

        # v17.3: Post-exit base pattern check
        if self._post_exit:
            if direction != self._exit_dir:
                # New direction entirely — genuinely fresh signal
                self._post_exit = False; self._base_seen = False
            elif score < self.BASE_THRESHOLD:
                # Signal weakened — base forming
                self._base_seen = True
            elif self._base_seen and score >= self.BASE_THRESHOLD:
                # Base seen + score recovered → base pattern complete
                self._post_exit = False; self._base_seen = False
                self._last_transition_tick = self._tick  # mark as fresh
            else:
                # Same direction, score still high, no base seen → block
                return False, f"POST_EXIT_no_base(score={score:.2f},need_drop<{self.BASE_THRESHOLD})"

        if direction != self._last_direction:
            self._last_direction = direction; self._sustained_ticks = 1; self._last_transition_tick = self._tick
            self._last_score = score
            return True, f"FRESH_transition"
        self._sustained_ticks += 1
        if score >= self._last_score + 0.05:
            self._last_transition_tick = self._tick; self._sustained_ticks = 1; self._last_score = score
            return True, f"RENEWED_score"
        self._last_score = max(self._last_score, score)
        ticks_since = self._tick - self._last_transition_tick
        if ticks_since > CONVICTION_STALE_TICKS: return False, f"STALE_{self._sustained_ticks}ticks"
        return True, f"recent_transition"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1: FUTURES PRICE vs TIME-WEIGHTED VWAP
# ─────────────────────────────────────────────────────────────────────────────
class FuturesVWAPSignal:
    def __init__(self):
        self._weighted_sum = 0.0
        self._total_time   = 0.0
        self._last_ts      = 0.0
        self._last_price   = 0.0
        self._session_start = 0.0
        self.vwap          = float('nan')

    def reset(self):
        self._weighted_sum = 0.0
        self._total_time   = 0.0
        self._last_ts      = 0.0
        self._last_price   = 0.0
        self._session_start = time.monotonic()
        self.vwap          = float('nan')

    def update(self, price: float) -> SignalVote:
        if price <= 0:
            return SignalVote("VWAP", "NEUTRAL", 0.0, "no_price")

        now = time.monotonic()

        if self._last_ts == 0.0:
            self._last_ts    = now
            self._last_price = price
            self._session_start = now
            return SignalVote("VWAP", "NEUTRAL", 0.0, "first_tick")

        dt = now - self._last_ts
        if dt > 60.0:
            dt = 60.0

        self._weighted_sum += self._last_price * dt
        self._total_time   += dt
        self._last_ts       = now
        self._last_price    = price

        session_age = now - self._session_start
        if session_age < FUTURES_VWAP_MIN_SECS or self._total_time < 30:
            return SignalVote("VWAP", "NEUTRAL", 0.0, f"warming({session_age:.0f}s)")

        self.vwap = self._weighted_sum / self._total_time
        dev = (price - self.vwap) / self.vwap

        # v17.3: raised from 0.0004 (4bps = ~1pt on NIFTY 23000 = pure noise)
        # 0.0015 = ~3.5pts — needs genuine deviation from VWAP to vote
        if dev > 0.0015:
            score = min(0.5 + dev * 200, 0.95)
            return SignalVote("VWAP", "CE", score, f"fut>{self.vwap:.0f}({dev:+.3%})")
        elif dev < -0.0015:
            score = min(0.5 + abs(dev) * 200, 0.95)
            return SignalVote("VWAP", "PE", score, f"fut<{self.vwap:.0f}({dev:+.3%})")
        else:
            return SignalVote("VWAP", "NEUTRAL", 0.3, f"at_vwap({dev:+.4%})")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 2: RSI ON 1-MINUTE CANDLE CLOSES
# ─────────────────────────────────────────────────────────────────────────────
class FuturesRSISignal:
    def __init__(self, period: int = RSI_PERIOD):
        self._period    = period
        self._closes: deque = deque(maxlen=period * 3)
        self._avg_gain  = 0.0
        self._avg_loss  = 0.0
        self._initialized = False
        self.rsi        = 50.0

    def reset(self):
        self._closes.clear()
        self._avg_gain  = 0.0
        self._avg_loss  = 0.0
        self._initialized = False
        self.rsi        = 50.0

    def on_candle(self, candle: Candle) -> SignalVote:
        self._closes.append(candle.close)

        if len(self._closes) < self._period + 1:
            return SignalVote("RSI", "NEUTRAL", 0.0, f"warming({len(self._closes)}/{self._period+1})")

        if not self._initialized:
            closes = list(self._closes)
            gains  = []
            losses = []
            for i in range(1, self._period + 1):
                delta = closes[i] - closes[i - 1]
                gains.append(max(delta, 0.0))
                losses.append(max(-delta, 0.0))
            self._avg_gain = sum(gains) / self._period
            self._avg_loss = sum(losses) / self._period
            self._initialized = True
        else:
            closes = list(self._closes)
            delta  = closes[-1] - closes[-2]
            gain   = max(delta, 0.0)
            loss   = max(-delta, 0.0)
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

        if self._avg_loss == 0:
            self.rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self.rsi = 100.0 - (100.0 / (1.0 + rs))

        if self.rsi > RSI_BULL_THRESHOLD:
            score = min(0.4 + (self.rsi - RSI_BULL_THRESHOLD) / 40, 0.9)
            return SignalVote("RSI", "CE", score, f"RSI={self.rsi:.0f}>{RSI_BULL_THRESHOLD}")
        elif self.rsi < RSI_BEAR_THRESHOLD:
            score = min(0.4 + (RSI_BEAR_THRESHOLD - self.rsi) / 40, 0.9)
            return SignalVote("RSI", "PE", score, f"RSI={self.rsi:.0f}<{RSI_BEAR_THRESHOLD}")
        else:
            return SignalVote("RSI", "NEUTRAL", 0.3, f"RSI={self.rsi:.0f}")

    def current_vote(self) -> SignalVote:
        if self.rsi > RSI_BULL_THRESHOLD:
            score = min(0.4 + (self.rsi - RSI_BULL_THRESHOLD) / 40, 0.9)
            return SignalVote("RSI", "CE", score, f"RSI={self.rsi:.0f}>{RSI_BULL_THRESHOLD}")
        elif self.rsi < RSI_BEAR_THRESHOLD:
            score = min(0.4 + (RSI_BEAR_THRESHOLD - self.rsi) / 40, 0.9)
            return SignalVote("RSI", "PE", score, f"RSI={self.rsi:.0f}<{RSI_BEAR_THRESHOLD}")
        return SignalVote("RSI", "NEUTRAL", 0.3, f"RSI={self.rsi:.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 3: FUTURES MOMENTUM (EMA CROSSOVER)
# ─────────────────────────────────────────────────────────────────────────────
class FuturesMomentumSignal:
    def __init__(self):
        self._ema9  = float('nan'); self._ema21 = float('nan')
        self._count = 0

    def reset(self):
        self._ema9  = float('nan'); self._ema21 = float('nan')
        self._count = 0

    def update(self, price: float) -> SignalVote:
        if price <= 0: return SignalVote("MOMENTUM", "NEUTRAL", 0.0, "no_price")
        a9  = 2 / (9  + 1); a21 = 2 / (21 + 1)
        if math.isnan(self._ema9):
            self._ema9 = self._ema21 = price
        else:
            self._ema9  = a9  * price + (1 - a9)  * self._ema9
            self._ema21 = a21 * price + (1 - a21) * self._ema21
        self._count += 1
        if self._count < 21: return SignalVote("MOMENTUM", "NEUTRAL", 0.0, f"warming({self._count})")

        fast_above = self._ema9 > self._ema21
        separation = abs(self._ema9 - self._ema21) / self._ema21

        if fast_above:
            score = min(0.4 + separation * 500, 0.9)
            return SignalVote("MOMENTUM", "CE", score, f"EMA9({self._ema9:.0f})>EMA21({self._ema21:.0f})")
        else:
            score = min(0.4 + separation * 500, 0.9)
            return SignalVote("MOMENTUM", "PE", score, f"EMA9({self._ema9:.0f})<EMA21({self._ema21:.0f})")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 4: OPTIONS OI STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
class OptionsOIStructure:
    def __init__(self):
        self._snapshots: deque = deque(maxlen=10)
        self._last_snap_ts: float = 0.0
        self._last_result = SignalVote("OPT_OI", "NEUTRAL", 0.0, "cold")

    def reset(self):
        self._snapshots.clear(); self._last_snap_ts = 0.0
        self._last_result = SignalVote("OPT_OI", "NEUTRAL", 0.0, "cold")

    def update(self, snap) -> Tuple[SignalVote, str]:
        now = time.monotonic()
        if now - self._last_snap_ts < OI_SNAP_INTERVAL:
            return self._last_result, "UNCLEAR"

        self._last_snap_ts = now
        atm = snap.atm_strike
        calls = snap.call_strikes
        puts = snap.put_strikes
        total_ce = snap.total_call_oi
        total_pe = snap.total_put_oi

        if total_ce <= 0 or total_pe <= 0:
            return SignalVote("OPT_OI", "NEUTRAL", 0.0, "no_oi"), "UNCLEAR"

        pcr = total_pe / total_ce
        calls_d = {s: sd for s, sd in calls.items()}
        puts_d  = {s: sd for s, sd in puts.items()}
        max_ce_s  = max(calls_d, key=lambda s: calls_d[s].oi) if calls_d else atm
        max_pe_s  = max(puts_d,  key=lambda s: puts_d[s].oi)  if puts_d  else atm

        smart_money = self._detect_strategy(snap, atm, calls, puts, pcr)

        self._snapshots.append({'ts': now, 'pcr': pcr, 'fut': snap.spot, 'max_ce': max_ce_s, 'max_pe': max_pe_s})

        pcr_shift = 0.0; pcr_velocity = 0.0
        if len(self._snapshots) >= 2: pcr_shift = pcr - self._snapshots[-2]['pcr']
        if len(self._snapshots) >= 3:
            oldest = self._snapshots[0]
            dt_mins = (now - oldest['ts']) / 60.0
            if dt_mins > 0.5: pcr_velocity = (pcr - oldest['pcr']) / dt_mins

        fut = snap.spot
        above_ce_wall = fut > max_ce_s + STRIKE_STEP
        below_pe_wall = fut < max_pe_s - STRIKE_STEP

        velocity_boost = min(abs(pcr_velocity) * 0.5, 0.15)

        if pcr > PCR_EXTREME_BULL:
            score = min(0.5 + (pcr - PCR_EXTREME_BULL) * 1.5 + velocity_boost, 0.90)
            vote = SignalVote("OPT_OI", "CE", score, f"PCR={pcr:.2f}>{PCR_EXTREME_BULL} shift={pcr_shift:+.2f}")
        elif pcr < PCR_EXTREME_BEAR:
            score = min(0.5 + (PCR_EXTREME_BEAR - pcr) * 1.5 + velocity_boost, 0.90)
            vote = SignalVote("OPT_OI", "PE", score, f"PCR={pcr:.2f}<{PCR_EXTREME_BEAR} shift={pcr_shift:+.2f}")
        elif pcr_shift > PCR_SHIFT_TRIGGER:
            vote = SignalVote("OPT_OI", "PE", 0.55 + velocity_boost, f"PCR_rising_fast {pcr:.2f} shift={pcr_shift:+.2f}")
        elif pcr_shift < -PCR_SHIFT_TRIGGER:
            vote = SignalVote("OPT_OI", "CE", 0.55 + velocity_boost, f"PCR_falling_fast {pcr:.2f} shift={pcr_shift:+.2f}")
        elif above_ce_wall:
            vote = SignalVote("OPT_OI", "CE", 0.60, f"Broke_CE_wall@{max_ce_s:.0f} PCR={pcr:.2f}")
        elif below_pe_wall:
            vote = SignalVote("OPT_OI", "PE", 0.60, f"Broke_PE_wall@{max_pe_s:.0f} PCR={pcr:.2f}")
        else:
            vote = SignalVote("OPT_OI", "NEUTRAL", 0.25, f"PCR={pcr:.2f} neutral")

        self._last_result = vote
        return vote, smart_money

    def _detect_strategy(self, snap, atm, calls, puts, pcr) -> str:
        levels = range(-STRIKES_AROUND_ATM, STRIKES_AROUND_ATM + 1)
        ce_oi = {}; pe_oi = {}
        for o in levels:
            strike = atm + o * STRIKE_STEP
            c = calls.get(strike); p = puts.get(strike)
            ce_oi[o] = c.oi if c is not None else 0.0
            pe_oi[o] = p.oi if p is not None else 0.0

        total_ce = sum(ce_oi.values()) or 1
        total_pe = sum(pe_oi.values()) or 1
        ce_f = {k: v / total_ce for k, v in ce_oi.items()}
        pe_f = {k: v / total_pe for k, v in pe_oi.items()}

        ce_wings = ce_f.get(2, 0) + ce_f.get(3, 0)
        pe_wings = pe_f.get(-2, 0) + pe_f.get(-3, 0)
        wing_sym = 1 - abs(ce_wings - pe_wings) / max(ce_wings + pe_wings, 0.01)
        if (ce_wings + pe_wings) / 2 > 0.35 and wing_sym > 0.6: return "CONDOR"

        ce_atm = ce_f.get(0, 0); pe_atm = pe_f.get(0, 0)
        if ce_atm > 0.25 and pe_atm > 0.25 and abs(ce_atm - pe_atm) < 0.1: return "STRADDLE"
        if pcr > 1.2 and ce_f.get(0, 0) + ce_f.get(1, 0) > 0.4: return "BEAR_CALL_SPREAD"
        if pcr < 0.8 and pe_f.get(0, 0) + pe_f.get(-1, 0) > 0.4: return "BULL_PUT_SPREAD"
        return "UNCLEAR"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 5: PREMIUM VELOCITY vs FUTURES VELOCITY
# ─────────────────────────────────────────────────────────────────────────────
class PremiumVsForwardSignal:
    WINDOW = 10

    def __init__(self):
        self._fut  = deque(maxlen=self.WINDOW + 2)
        self._ce   = deque(maxlen=self.WINDOW + 2)
        self._pe   = deque(maxlen=self.WINDOW + 2)
        self._ts   = deque(maxlen=self.WINDOW + 2)
        self._n    = 0

    def reset(self):
        self._fut.clear(); self._ce.clear(); self._pe.clear(); self._ts.clear()
        self._n = 0

    def update(self, futures: float, ce_ltp: float, pe_ltp: float) -> SignalVote:
        self._fut.append(futures)
        self._ce.append(ce_ltp  if ce_ltp  > 0 else float('nan'))
        self._pe.append(pe_ltp  if pe_ltp  > 0 else float('nan'))
        self._ts.append(time.monotonic())
        self._n += 1

        if self._n < self.WINDOW:
            return SignalVote("PREM_VEL", "NEUTRAL", 0.0, f"warm({self._n})")

        dt = self._ts[-1] - self._ts[-self.WINDOW]
        if dt < 0.2:
            return SignalVote("PREM_VEL", "NEUTRAL", 0.0, "dt_zero")

        f_arr = list(self._fut)[-self.WINDOW:]
        c_arr = [x for x in list(self._ce)[-self.WINDOW:] if not math.isnan(x)]
        p_arr = [x for x in list(self._pe)[-self.WINDOW:] if not math.isnan(x)]

        if len(c_arr) < 2 or len(p_arr) < 2:
            return SignalVote("PREM_VEL", "NEUTRAL", 0.0, "no_prem")

        f_vel_pts = (f_arr[-1] - f_arr[0]) / dt
        c_vel_pct = (c_arr[-1] - c_arr[0]) / c_arr[0] / dt if c_arr[0] > 0 else 0
        p_vel_pct = (p_arr[-1] - p_arr[0]) / p_arr[0] / dt if p_arr[0] > 0 else 0

        abs_c_vel = abs(c_vel_pct)
        abs_p_vel = abs(p_vel_pct)

        if f_vel_pts > 0.5:
            if c_vel_pct > 0 and abs_c_vel > (abs_p_vel * 1.5) and abs_c_vel > 0.001:
                score = min(0.65 + abs_c_vel * 80, 0.95)
                return SignalVote("PREM_VEL", "CE", score, f"ASYM_ACCEL: CE+{c_vel_pct:+.2%}/s > PE-{abs_p_vel:+.2%}/s")
            elif c_vel_pct > 0.0008:
                return SignalVote("PREM_VEL", "CE", 0.60, f"CE_confirmed c={c_vel_pct:+.2%}/s")

        elif f_vel_pts < -0.5:
            if p_vel_pct > 0 and abs_p_vel > (abs_c_vel * 1.5) and abs_p_vel > 0.001:
                score = min(0.65 + abs_p_vel * 80, 0.95)
                return SignalVote("PREM_VEL", "PE", score, f"ASYM_ACCEL: PE+{p_vel_pct:+.2%}/s > CE-{abs_c_vel:+.2%}/s")
            elif p_vel_pct > 0.0008:
                return SignalVote("PREM_VEL", "PE", 0.60, f"PE_confirmed p={p_vel_pct:+.2%}/s")

        return SignalVote("PREM_VEL", "NEUTRAL", 0.25, f"mixed c={c_vel_pct:+.3%} p={p_vel_pct:+.3%}")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 6: VWAP RETEST BOUNCE
# ─────────────────────────────────────────────────────────────────────────────
class VWAPRetestSignal:
    TOUCH_THRESHOLD = 0.0003
    BOUNCE_MIN      = 0.0004
    MIN_TICKS_ABOVE = 5

    def __init__(self):
        self._vwap_ref: Optional[FuturesVWAPSignal] = None
        self._state     = "NEUTRAL"
        self._ticks_above = 0
        self._ticks_below = 0
        self._retest_vwap = float('nan')
        self._n         = 0

    def reset(self):
        self._state = "NEUTRAL"; self._ticks_above = 0; self._ticks_below = 0
        self._retest_vwap = float('nan'); self._n = 0

    def set_vwap_ref(self, vwap_signal: FuturesVWAPSignal):
        self._vwap_ref = vwap_signal

    def update(self, price: float) -> SignalVote:
        self._n += 1
        if self._vwap_ref is not None and not math.isnan(self._vwap_ref.vwap):
            vwap = self._vwap_ref.vwap
        else: return SignalVote("VWAP_RETEST", "NEUTRAL", 0.0, f"no_vwap({self._n})")

        if self._n < 30: return SignalVote("VWAP_RETEST", "NEUTRAL", 0.0, f"warm({self._n})")

        dev = (price - vwap) / vwap
        if dev > 0: self._ticks_above += 1; self._ticks_below  = 0
        else: self._ticks_below += 1; self._ticks_above  = 0

        if self._state == "NEUTRAL":
            if self._ticks_above >= self.MIN_TICKS_ABOVE: self._state = "BROKEN_ABOVE"
            elif self._ticks_below >= self.MIN_TICKS_ABOVE: self._state = "BROKEN_BELOW"

        elif self._state == "BROKEN_ABOVE":
            if abs(dev) <= self.TOUCH_THRESHOLD: self._state = "RETEST_UP"; self._retest_vwap = vwap
            elif dev < -self.BOUNCE_MIN: self._state = "BROKEN_BELOW"

        elif self._state == "BROKEN_BELOW":
            if abs(dev) <= self.TOUCH_THRESHOLD: self._state = "RETEST_DN"; self._retest_vwap = vwap
            elif dev > self.BOUNCE_MIN: self._state = "BROKEN_ABOVE"

        elif self._state == "RETEST_UP":
            if dev > self.BOUNCE_MIN:
                score = min(0.65 + dev * 300, 0.92)
                self._state = "BROKEN_ABOVE"
                return SignalVote("VWAP_RETEST", "CE", score, f"Bounce@{self._retest_vwap:.0f} dev={dev:+.3%}")
            elif dev < -self.BOUNCE_MIN * 2: self._state = "BROKEN_BELOW"

        elif self._state == "RETEST_DN":
            if dev < -self.BOUNCE_MIN:
                score = min(0.65 + abs(dev) * 300, 0.92)
                self._state = "BROKEN_BELOW"
                return SignalVote("VWAP_RETEST", "PE", score, f"Reject@{self._retest_vwap:.0f} dev={dev:+.3%}")
            elif dev > self.BOUNCE_MIN * 2: self._state = "BROKEN_ABOVE"

        return SignalVote("VWAP_RETEST", "NEUTRAL", 0.2, f"state={self._state} dev={dev:+.3%}")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 7: MAX PAIN + OI WALL
# ─────────────────────────────────────────────────────────────────────────────
class MaxPainOIWall:
    def __init__(self):
        self._last_update = 0.0; self._last_result = SignalVote("MAX_PAIN", "NEUTRAL", 0.0, "cold")

    def reset(self):
        self._last_update = 0.0; self._last_result = SignalVote("MAX_PAIN", "NEUTRAL", 0.0, "cold")

    def update(self, snap) -> SignalVote:
        now = time.monotonic()
        if now - self._last_update < OI_SNAP_INTERVAL: return self._last_result
        self._last_update = now

        spot = snap.spot; atm = snap.atm_strike; calls = snap.call_strikes; puts = snap.put_strikes
        if not calls or not puts:
            self._last_result = SignalVote("MAX_PAIN", "NEUTRAL", 0.0, "no_data"); return self._last_result

        max_pain = self._calc_max_pain(calls, puts)
        calls_d = {s: sd for s, sd in calls.items()}; puts_d = {s: sd for s, sd in puts.items()}

        ce_wall = max(calls_d, key=lambda s: calls_d[s].oi) if calls_d else atm
        pe_wall = max(puts_d,  key=lambda s: puts_d[s].oi)  if puts_d  else atm

        dist_to_pain = spot - max_pain; dist_to_ce = ce_wall - spot; dist_to_pe = spot - pe_wall
        reason = f"MaxPain={max_pain:.0f} CE_wall={ce_wall:.0f} PE_wall={pe_wall:.0f}"

        if 0 < dist_to_pain < 200 and dist_to_ce > 0:
            score = min(0.50 + (200 - dist_to_pain) / 400, 0.80)
            self._last_result = SignalVote("MAX_PAIN", "CE", score, f"AbovePain+{dist_to_pain:.0f} BelowCEwall {reason}")
        elif -200 < dist_to_pain < 0 and dist_to_pe > 0:
            score = min(0.50 + (200 + dist_to_pain) / 400, 0.80)
            self._last_result = SignalVote("MAX_PAIN", "PE", score, f"BelowPain{dist_to_pain:.0f} AbovePEwall {reason}")
        elif abs(dist_to_pain) < 50:
            self._last_result = SignalVote("MAX_PAIN", "NEUTRAL", 0.2, f"AtMaxPain±{dist_to_pain:.0f} {reason}")
        else: self._last_result = SignalVote("MAX_PAIN", "NEUTRAL", 0.25, reason)
        return self._last_result

    @staticmethod
    def _calc_max_pain(calls, puts) -> float:
        all_strikes = set()
        for s, _ in calls.items(): all_strikes.add(s)
        for s, _ in puts.items():  all_strikes.add(s)
        if not all_strikes: return 0.0
        min_loss = float('inf'); max_pain = 0.0
        for expiry in sorted(all_strikes):
            total_loss = 0.0
            for strike, sd in calls.items():
                if expiry > strike: total_loss += (expiry - strike) * sd.oi
            for strike, sd in puts.items():
                if expiry < strike: total_loss += (strike - expiry) * sd.oi
            if total_loss < min_loss: min_loss = total_loss; max_pain = expiry
        return max_pain

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 8: FUTURES PRICE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
class FuturesPriceStructure:
    PIVOT_WINDOW = 5; MIN_PIVOTS = 4

    def __init__(self):
        self._prices = deque(maxlen=50); self._highs = deque(maxlen=8); self._lows = deque(maxlen=8)
        self._n = 0; self._last_check = 0

    def reset(self):
        self._prices.clear(); self._highs.clear(); self._lows.clear(); self._n = 0; self._last_check = 0

    def update(self, price: float) -> SignalVote:
        self._prices.append(price); self._n += 1; self._last_check += 1
        if self._n < self.PIVOT_WINDOW * 2 + 2: return SignalVote("STRUCTURE", "NEUTRAL", 0.0, f"warm({self._n})")
        if self._last_check >= 3: self._last_check = 0; self._detect_pivots()

        if len(self._highs) < 2 or len(self._lows) < 2: return SignalVote("STRUCTURE", "NEUTRAL", 0.2, "collecting_pivots")
        highs = list(self._highs); lows = list(self._lows)
        hh = highs[-1] > highs[-2]; hl = lows[-1] > lows[-2]; lh = highs[-1] < highs[-2]; ll = lows[-1] < lows[-2]

        if hh and hl:
            strength = min((highs[-1] - highs[-2]) / highs[-2] * 500, 1.0)
            return SignalVote("STRUCTURE", "CE", min(0.55 + strength * 0.3, 0.88), f"HH+HL h={highs[-1]:.0f}>{highs[-2]:.0f}")
        elif lh and ll:
            strength = min((highs[-2] - highs[-1]) / highs[-2] * 500, 1.0)
            return SignalVote("STRUCTURE", "PE", min(0.55 + strength * 0.3, 0.88), f"LH+LL h={highs[-1]:.0f}<{highs[-2]:.0f}")
        elif lh and not ll and not hl: return SignalVote("STRUCTURE", "PE", 0.45, f"LH_only")
        elif hl and not hh and not lh: return SignalVote("STRUCTURE", "CE", 0.45, f"HL_only")
        return SignalVote("STRUCTURE", "NEUTRAL", 0.25, "range")

    def _detect_pivots(self):
        prices = list(self._prices); w = self.PIVOT_WINDOW
        if len(prices) < w * 2 + 1: return
        mid = len(prices) // 2
        window = prices[mid - w: mid + w + 1]
        if not window: return
        candidate = prices[mid]
        if candidate == max(window):
            if not self._highs or candidate != self._highs[-1]: self._highs.append(candidate)
        if candidate == min(window):
            if not self._lows or candidate != self._lows[-1]: self._lows.append(candidate)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 9: MACD HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────
class MACDHistogramSignal:
    FAST = 12; SLOW = 26; SIG = 9

    def __init__(self):
        self._ema_fast = float('nan'); self._ema_slow = float('nan'); self._ema_sig = float('nan')
        self._n = 0; self.histogram = 0.0; self.macd_line = 0.0

    def reset(self):
        self._ema_fast = self._ema_slow = self._ema_sig = float('nan')
        self._n = 0; self.histogram = 0.0; self.macd_line = 0.0

    def on_candle(self, candle: Candle) -> SignalVote:
        c = candle.close
        af = 2 / (self.FAST + 1); as_ = 2 / (self.SLOW + 1); ag = 2 / (self.SIG + 1)
        if math.isnan(self._ema_fast):
            self._ema_fast = self._ema_slow = c; return SignalVote("MACD_HIST", "NEUTRAL", 0.0, "init")
        self._ema_fast = af * c + (1 - af) * self._ema_fast
        self._ema_slow = as_ * c + (1 - as_) * self._ema_slow
        self.macd_line = self._ema_fast - self._ema_slow
        if math.isnan(self._ema_sig): self._ema_sig = self.macd_line
        else: self._ema_sig = ag * self.macd_line + (1 - ag) * self._ema_sig

        self._n += 1
        if self._n < self.SLOW + 3: return SignalVote("MACD_HIST", "NEUTRAL", 0.0, f"warming")

        prev_hist = self.histogram
        self.histogram = self.macd_line - self._ema_sig
        rising = self.histogram > prev_hist

        if self.histogram > 0 and rising:
            return SignalVote("MACD_HIST", "CE", min(0.50 + abs(self.histogram) * 20, 0.90), f"hist={self.histogram:+.3f}↑")
        elif self.histogram < 0 and not rising:
            return SignalVote("MACD_HIST", "PE", min(0.50 + abs(self.histogram) * 20, 0.90), f"hist={self.histogram:+.3f}↓")
        elif self.histogram > 0: return SignalVote("MACD_HIST", "CE", 0.40, f"weakening_bull")
        elif self.histogram < 0: return SignalVote("MACD_HIST", "PE", 0.40, f"weakening_bear")
        return SignalVote("MACD_HIST", "NEUTRAL", 0.20, "hist~0")

    def current_vote(self) -> SignalVote:
        if self.histogram > 0: return SignalVote("MACD_HIST", "CE", 0.38, f"hist>0")
        elif self.histogram < 0: return SignalVote("MACD_HIST", "PE", 0.38, f"hist<0")
        return SignalVote("MACD_HIST", "NEUTRAL", 0.20, "hist=0")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 10: SUPERTREND
# ─────────────────────────────────────────────────────────────────────────────
class SupertrendSignal:
    PERIOD = 10; MULT = 3.0
    def __init__(self):
        self._atr_buf = deque(maxlen=self.PERIOD); self._prev_close = float('nan')
        self._upper = float('nan'); self._lower = float('nan'); self._trend = 0
        self._st_line = float('nan'); self._n = 0; self._just_flipped = False

    def reset(self):
        self._atr_buf.clear(); self._prev_close = float('nan'); self._upper = self._lower = self._st_line = float('nan')
        self._trend = 0; self._n = 0; self._just_flipped = False

    def on_candle(self, candle: Candle) -> SignalVote:
        self._n += 1
        tr = candle.high - candle.low if math.isnan(self._prev_close) else max(candle.high - candle.low, abs(candle.high - self._prev_close), abs(candle.low - self._prev_close))
        self._atr_buf.append(tr)
        prev_close_for_bands = self._prev_close; self._prev_close = candle.close

        if self._n < self.PERIOD: return SignalVote("SUPERTREND", "NEUTRAL", 0.0, "warming")

        atr = float(np.mean(list(self._atr_buf)))
        hl2 = (candle.high + candle.low) / 2.0
        basic_upper = hl2 + self.MULT * atr; basic_lower = hl2 - self.MULT * atr

        if math.isnan(self._upper): self._upper = basic_upper; self._lower = basic_lower
        else:
            self._upper = basic_upper if basic_upper < self._upper or (not math.isnan(prev_close_for_bands) and prev_close_for_bands > self._upper) else self._upper
            self._lower = basic_lower if basic_lower > self._lower or (not math.isnan(prev_close_for_bands) and prev_close_for_bands < self._lower) else self._lower

        prev_trend = self._trend; close = candle.close
        if close > self._upper: self._trend = 1; self._st_line = self._lower
        elif close < self._lower: self._trend = -1; self._st_line = self._upper
        elif self._trend == 0:
            if close > hl2: self._trend = 1; self._st_line = self._lower
            else: self._trend = -1; self._st_line = self._upper

        self._just_flipped = (self._trend != prev_trend and prev_trend != 0)
        dist_pct = abs(close - self._st_line) / close if close > 0 else 0

        if self._trend == 1: return SignalVote("SUPERTREND", "CE", min(0.55 + dist_pct * 50 + (0.10 if self._just_flipped else 0), 0.92), f"ST_bull line={self._st_line:.0f}")
        elif self._trend == -1: return SignalVote("SUPERTREND", "PE", min(0.55 + dist_pct * 50 + (0.10 if self._just_flipped else 0), 0.92), f"ST_bear line={self._st_line:.0f}")
        return SignalVote("SUPERTREND", "NEUTRAL", 0.20, "ST_init")

    def current_vote(self) -> SignalVote:
        if self._trend == 1: return SignalVote("SUPERTREND", "CE", 0.50, f"ST_bull")
        elif self._trend == -1: return SignalVote("SUPERTREND", "PE", 0.50, f"ST_bear")
        return SignalVote("SUPERTREND", "NEUTRAL", 0.15, "ST_uninit")

    @property
    def direction(self) -> str:
        if self._trend == 1: return "CE"
        elif self._trend == -1: return "PE"
        return "NEUTRAL"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 11: BOLLINGER SQUEEZE
# ─────────────────────────────────────────────────────────────────────────────
class BollingerSqueezeSignal:
    BB_PERIOD = 20; BB_MULT = 2.0; KC_MULT = 1.5
    def __init__(self):
        self._closes = deque(maxlen=self.BB_PERIOD); self._trs = deque(maxlen=self.BB_PERIOD)
        self._prev_close = float('nan'); self._in_squeeze = False
        self._fire_direction = "NEUTRAL"; self._fire_ticks = 0; self._n = 0

    def reset(self):
        self._closes.clear(); self._trs.clear(); self._prev_close = float('nan')
        self._in_squeeze = False; self._fire_direction = "NEUTRAL"; self._fire_ticks = 0; self._n = 0

    def on_candle(self, candle: Candle) -> SignalVote:
        self._n += 1; self._closes.append(candle.close)
        tr = candle.high - candle.low if math.isnan(self._prev_close) else max(candle.high - candle.low, abs(candle.high - self._prev_close), abs(candle.low - self._prev_close))
        self._trs.append(tr); self._prev_close = candle.close

        if self._n < self.BB_PERIOD: return SignalVote("BB_SQUEEZE", "NEUTRAL", 0.0, "warming")

        closes = list(self._closes); trs = list(self._trs)
        bb_mid = float(np.mean(closes)); bb_std = float(np.std(closes))
        bb_upper = bb_mid + self.BB_MULT * bb_std; bb_lower = bb_mid - self.BB_MULT * bb_std
        kc_atr  = float(np.mean(trs))
        kc_upper = bb_mid + self.KC_MULT * kc_atr; kc_lower = bb_mid - self.KC_MULT * kc_atr

        prev_squeeze = self._in_squeeze
        self._in_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        if prev_squeeze and not self._in_squeeze:
            self._fire_direction = "CE" if candle.close > bb_mid else "PE"
            self._fire_ticks = 3

        if self._fire_ticks > 0:
            self._fire_ticks -= 1
            return SignalVote("BB_SQUEEZE", self._fire_direction, 0.55 + (self._fire_ticks / 3.0) * 0.30, f"SQ_FIRE")

        if self._in_squeeze: return SignalVote("BB_SQUEEZE", "NEUTRAL", 0.15, f"SQ_ON")
        pct_b = (candle.close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        if pct_b > 0.75: return SignalVote("BB_SQUEEZE", "CE", 0.35, f"%B>0.75")
        elif pct_b < 0.25: return SignalVote("BB_SQUEEZE", "PE", 0.35, f"%B<0.25")
        return SignalVote("BB_SQUEEZE", "NEUTRAL", 0.15, "neutral")

    def current_vote(self) -> SignalVote:
        if self._fire_ticks > 0: return SignalVote("BB_SQUEEZE", self._fire_direction, 0.55, "SQ_FIRE(cached)")
        if self._in_squeeze: return SignalVote("BB_SQUEEZE", "NEUTRAL", 0.15, "SQ_ON(cached)")
        return SignalVote("BB_SQUEEZE", "NEUTRAL", 0.15, "no_squeeze")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 12: ADX TREND FILTER
# ─────────────────────────────────────────────────────────────────────────────
class ADXTrendFilter:
    PERIOD = 14
    def __init__(self):
        self._highs = deque(maxlen=self.PERIOD + 1); self._lows = deque(maxlen=self.PERIOD + 1); self._closes = deque(maxlen=self.PERIOD + 1)
        self._tr_buf = deque(maxlen=self.PERIOD); self._dm_plus_buf = deque(maxlen=self.PERIOD); self._dm_minus_buf = deque(maxlen=self.PERIOD)
        self._dx_buf = deque(maxlen=self.PERIOD); self._adx_history = deque(maxlen=6)
        self.adx = 0.0; self.di_plus = 0.0; self.di_minus = 0.0; self._n = 0

    def reset(self):
        self._highs.clear(); self._lows.clear(); self._closes.clear(); self._tr_buf.clear(); self._dm_plus_buf.clear(); self._dm_minus_buf.clear()
        self._dx_buf.clear(); self._adx_history.clear(); self.adx = self.di_plus = self.di_minus = 0.0; self._n = 0

    def on_candle(self, candle: Candle) -> SignalVote:
        self._n += 1; self._highs.append(candle.high); self._lows.append(candle.low); self._closes.append(candle.close)
        if len(self._highs) < 2: return SignalVote("ADX", "NEUTRAL", 0.0, "init")

        ph = self._highs[-2]; pl = self._lows[-2]; pc = self._closes[-2]
        tr = max(candle.high - candle.low, abs(candle.high - pc), abs(candle.low - pc))
        dm_plus  = max(candle.high - ph, 0) if (candle.high - ph) > (pl - candle.low) else 0
        dm_minus = max(pl - candle.low, 0) if (pl - candle.low) > (candle.high - ph) else 0

        self._tr_buf.append(tr); self._dm_plus_buf.append(dm_plus); self._dm_minus_buf.append(dm_minus)
        if self._n < self.PERIOD + 1: return SignalVote("ADX", "NEUTRAL", 0.0, "warming")

        atr14 = float(np.mean(list(self._tr_buf)))
        if atr14 <= 0: return SignalVote("ADX", "NEUTRAL", 0.15, "atr_zero")

        self.di_plus  = 100 * float(np.mean(list(self._dm_plus_buf)))  / atr14
        self.di_minus = 100 * float(np.mean(list(self._dm_minus_buf))) / atr14
        dx = 100 * abs(self.di_plus - self.di_minus) / (self.di_plus + self.di_minus) if (self.di_plus + self.di_minus) > 0 else 0
        self._dx_buf.append(dx); self.adx = float(np.mean(list(self._dx_buf))); self._adx_history.append(self.adx)

        if self.adx < 20: return SignalVote("ADX", "NEUTRAL", 0.05, f"ADX={self.adx:.0f}<20")
        base_score = min(0.50 + (self.adx - 25) / 100, 0.85) if self.adx >= 25 else 0.42

        if self.di_plus > self.di_minus: return SignalVote("ADX", "CE", min(base_score, 0.92), f"ADX={self.adx:.0f} DI+>DI-")
        elif self.di_minus > self.di_plus: return SignalVote("ADX", "PE", min(base_score, 0.92), f"ADX={self.adx:.0f} DI->DI+")
        return SignalVote("ADX", "NEUTRAL", 0.20, f"ADX={self.adx:.0f} DI_equal")

    def current_vote(self) -> SignalVote:
        if self.adx < 20: return SignalVote("ADX", "NEUTRAL", 0.05, f"ADX<20")
        if self.adx >= 25 and self.di_plus > self.di_minus: return SignalVote("ADX", "CE", min(0.50 + (self.adx-25)/100, 0.90), f"ADX")
        if self.adx >= 25 and self.di_minus > self.di_plus: return SignalVote("ADX", "PE", min(0.50 + (self.adx-25)/100, 0.90), f"ADX")
        return SignalVote("ADX", "NEUTRAL", 0.20, "ADX")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 13: 15-MINUTE BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────
class FifteenMinBreakout:
    CANDLES_PER_BAR = 15; MIN_BARS = 2
    def __init__(self):
        self._min_candles = []; self._completed_bars = deque(maxlen=10)
        self._current_high = float('-inf'); self._current_low  = float('inf'); self._current_open = float('nan')
        self._last_vote = SignalVote("BREAKOUT15", "NEUTRAL", 0.0, "cold")

    def reset(self):
        self._min_candles.clear(); self._completed_bars.clear()
        self._current_high = float('-inf'); self._current_low = float('inf'); self._current_open = float('nan')
        self._last_vote = SignalVote("BREAKOUT15", "NEUTRAL", 0.0, "cold")

    def on_1min_candle(self, candle: Candle) -> Optional[SignalVote]:
        self._min_candles.append(candle)
        if math.isnan(self._current_open): self._current_open = candle.open
        self._current_high = max(self._current_high, candle.high); self._current_low  = min(self._current_low,  candle.low)

        if len(self._min_candles) >= self.CANDLES_PER_BAR:
            bar = Candle(ts=self._min_candles[0].ts, open=self._min_candles[0].open, high=self._current_high, low=self._current_low, close=self._min_candles[-1].close)
            self._completed_bars.append(bar); self._min_candles.clear()
            self._current_high = float('-inf'); self._current_low = float('inf'); self._current_open = float('nan')
            self._last_vote = self._compute(bar.close)
            return self._last_vote
        return None

    def _compute(self, current_price: float) -> SignalVote:
        if len(self._completed_bars) < self.MIN_BARS: return SignalVote("BREAKOUT15", "NEUTRAL", 0.0, "warming")
        bars = list(self._completed_bars); prev = bars[-2]; current_bar = bars[-1]

        if current_bar.close > prev.high:
            pct = (current_bar.close - prev.high) / prev.high
            return SignalVote("BREAKOUT15", "CE", min(0.55 + pct * 1000, 0.93), f"Close>PrevHigh +{pct:.3%}")
        if current_bar.close < prev.low:
            pct = (prev.low - current_bar.close) / prev.low
            return SignalVote("BREAKOUT15", "PE", min(0.55 + pct * 1000, 0.93), f"Close<PrevLow -{pct:.3%}")
        return SignalVote("BREAKOUT15", "NEUTRAL", 0.3, "Inside")

    def current_vote(self) -> SignalVote:
        return self._last_vote

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 14: OPENING RANGE BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────
class OpeningRangeBreakout:
    ORB_MINUTES = 15
    def __init__(self):
        self._orb_high = float('nan'); self._orb_low = float('nan'); self._orb_set = False; self._orb_candles_seen = 0
        self._last_vote = SignalVote("ORB", "NEUTRAL", 0.0, "forming")

    def reset(self):
        self._orb_high = float('nan'); self._orb_low = float('nan'); self._orb_set = False; self._orb_candles_seen = 0
        self._last_vote = SignalVote("ORB", "NEUTRAL", 0.0, "forming")

    def on_1min_candle(self, candle: Candle):
        if self._orb_set: return
        self._orb_candles_seen += 1
        if math.isnan(self._orb_high): self._orb_high = candle.high; self._orb_low = candle.low
        else: self._orb_high = max(self._orb_high, candle.high); self._orb_low  = min(self._orb_low, candle.low)
        if self._orb_candles_seen >= self.ORB_MINUTES: self._orb_set = True

    def update(self, price: float) -> SignalVote:
        if not self._orb_set or math.isnan(self._orb_high): return SignalVote("ORB", "NEUTRAL", 0.0, "forming")
        orb_range = max(self._orb_high - self._orb_low, 1.0)
        if price > self._orb_high:
            self._last_vote = SignalVote("ORB", "CE", min(0.58 + ((price - self._orb_high) / orb_range) * 0.7, 0.93), "Above_ORB")
        elif price < self._orb_low:
            self._last_vote = SignalVote("ORB", "PE", min(0.58 + ((self._orb_low - price) / orb_range) * 0.7, 0.93), "Below_ORB")
        else: self._last_vote = SignalVote("ORB", "NEUTRAL", 0.2, "Inside_ORB")
        return self._last_vote

    def current_vote(self) -> SignalVote:
        return self._last_vote

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 15: FUTURES OI BUILDUP
# ─────────────────────────────────────────────────────────────────────────────
class FuturesOIBuildup:
    SNAP_INTERVAL = 120; MIN_OI_CHANGE = 0.005
    def __init__(self):
        self._snapshots = deque(maxlen=6); self._last_ts = 0.0
        self._last_vote = SignalVote("MKT_OI", "NEUTRAL", 0.0, "cold")

    def reset(self):
        self._snapshots.clear(); self._last_ts = 0.0; self._last_vote = SignalVote("MKT_OI", "NEUTRAL", 0.0, "cold")

    def update(self, futures_price: float, futures_oi: float, ce_oi: float = 0.0, pe_oi: float = 0.0) -> SignalVote:
        now = time.monotonic(); total_oi = ce_oi + pe_oi
        if total_oi <= 0: return self._last_vote
        self._snapshots.append({'ts': now, 'price': futures_price, 'ce_oi': ce_oi, 'pe_oi': pe_oi, 'total': total_oi})
        if now - self._last_ts < self.SNAP_INTERVAL or len(self._snapshots) < 2: return self._last_vote
        self._last_ts = now
        snap_old = self._snapshots[0]; snap_new = self._snapshots[-1]
        price_delta = snap_new['price'] - snap_old['price']; total_old = max(snap_old['total'], 1)
        ce_delta_pct = (snap_new['ce_oi'] - snap_old['ce_oi']) / total_old; pe_delta_pct = (snap_new['pe_oi'] - snap_old['pe_oi']) / total_old
        net_oi = pe_delta_pct - ce_delta_pct; mag = min(abs(net_oi) * 20, 0.25)
        price_up = price_delta > 1.0; price_dn = price_delta < -1.0
        pe_dom = (snap_new['pe_oi'] - snap_old['pe_oi']) > (snap_new['ce_oi'] - snap_old['ce_oi']) and abs(net_oi) > self.MIN_OI_CHANGE
        ce_dom = (snap_new['ce_oi'] - snap_old['ce_oi']) > (snap_new['pe_oi'] - snap_old['pe_oi']) and abs(net_oi) > self.MIN_OI_CHANGE

        if price_up and ce_dom: self._last_vote = SignalVote("MKT_OI", "CE", min(0.62+mag, 0.90), "LongBuildup")
        elif price_dn and pe_dom: self._last_vote = SignalVote("MKT_OI", "PE", min(0.62+mag, 0.90), "ShortBuildup")
        elif price_up and pe_dom: self._last_vote = SignalVote("MKT_OI", "CE", min(0.52+mag, 0.75), "ShortCovering")
        elif price_dn and ce_dom: self._last_vote = SignalVote("MKT_OI", "PE", min(0.52+mag, 0.75), "LongUnwinding")
        else: self._last_vote = SignalVote("MKT_OI", "NEUTRAL", 0.2, "unclear")
        return self._last_vote

    def current_vote(self) -> SignalVote: return self._last_vote

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 16: COI PCR
# ─────────────────────────────────────────────────────────────────────────────
class COIPCRSignal:
    # v28 Innovation J: Enhanced COI Strength % — expose raw institutional positioning
    # (Aristotle Phase 2 — Examine Every Assumption):
    #   Original code computed COI strength but only used it for signal voting.
    #   The raw % is GOLD for the dashboard — it tells the trader exactly what
    #   institutions are doing RIGHT NOW. >60% = writing puts (bullish), <-60% = writing
    #   calls (bearish). Sudden shifts (±30% in one refresh) precede trend changes.
    #   Enhancement: expose strength_pct for dashboard + meta-confirmation.
    SNAP_INTERVAL = 120; STRONG_BULL = 0.30; STRONG_BEAR = -0.30
    def __init__(self):
        self._base_ce = 0.0; self._base_pe = 0.0; self._baseline_set = False; self._last_ts = 0.0
        self._last_vote = SignalVote("COI_PCR", "NEUTRAL", 0.0, "cold")
        self.strength_pct: float = 0.0  # v28: raw COI Strength % for dashboard

    def reset(self):
        self._base_ce = 0.0; self._base_pe = 0.0; self._baseline_set = False; self._last_ts = 0.0
        self._last_vote = SignalVote("COI_PCR", "NEUTRAL", 0.0, "cold")
        self.strength_pct = 0.0

    def update(self, total_ce_oi: float, total_pe_oi: float) -> SignalVote:
        now = time.monotonic()
        if total_ce_oi <= 0 or total_pe_oi <= 0: return self._last_vote
        if not self._baseline_set:
            self._base_ce = total_ce_oi; self._base_pe = total_pe_oi; self._baseline_set = True
            return SignalVote("COI_PCR", "NEUTRAL", 0.0, "baseline_set")
        if now - self._last_ts < self.SNAP_INTERVAL: return self._last_vote
        self._last_ts = now
        ce_coi = total_ce_oi - self._base_ce; pe_coi = total_pe_oi - self._base_pe
        denom  = abs(pe_coi) + abs(ce_coi)
        if denom < 1:
            self.strength_pct = 0.0
            return SignalVote("COI_PCR", "NEUTRAL", 0.2, "insufficient_coi")
        strength = (pe_coi - ce_coi) / denom
        self.strength_pct = strength * 100.0  # v28: store raw % for dashboard
        if strength > self.STRONG_BULL: self._last_vote = SignalVote("COI_PCR", "PE", min(0.50 + strength * 0.6, 0.88), "PE_Buy")
        elif strength < self.STRONG_BEAR: self._last_vote = SignalVote("COI_PCR", "CE", min(0.50 + abs(strength) * 0.6, 0.88), "CE_Buy")
        else: self._last_vote = SignalVote("COI_PCR", "NEUTRAL", 0.2, "balanced")
        return self._last_vote

    def current_vote(self) -> SignalVote: return self._last_vote

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 17: PUT-CALL PARITY
# ─────────────────────────────────────────────────────────────────────────────
class PutCallParitySignal:
    NOISE_THRESHOLD = 8.0; SIGNAL_THRESHOLD = 15.0; STRONG_THRESHOLD = 25.0; MIN_TICKS = 3
    def __init__(self): self._drift_buf = deque(maxlen=10); self._n = 0
    def reset(self): self._drift_buf.clear(); self._n = 0

    def update(self, atm_strike: float, ce_ltp: float, pe_ltp: float, futures_price: float) -> SignalVote:
        self._n += 1
        if atm_strike <= 0 or ce_ltp <= 0 or pe_ltp <= 0 or futures_price <= 0: return SignalVote("PCP", "NEUTRAL", 0.0, "no_data")
        f_implied = atm_strike + ce_ltp - pe_ltp; drift = f_implied - futures_price
        self._drift_buf.append(drift)
        if len(self._drift_buf) < self.MIN_TICKS: return SignalVote("PCP", "NEUTRAL", 0.0, "warming")

        sorted_drifts = sorted(self._drift_buf)
        median_drift = sorted_drifts[len(sorted_drifts) // 2]; abs_drift = abs(median_drift)

        if abs_drift <= self.NOISE_THRESHOLD: return SignalVote("PCP", "NEUTRAL", 0.2, "noise")
        if abs_drift >= self.STRONG_THRESHOLD: score = min(0.75 + (abs_drift - self.STRONG_THRESHOLD) / 100, 0.92)
        elif abs_drift >= self.SIGNAL_THRESHOLD: score = 0.60 + (abs_drift - self.SIGNAL_THRESHOLD) / (self.STRONG_THRESHOLD - self.SIGNAL_THRESHOLD) * 0.15
        else: score = 0.50 + (abs_drift - self.NOISE_THRESHOLD) / (self.SIGNAL_THRESHOLD - self.NOISE_THRESHOLD) * 0.10

        if median_drift > 0: return SignalVote("PCP", "CE", round(score, 3), "PCP_CE")
        else: return SignalVote("PCP", "PE", round(score, 3), "PCP_PE")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 18: ANCHORED VWAP
# ─────────────────────────────────────────────────────────────────────────────
class AnchoredVWAPSignal:
    PIVOT_LOOKBACK = 5; MAX_ANCHORS = 3; MIN_CANDLES = 10
    def __init__(self):
        self._candle_closes = deque(maxlen=200); self._candle_highs = deque(maxlen=200); self._candle_lows = deque(maxlen=200)
        self._n = 0; self._res_anchors = []; self._sup_anchors = []; self._prev_fut = float('nan')
        self._last_vote = SignalVote("AVWAP", "NEUTRAL", 0.0, "cold")

    def reset(self):
        self._candle_closes.clear(); self._candle_highs.clear(); self._candle_lows.clear()
        self._n = 0; self._res_anchors.clear(); self._sup_anchors.clear(); self._prev_fut = float('nan')
        self._last_vote = SignalVote("AVWAP", "NEUTRAL", 0.0, "cold")

    def _compute_avwap(self, anchor_idx: int) -> float:
        closes = list(self._candle_closes)
        if anchor_idx >= len(closes) or anchor_idx < 0: return float('nan')
        subset = closes[anchor_idx:]
        return float(np.mean(subset)) if subset else float('nan')

    def on_candle(self, candle: Candle):
        self._candle_closes.append(candle.close); self._candle_highs.append(candle.high); self._candle_lows.append(candle.low)
        self._n += 1
        pl = self.PIVOT_LOOKBACK
        if self._n < pl * 2 + 1: return

        highs = list(self._candle_highs); lows = list(self._candle_lows)
        pivot_idx = len(highs) - pl - 1
        if pivot_idx < pl: return
        pivot_high_val = highs[pivot_idx]; pivot_low_val = lows[pivot_idx]

        window_h = highs[pivot_idx - pl: pivot_idx + pl + 1]; window_l = lows[pivot_idx - pl: pivot_idx + pl + 1]
        if pivot_high_val == max(window_h):
            self._res_anchors.append((len(self._candle_closes) - pl - 1, pivot_high_val))
            if len(self._res_anchors) > self.MAX_ANCHORS: self._res_anchors.pop(0)
        if pivot_low_val == min(window_l):
            self._sup_anchors.append((len(self._candle_closes) - pl - 1, pivot_low_val))
            if len(self._sup_anchors) > self.MAX_ANCHORS: self._sup_anchors.pop(0)

    def update(self, futures_price: float) -> SignalVote:
        if self._n < self.MIN_CANDLES: return SignalVote("AVWAP", "NEUTRAL", 0.0, "warming")
        prev = self._prev_fut; curr = futures_price; self._prev_fut = curr
        if math.isnan(prev) or prev <= 0 or curr <= 0: return self._last_vote

        best_ce_score = 0.0; best_ce_reason = ""; best_pe_score = 0.0; best_pe_reason = ""
        for anchor_idx, anchor_price in self._res_anchors:
            avwap = self._compute_avwap(anchor_idx)
            if math.isnan(avwap) or avwap <= 0: continue
            if prev <= avwap < curr:
                score = min(0.62 + ((curr - avwap) / avwap) * 50, 0.88)
                if score > best_ce_score: best_ce_score = score; best_ce_reason = "AVWAP_break↑"
            elif curr > avwap and anchor_price > avwap:
                score = min(0.52 + ((curr - avwap) / avwap) * 30, 0.72)
                if score > best_ce_score: best_ce_score = score; best_ce_reason = "Above_res_AVWAP"

        for anchor_idx, anchor_price in self._sup_anchors:
            avwap = self._compute_avwap(anchor_idx)
            if math.isnan(avwap) or avwap <= 0: continue
            if prev >= avwap > curr:
                score = min(0.62 + ((avwap - curr) / avwap) * 50, 0.88)
                if score > best_pe_score: best_pe_score = score; best_pe_reason = "AVWAP_break↓"
            elif curr < avwap and anchor_price < avwap:
                score = min(0.52 + ((avwap - curr) / avwap) * 30, 0.72)
                if score > best_pe_score: best_pe_score = score; best_pe_reason = "Below_sup_AVWAP"

        if best_ce_score > 0.60 and best_ce_score >= best_pe_score: self._last_vote = SignalVote("AVWAP", "CE", best_ce_score, best_ce_reason)
        elif best_pe_score > 0.60: self._last_vote = SignalVote("AVWAP", "PE", best_pe_score, best_pe_reason)
        else: self._last_vote = SignalVote("AVWAP", "NEUTRAL", 0.20, "no_break")
        return self._last_vote

    def current_vote(self) -> SignalVote: return self._last_vote

# ─────────────────────────────────────────────────────────────────────────────
# DIVERGENCE DETECTOR (used by engine)
# ─────────────────────────────────────────────────────────────────────────────
class RSIDivergenceDetector:
    MIN_CANDLES = 5
    def __init__(self):
        self._price_highs: deque = deque(maxlen=10)
        self._price_lows: deque  = deque(maxlen=10)
        self._rsi_at_high: deque = deque(maxlen=10)
        self._rsi_at_low: deque  = deque(maxlen=10)
        self._candle_count = 0
        self._last_candle_high = 0.0
        self._last_candle_low  = float('inf')
        self.bearish_divergence = False
        self.bullish_divergence = False
        self._warning = ""

    def reset(self):
        self._price_highs.clear(); self._price_lows.clear()
        self._rsi_at_high.clear(); self._rsi_at_low.clear()
        self._candle_count = 0; self._last_candle_high = 0.0; self._last_candle_low = float('inf')
        self.bearish_divergence = False; self.bullish_divergence = False; self._warning = ""

    def on_candle(self, candle: Candle, rsi: float):
        self._candle_count += 1
        self.bearish_divergence = False; self.bullish_divergence = False; self._warning = ""

        if candle.high > self._last_candle_high:
            self._price_highs.append(candle.high)
            self._rsi_at_high.append(rsi)
        if candle.low < self._last_candle_low:
            self._price_lows.append(candle.low)
            self._rsi_at_low.append(rsi)

        self._last_candle_high = candle.high
        self._last_candle_low  = candle.low

        if self._candle_count < self.MIN_CANDLES:
            return

        if len(self._price_highs) >= 2 and len(self._rsi_at_high) >= 2:
            ph = list(self._price_highs); rh = list(self._rsi_at_high)
            if ph[-1] > ph[-2] and rh[-1] < rh[-2] - 2.0:
                self.bearish_divergence = True
                self._warning = f"BEARISH_DIV: price HH {ph[-1]:.0f}>{ph[-2]:.0f} but RSI LH {rh[-1]:.0f}<{rh[-2]:.0f}"

        if len(self._price_lows) >= 2 and len(self._rsi_at_low) >= 2:
            pl = list(self._price_lows); rl = list(self._rsi_at_low)
            if pl[-1] < pl[-2] and rl[-1] > rl[-2] + 2.0:
                self.bullish_divergence = True
                self._warning = f"BULLISH_DIV: price LL {pl[-1]:.0f}<{pl[-2]:.0f} but RSI HL {rl[-1]:.0f}>{rl[-2]:.0f}"

    @property
    def warning(self) -> str:
        return self._warning

# ─────────────────────────────────────────────────────────────────────────────
# STRADDLE GAMMA SIGNAL (v17.2)
# ATM straddle price = CE + PE. When gamma accelerates, straddle rises.
# This is a LEADING indicator — fires BEFORE the directional move completes.
# ─────────────────────────────────────────────────────────────────────────────
class StraddleGammaSignal:
    """Tracks ATM straddle velocity and acceleration to detect gamma expansion.

    Logic:
      - Straddle velocity > 0 AND accelerating → vol expansion, big move coming
      - Which leg is gaining faster → direction of the move
      - Straddle decelerating or falling → vol contraction, avoid
    """
    WINDOW = 30          # ~8s at 3.6Hz: enough for velocity estimation
    ACCEL_WINDOW = 15    # compare recent velocity to older velocity
    MIN_TICKS = 20       # need this many ticks before voting

    # Thresholds (in ₹/tick for straddle)
    VEL_THRESHOLD = 0.03     # straddle gaining ₹0.03/tick = meaningful
    ACCEL_THRESHOLD = 0.005  # acceleration must be positive and meaningful
    LEG_DOMINANCE = 0.6      # one leg must contribute >60% of straddle gain

    def __init__(self):
        self._straddle_history: deque = deque(maxlen=60)
        self._ce_history: deque = deque(maxlen=60)
        self._pe_history: deque = deque(maxlen=60)
        self._n = 0

    def reset(self):
        self._straddle_history.clear()
        self._ce_history.clear()
        self._pe_history.clear()
        self._n = 0

    def update(self, ce_ltp: float, pe_ltp: float) -> SignalVote:
        if ce_ltp <= 0 or pe_ltp <= 0:
            return SignalVote("STRADDLE_GAMMA", "NEUTRAL", 0.0, "no_data")

        straddle = ce_ltp + pe_ltp
        self._straddle_history.append(straddle)
        self._ce_history.append(ce_ltp)
        self._pe_history.append(pe_ltp)
        self._n += 1

        if self._n < self.MIN_TICKS:
            return SignalVote("STRADDLE_GAMMA", "NEUTRAL", 0.0, "warmup")

        hist = list(self._straddle_history)
        ce_hist = list(self._ce_history)
        pe_hist = list(self._pe_history)
        n = len(hist)

        # Straddle velocity: rate of change over last WINDOW ticks
        lookback = min(self.WINDOW, n - 1)
        if lookback < 5:
            return SignalVote("STRADDLE_GAMMA", "NEUTRAL", 0.0, "insufficient")

        straddle_vel = (hist[-1] - hist[-lookback]) / lookback

        # Straddle acceleration: is velocity increasing?
        half = lookback // 2
        if half < 3:
            straddle_accel = 0.0
        else:
            vel_recent = (hist[-1] - hist[-half]) / half
            vel_older = (hist[-half] - hist[-lookback]) / (lookback - half)
            straddle_accel = vel_recent - vel_older

        # Leg analysis: which leg is driving the straddle change?
        ce_change = ce_hist[-1] - ce_hist[-lookback]
        pe_change = pe_hist[-1] - pe_hist[-lookback]
        total_change = abs(ce_change) + abs(pe_change)

        # Determine direction from leg dominance
        direction = "NEUTRAL"
        reason_parts = []

        if straddle_vel > self.VEL_THRESHOLD:
            reason_parts.append(f"strad_vel={straddle_vel:+.3f}")

            if straddle_accel > self.ACCEL_THRESHOLD:
                reason_parts.append(f"accel={straddle_accel:+.4f}")

                # Which leg is gaining?
                if total_change > 0.01:
                    ce_frac = ce_change / total_change if ce_change > 0 else 0
                    pe_frac = pe_change / total_change if pe_change > 0 else 0

                    if ce_change > 0 and ce_frac > self.LEG_DOMINANCE:
                        direction = "CE"
                        reason_parts.append(f"CE_leading={ce_frac:.0%}")
                    elif pe_change > 0 and pe_frac > self.LEG_DOMINANCE:
                        direction = "PE"
                        reason_parts.append(f"PE_leading={pe_frac:.0%}")
                    elif ce_change > pe_change and ce_change > 0:
                        # CE gaining more but not dominant — mild CE signal
                        direction = "CE"
                        reason_parts.append(f"CE_edge={ce_frac:.0%}")
                    elif pe_change > ce_change and pe_change > 0:
                        direction = "PE"
                        reason_parts.append(f"PE_edge={pe_frac:.0%}")

            elif straddle_accel < -self.ACCEL_THRESHOLD:
                # Straddle rising but decelerating — momentum fading
                reason_parts.append(f"decel={straddle_accel:+.4f}")

        elif straddle_vel < -self.VEL_THRESHOLD:
            # Straddle FALLING — vol contraction, bad for directional trades
            reason_parts.append(f"vol_contract={straddle_vel:+.3f}")

        # Score based on straddle velocity + acceleration strength
        if direction != "NEUTRAL":
            vel_score = min(abs(straddle_vel) / 0.10, 1.0)  # normalize: 0.10/tick = full score
            accel_score = min(abs(straddle_accel) / 0.02, 1.0)
            score = 0.5 * vel_score + 0.5 * accel_score
            score = max(0.1, min(0.9, score))
        else:
            score = 0.0

        reason = " ".join(reason_parts) if reason_parts else "flat"
        return SignalVote("STRADDLE_GAMMA", direction, score, reason)

    @property
    def straddle_velocity(self) -> float:
        """Current straddle velocity for external use (e.g., P(win) context)."""
        hist = list(self._straddle_history)
        if len(hist) < 10:
            return 0.0
        return (hist[-1] - hist[-10]) / 10.0

    @property
    def is_expanding(self) -> bool:
        """Is straddle in sustained expansion? Requires rising over BOTH recent and medium windows."""
        hist = list(self._straddle_history)
        if len(hist) < 40:
            return False
        # Medium window: last 30 ticks (~8s)
        vel_medium = (hist[-1] - hist[-30]) / 30.0
        # Short window: last 10 ticks (~3s)
        vel_short = (hist[-1] - hist[-10]) / 10.0
        # Both must be positive AND above threshold — prevents momentary blips
        return (vel_medium > self.VEL_THRESHOLD * 0.7 and
                vel_short > self.VEL_THRESHOLD and
                vel_short > vel_medium * 0.8)  # not decelerating sharply


# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM DIVERGENCE — strongest direction confirmation available
# ─────────────────────────────────────────────────────────────────────────────
class PremiumDivergenceFilter:
    """
    Direction confirmation via Synthetic Delta + CE/PE divergence.

    v18.1 — Enhanced with research findings:

    Layer 1: SYNTHETIC DELTA (strongest signal)
      SyntheticDelta = CE_price - PE_price (at same ATM strike)
      This is mathematically equivalent to the synthetic futures premium.
      When SynDelta rises and spot lags → leading bullish signal (and vice versa).
      Research: De Jong & Donders (intraday lead-lag), Khan (Nifty futures-spot).

    Layer 2: CE/PE VELOCITY DIVERGENCE (confirmation)
      CE rising + PE falling = genuine directional flow (not IV change).
      IV changes affect both sides equally and cancel in the difference.
      TRUE divergence (one up, one down) is 50% stronger than relative speed.

    Layer 3: FUTURES BASIS DEVIATION (if futures data available)
      BasisDev = (Futures - Spot) - FairBasis
      Rising BasisDev = bullish (futures pulling spot up).
      Kawaller et al: futures lead spot by 20-45min. Strongest during vol events.
    """
    LOOKBACK_SHORT = 20    # ~6s at 3.6Hz — immediate
    LOOKBACK_MED   = 60    # ~17s — confirmed
    MIN_HISTORY    = 30
    # Synthetic delta thresholds (in points)
    SYNDELTA_MIN_CHANGE = 0.5   # minimum meaningful change in synthetic delta
    SYNDELTA_STRONG     = 3.0   # strong directional signal in synthetic delta

    def __init__(self):
        self._ce_history = deque(maxlen=200)
        self._pe_history = deque(maxlen=200)
        self._spot_history = deque(maxlen=200)
        self._syndelta_history = deque(maxlen=200)  # CE - PE over time
        self._futures_history = deque(maxlen=200)

    def update(self, ce_ltp: float, pe_ltp: float, spot: float, futures: float = 0.0):
        if ce_ltp > 0 and not math.isnan(ce_ltp):
            self._ce_history.append(ce_ltp)
        if pe_ltp > 0 and not math.isnan(pe_ltp):
            self._pe_history.append(pe_ltp)
        self._spot_history.append(spot)
        # Synthetic delta: CE - PE (put-call parity proxy)
        if ce_ltp > 0 and pe_ltp > 0 and not math.isnan(ce_ltp) and not math.isnan(pe_ltp):
            self._syndelta_history.append(ce_ltp - pe_ltp)
        if futures > 0:
            self._futures_history.append(futures)

    def _synthetic_delta_signal(self) -> tuple:
        """Synthetic Delta direction: rate of change of (CE - PE).

        Returns (direction, strength, reason)
        """
        sd = list(self._syndelta_history)
        if len(sd) < self.MIN_HISTORY:
            return "NEUTRAL", 0.0, "syndelta_insufficient"

        # Short-term rate of change (~6s)
        n_s = min(self.LOOKBACK_SHORT, len(sd) - 1)
        delta_short = sd[-1] - sd[-1 - n_s]

        # Medium-term rate of change (~17s)
        n_m = min(self.LOOKBACK_MED, len(sd) - 1)
        delta_med = sd[-1] - sd[-1 - n_m]

        # Both timeframes must agree
        if delta_short > self.SYNDELTA_MIN_CHANGE and delta_med > self.SYNDELTA_MIN_CHANGE:
            # Synthetic delta rising = CE outperforming = bullish
            strength = min(abs(delta_med) / self.SYNDELTA_STRONG, 1.0)
            return "CE", strength, f"SYNDELTA_UP s={delta_short:+.1f} m={delta_med:+.1f}"
        elif delta_short < -self.SYNDELTA_MIN_CHANGE and delta_med < -self.SYNDELTA_MIN_CHANGE:
            # Synthetic delta falling = PE outperforming = bearish
            strength = min(abs(delta_med) / self.SYNDELTA_STRONG, 1.0)
            return "PE", strength, f"SYNDELTA_DN s={delta_short:+.1f} m={delta_med:+.1f}"
        else:
            return "NEUTRAL", 0.0, f"SYNDELTA_FLAT s={delta_short:+.1f} m={delta_med:+.1f}"

    def _futures_basis_signal(self) -> tuple:
        """Futures basis deviation: when futures lead spot.

        Returns (direction, strength, reason)
        """
        fh = list(self._futures_history)
        sh = list(self._spot_history)
        if len(fh) < self.MIN_HISTORY or len(sh) < self.MIN_HISTORY:
            return "NEUTRAL", 0.0, "basis_insufficient"

        # Current basis = futures - spot
        n_m = min(self.LOOKBACK_MED, len(fh) - 1, len(sh) - 1)
        basis_now = fh[-1] - sh[-1]
        basis_prev = fh[-1 - n_m] - sh[-1 - n_m]
        basis_change = basis_now - basis_prev

        # Basis expanding = futures pulling ahead = directional signal
        if abs(basis_change) < 1.0:
            return "NEUTRAL", 0.0, f"BASIS_FLAT chg={basis_change:+.1f}"

        if basis_change > 0:
            # Futures running ahead of spot = bullish
            strength = min(abs(basis_change) / 8.0, 1.0)
            return "CE", strength, f"BASIS_UP chg={basis_change:+.1f}"
        else:
            # Futures lagging/falling vs spot = bearish
            strength = min(abs(basis_change) / 8.0, 1.0)
            return "PE", strength, f"BASIS_DN chg={basis_change:+.1f}"

    def _premium_velocity_signal(self) -> tuple:
        """Original CE/PE velocity divergence (Layer 2).

        Returns (direction, strength, reason)
        """
        ce_h = list(self._ce_history)
        pe_h = list(self._pe_history)

        if len(ce_h) < self.MIN_HISTORY or len(pe_h) < self.MIN_HISTORY:
            return "NEUTRAL", 0.0, "vel_insufficient"

        n_short = min(self.LOOKBACK_SHORT, len(ce_h) - 1, len(pe_h) - 1)
        ce_vel_s = (ce_h[-1] - ce_h[-1 - n_short]) / max(ce_h[-1 - n_short], 1.0)
        pe_vel_s = (pe_h[-1] - pe_h[-1 - n_short]) / max(pe_h[-1 - n_short], 1.0)

        n_med = min(self.LOOKBACK_MED, len(ce_h) - 1, len(pe_h) - 1)
        ce_vel_m = (ce_h[-1] - ce_h[-1 - n_med]) / max(ce_h[-1 - n_med], 1.0)
        pe_vel_m = (pe_h[-1] - pe_h[-1 - n_med]) / max(pe_h[-1 - n_med], 1.0)

        div_score_m = ce_vel_m - pe_vel_m
        div_score_s = ce_vel_s - pe_vel_s

        if div_score_m > 0 and div_score_s > 0:
            direction = "CE"
            strength = min(abs(div_score_m) / 0.08, 1.0)
            if ce_vel_m > 0 and pe_vel_m < 0:
                strength = min(strength * 1.5, 1.0)
                reason = f"TRUE_DIV CE+{ce_vel_m:.3f} PE{pe_vel_m:.3f}"
            else:
                reason = f"REL_DIV CE{ce_vel_m:+.3f} PE{pe_vel_m:+.3f}"
        elif div_score_m < 0 and div_score_s < 0:
            direction = "PE"
            strength = min(abs(div_score_m) / 0.08, 1.0)
            if pe_vel_m > 0 and ce_vel_m < 0:
                strength = min(strength * 1.5, 1.0)
                reason = f"TRUE_DIV PE+{pe_vel_m:.3f} CE{ce_vel_m:.3f}"
            else:
                reason = f"REL_DIV PE{pe_vel_m:+.3f} CE{ce_vel_m:+.3f}"
        else:
            direction = "NEUTRAL"
            strength = 0.0
            reason = f"VEL_MIXED s={div_score_s:+.3f} m={div_score_m:+.3f}"

        return direction, round(strength, 3), reason

    def direction_strength(self) -> tuple:
        """Combined direction from all 3 layers.

        Returns (direction, strength, reason).
        Layers are weighted: SynDelta 40%, PremVel 35%, FutBasis 25%.
        Agreement across layers multiplies strength (consensus bonus).
        """
        sd_dir, sd_str, sd_reason = self._synthetic_delta_signal()
        pv_dir, pv_str, pv_reason = self._premium_velocity_signal()
        fb_dir, fb_str, fb_reason = self._futures_basis_signal()

        # Count directional votes (excluding NEUTRAL)
        ce_score = 0.0
        pe_score = 0.0
        reasons = []

        for d, s, w, r in [(sd_dir, sd_str, 0.40, sd_reason),
                            (pv_dir, pv_str, 0.35, pv_reason),
                            (fb_dir, fb_str, 0.25, fb_reason)]:
            if d == "CE":
                ce_score += s * w
                reasons.append(r)
            elif d == "PE":
                pe_score += s * w
                reasons.append(r)

        # Direction is whichever has higher weighted score
        if ce_score > pe_score and ce_score > 0.05:
            direction = "CE"
            raw_strength = ce_score
            # Consensus bonus: if all non-neutral layers agree, boost 30%
            non_neutral = [d for d in [sd_dir, pv_dir, fb_dir] if d != "NEUTRAL"]
            if len(non_neutral) >= 2 and all(d == "CE" for d in non_neutral):
                raw_strength = min(raw_strength * 1.3, 1.0)
                reasons.append("CONSENSUS")
            # Opposition penalty: if any layer opposes, reduce
            if pe_score > 0.05:
                raw_strength *= max(0.3, 1.0 - pe_score)
                reasons.append(f"opp={pe_score:.2f}")
        elif pe_score > ce_score and pe_score > 0.05:
            direction = "PE"
            raw_strength = pe_score
            non_neutral = [d for d in [sd_dir, pv_dir, fb_dir] if d != "NEUTRAL"]
            if len(non_neutral) >= 2 and all(d == "PE" for d in non_neutral):
                raw_strength = min(raw_strength * 1.3, 1.0)
                reasons.append("CONSENSUS")
            if ce_score > 0.05:
                raw_strength *= max(0.3, 1.0 - ce_score)
                reasons.append(f"opp={ce_score:.2f}")
        else:
            direction = "NEUTRAL"
            raw_strength = 0.0

        strength = min(round(raw_strength, 3), 1.0)
        reason = " | ".join(reasons) if reasons else "no_signal"
        return direction, strength, reason

    def confirms_direction(self, proposed_direction: str) -> tuple:
        """Check if premium divergence confirms a proposed entry direction.

        Returns (confirms: bool, strength: float, reason: str)
        """
        div_dir, strength, reason = self.direction_strength()

        if div_dir == "NEUTRAL":
            return True, 0.0, f"NEUTRAL ({reason})"

        if div_dir == proposed_direction:
            return True, strength, f"CONFIRMED: {reason}"

        # Divergence OPPOSES proposed direction
        if strength >= 0.3:
            return False, strength, f"OPPOSED: {reason} (str={strength:.2f})"

        # Weak opposition — allow but with reduced conviction
        return True, -strength, f"WEAK_OPP: {reason} (str={strength:.2f})"

    def reset(self):
        self._ce_history.clear()
        self._pe_history.clear()
        self._spot_history.clear()
        self._syndelta_history.clear()
        self._futures_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME DETECTOR — trending vs range-bound
# ─────────────────────────────────────────────────────────────────────────────
class MarketRegimeDetector:
    """
    Classifies market regime using ADX + ATR expansion + price structure.

    Regimes (5 states):
      BREAKOUT  — BB squeeze fires + ATR expanding fast. Best entries.
      TRENDING  — ADX > 22, clear directional movement. Trust signals.
      RANGE     — ADX < 22, moderate vol. Need straddle expansion.
      CHOPPY    — ADX < 15, frequent direction changes. Most stall kills come from here.
      LOW_VOL   — ATR < 60% median AND ADX < 18. Dead market, avoid entries.

    Impact on trading:
      TRENDING:  Enter on pullbacks, trust signals, hold longer
      BREAKOUT:  Best entries — fresh move with expanding vol
      RANGE:     Require straddle expansion (gamma signal) before entry
      CHOPPY:    Heavy conviction penalty — signals are unreliable
      LOW_VOL:   Strongest penalty — market isn't moving enough for directional trades
    """
    # Thresholds for choppy detection
    CHOPPY_ADX = 15            # ADX below this = choppy (no trend at all)
    LOW_VOL_ADX = 18           # ADX threshold for low-vol regime
    DIRECTION_FLIP_WINDOW = 30 # candles to track direction changes
    CHOPPY_FLIP_THRESHOLD = 6  # 6+ direction flips in 30 candles = choppy

    def __init__(self):
        self.regime = "UNKNOWN"
        self._adx_ref = None      # reference to ADXTrendFilter
        self._atr_ref = None      # reference to ATRCalculator
        self._bb_ref = None       # reference to BollingerSqueezeSignal
        self._straddle_ref = None # reference to StraddleGammaSignal
        self._direction_history: deque = deque(maxlen=self.DIRECTION_FLIP_WINDOW)
        self._last_candle_dir = "NEUTRAL"
        self._last_candle_count = 0  # v20: track candle count to only append on new candle close

    def set_refs(self, adx, atr, bb_squeeze, straddle):
        self._adx_ref = adx
        self._atr_ref = atr
        self._bb_ref = bb_squeeze
        self._straddle_ref = straddle

    def _count_direction_flips(self) -> int:
        """Count how many times direction flipped in recent history."""
        dirs = [d for d in self._direction_history if d != "NEUTRAL"]
        if len(dirs) < 3:
            return 0
        flips = 0
        for i in range(1, len(dirs)):
            if dirs[i] != dirs[i-1]:
                flips += 1
        return flips

    def update(self, candle_direction: str = "NEUTRAL", candle_count: int = 0) -> str:
        """Update regime. candle_direction = direction of last closed candle (CE/PE/NEUTRAL)."""
        adx = self._adx_ref.adx if self._adx_ref else 0
        is_hv = self._atr_ref.is_high_volatility() if self._atr_ref else False
        is_lv = self._atr_ref.is_low_volatility() if self._atr_ref else False
        sq_fire = self._bb_ref._fire_ticks > 0 if self._bb_ref else False

        # v20: Only track direction when a NEW candle closes, not every tick.
        # Previously appended same direction 1000+ times per candle, making flips always 0.
        if candle_count > self._last_candle_count and candle_direction != "NEUTRAL":
            self._direction_history.append(candle_direction)
            self._last_candle_count = candle_count
        flips = self._count_direction_flips()

        # Priority order: BREAKOUT > TRENDING > CHOPPY > LOW_VOL > RANGE
        if sq_fire and is_hv:
            self.regime = "BREAKOUT"
        elif adx >= ADX_TRENDING_THRESHOLD:
            self.regime = "TRENDING"
        elif is_lv and adx < self.LOW_VOL_ADX:
            # Dead market — ATR collapsed AND no trend
            self.regime = "LOW_VOL"
        elif adx < self.CHOPPY_ADX or flips >= self.CHOPPY_FLIP_THRESHOLD:
            # Very low ADX or frequent direction flips = choppy noise
            self.regime = "CHOPPY"
        else:
            self.regime = "RANGE"

        return self.regime

    @property
    def is_trending(self) -> bool:
        return self.regime in ("TRENDING", "BREAKOUT")

    @property
    def is_range(self) -> bool:
        return self.regime == "RANGE"

    @property
    def is_breakout(self) -> bool:
        return self.regime == "BREAKOUT"

    @property
    def is_choppy(self) -> bool:
        return self.regime == "CHOPPY"

    @property
    def is_low_vol(self) -> bool:
        return self.regime == "LOW_VOL"

    def reset(self):
        self.regime = "UNKNOWN"
        self._direction_history.clear()
        self._last_candle_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# MARKET PHASE STATE MACHINE (v28 Innovation A)
# ─────────────────────────────────────────────────────────────────────────────
# FIRST PRINCIPLES:
#   Markets move in PHASES: accumulation → expansion → trend → exhaustion → distribution.
#   The same signal means different things in different phases.
#   Buying during EXPANSION is genius. Buying during EXHAUSTION is suicide.
#   This is the CONTEXT LAYER above signals — it determines WHEN to trade, not WHAT direction.
#
# This is fundamentally different from MarketRegimeDetector:
#   - MarketRegimeDetector answers: "What IS the market doing right now?" (static snapshot)
#   - MarketPhaseTracker answers: "WHERE in the move cycle are we?" (sequential lifecycle)
#
# PHASE DEFINITIONS:
#   ACCUMULATION — Market is coiling. Low ATR, low ADX, tight BBands. Energy is building.
#                  Action: NO ENTRY. Wait for the spring to release.
#   EXPANSION    — Breakout happening NOW. BB squeeze fires, ATR spikes, SV pre-expansion.
#                  Action: AGGRESSIVE ENTRY. This is THE moment. Relax gates.
#   TREND        — Sustained directional move. ADX rising, velocity positive.
#                  Action: PULLBACK ENTRY ONLY. Don't chase; enter on retests.
#   EXHAUSTION   — Move is dying. High displacement, velocity decaying, volume thinning.
#                  Action: NO NEW ENTRY. Tighten existing trailing stops.
#   DISTRIBUTION — Move is over. Choppy action, mixed signals. Market is resettling.
#                  Action: NO ENTRY. Wait for next accumulation cycle.
# ─────────────────────────────────────────────────────────────────────────────
class MarketPhaseTracker:
    """
    Sequential phase state machine tracking the lifecycle of market moves.

    Key insight: phases are SEQUENTIAL with logical transitions.
    The system tracks WHERE we are in the move cycle, not just
    what the market is doing at this instant.

    Requires references to: ATRCalculator, ADXTrendFilter, BollingerSqueezeSignal,
    StraddleGammaSignal, FuturesVelocitySignal, MarketRegimeDetector.
    """

    # ── Phase constants ──────────────────────────────────────────────────
    ACCUMULATION = "ACCUMULATION"
    EXPANSION    = "EXPANSION"
    TREND        = "TREND"
    EXHAUSTION   = "EXHAUSTION"
    DISTRIBUTION = "DISTRIBUTION"
    UNKNOWN      = "UNKNOWN"

    # ── Detection thresholds ─────────────────────────────────────────────
    # These are relative to ATR (volatility-normalized), not fixed points.
    EXPANSION_ATR_MULT    = 1.3    # ATR must be 1.3x median to confirm expansion
    EXHAUSTION_DISP_MULT  = 2.5    # displacement > 2.5x ATR = extended move
    EXHAUSTION_VEL_DECAY  = -0.01  # velocity acceleration must be negative (fading)
    TREND_MIN_ADX         = 22     # ADX above this = trending
    ACCUM_MAX_ADX         = 18     # ADX below this = accumulating/quiet
    PHASE_HOLD_TICKS      = 10     # minimum ticks before phase can transition (debounce)

    # v28 FIX (Aristotle Phase 2 — EXHAUSTION triggers too fast):
    #   Old code fired EXHAUSTION on the FIRST tick all three conditions were true.
    #   A 3-5 tick pullback in a strong trend shows velocity decay momentarily —
    #   that's a retest, not exhaustion.  True exhaustion has SUSTAINED velocity decay.
    #   Require 25 consecutive qualifying ticks (~8s at ~3 ticks/sec) before
    #   transitioning.  Brief pullbacks can't sustain 25 ticks; genuine exhaustion
    #   (60+ ticks of decay) hits 25 easily.
    #   Validated (Phase 4): 5-tick pullback → resets, never fires.
    #                        60-tick exhaustion → fires at tick 25.
    #                        1-tick bounce mid-decay → resets, filters dead-cat bounce.
    EXHAUSTION_CONFIRM_TICKS = 25   # consecutive qualifying ticks before EXHAUSTION fires

    # v28 Innovation I: CUSUM early detection thresholds
    # CUSUM drift: the expected "dead" SV is near 0. We look for deviation from 0.
    # Threshold: when accumulated deviation hits this, premium flow has STARTED.
    # Calibrated: dead SV noise is ±0.002. Threshold 0.025 needs ~10 ticks of
    # sustained 0.005 shift — filters noise while catching real flow early.
    CUSUM_THRESHOLD = 0.025   # accumulated SV deviation to fire early detection
    CUSUM_DRIFT     = 0.002   # allowance for noise (subtracted each tick)

    def __init__(self):
        self.phase: str = self.UNKNOWN
        self._prev_phase: str = self.UNKNOWN
        self._phase_start_tick: int = 0     # tick when current phase started
        self._tick_count: int = 0

        # References to other engine components (set via set_refs)
        self._atr_ref = None        # ATRCalculator
        self._adx_ref = None        # ADXTrendFilter
        self._bb_ref  = None        # BollingerSqueezeSignal
        self._straddle_ref = None   # StraddleGammaSignal
        self._fut_vel_ref  = None   # FuturesVelocitySignal
        self._regime_ref   = None   # MarketRegimeDetector

        # Move tracking — cumulative displacement within current move
        self._move_start_price: float = 0.0
        self._move_peak_displacement: float = 0.0
        self._velocity_history: deque = deque(maxlen=30)  # track velocity trend

        # v28 FIX: EXHAUSTION confirmation counter — must see sustained velocity
        # decay for EXHAUSTION_CONFIRM_TICKS consecutive ticks before firing.
        # Resets to 0 any tick the conditions are NOT met, filtering brief pullbacks.
        self._exhaustion_confirm_count: int = 0

        # v28 Innovation I: CUSUM Early Detection on Straddle Velocity
        # (Aristotle Phase 2 — Examine Every Assumption):
        #   The phase machine needs BB squeeze + ATR expansion to confirm EXPANSION.
        #   By then, the move is 15-20 ticks old. CUSUM detects the MOMENT straddle
        #   velocity shifts from dead (0.000) to alive (-0.01+), firing 10-15 ticks
        #   earlier. This gives us a HEAD START on entries.
        #   CUSUM is O(1) per tick: one addition, one comparison. Zero performance hit.
        #   It does NOT change the phase — it only sets a pre_expansion flag that
        #   the entry logic can use for earlier, still-gated entries.
        self._cusum_pos: float = 0.0     # upper CUSUM: accumulates when SV > 0 (IV expanding)
        self._cusum_neg: float = 0.0     # lower CUSUM: accumulates when SV < 0 (directional flow)
        self._cusum_fired: bool = False  # True when either CUSUM exceeds threshold
        self._cusum_direction: str = ""  # "AWAKENING" (directional) or "IV_SURGE" (expansion)

    def set_refs(self, atr, adx, bb_squeeze, straddle, fut_vel, regime):
        """Wire up references to other engine components."""
        self._atr_ref = atr
        self._adx_ref = adx
        self._bb_ref = bb_squeeze
        self._straddle_ref = straddle
        self._fut_vel_ref = fut_vel
        self._regime_ref = regime

    def update(self, fut_price: float, straddle_vel: float) -> str:
        """
        Update phase detection. Called once per engine tick.

        Returns the current phase string.

        Logic flow:
          1. Gather all component readings (ATR, ADX, BB, velocity, displacement)
          2. Evaluate phase conditions in PRIORITY ORDER
          3. Apply transition debounce (hold for PHASE_HOLD_TICKS before switching)
          4. Track move displacement for exhaustion detection
        """
        self._tick_count += 1

        # ── v28.4 FIX: Proper two-sided CUSUM on straddle velocity ──────
        # (Aristotle Phase 3 — Reconstruct from Atoms):
        #   v28.3 used abs(straddle_vel) which made _cusum_neg mathematically
        #   inert: max(0, X - |SV| + drift) → 0 whenever |SV| > drift.
        #   That reduced the "two-sided" CUSUM to a one-sided noise detector.
        #   FIX: use RAW straddle_vel (matching MomentumBurstDetector at L658).
        #   _cusum_pos accumulates when SV > drift (IV expanding, premium bid up)
        #   _cusum_neg accumulates when SV < -drift (directional contraction)
        #   Either side exceeding threshold = market waking from dead state.
        #   Bonus: alternating +/- noise now CANCELS instead of accumulating.
        self._cusum_pos = max(0.0, self._cusum_pos + straddle_vel - self.CUSUM_DRIFT)
        self._cusum_neg = max(0.0, self._cusum_neg - straddle_vel - self.CUSUM_DRIFT)

        if not self._cusum_fired:
            if self._cusum_neg >= self.CUSUM_THRESHOLD:
                self._cusum_fired = True
                self._cusum_direction = "AWAKENING"   # directional premium flow (SV < 0)
            elif self._cusum_pos >= self.CUSUM_THRESHOLD:
                self._cusum_fired = True
                self._cusum_direction = "IV_SURGE"    # volatility expansion (SV > 0)
        # Reset CUSUM once phase machine confirms (detection no longer needed)
        if self.phase in (self.EXPANSION, self.TREND):
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
            self._cusum_fired = False
            self._cusum_direction = ""

        # ── Gather readings from engine components ───────────────────────
        atr_val = self._atr_ref.atr if self._atr_ref and not math.isnan(self._atr_ref.atr) else 5.0
        is_hv = self._atr_ref.is_high_volatility() if self._atr_ref else False
        is_lv = self._atr_ref.is_low_volatility() if self._atr_ref else False
        adx = self._adx_ref.adx if self._adx_ref else 0
        sq_fire = self._bb_ref._fire_ticks > 0 if self._bb_ref else False
        is_moving = self._fut_vel_ref.is_moving if self._fut_vel_ref else False
        is_exhausted = self._fut_vel_ref.is_exhausted if self._fut_vel_ref else False
        fut_vel_displacement = self._fut_vel_ref.displacement if self._fut_vel_ref else 0.0

        # ── Track velocity trend (is move accelerating or decelerating?) ─
        self._velocity_history.append(fut_vel_displacement)
        vel_accel = 0.0
        if len(self._velocity_history) >= 10:
            recent_vel = abs(self._velocity_history[-1])
            older_vel = abs(self._velocity_history[-10])
            if older_vel > 0.5:  # only compute if there was meaningful velocity before
                vel_accel = (recent_vel - older_vel) / older_vel  # fractional change in velocity magnitude

        # ── Track cumulative move displacement ───────────────────────────
        if self._move_start_price <= 0 and fut_price > 0:
            self._move_start_price = fut_price
        if fut_price > 0 and self._move_start_price > 0:
            current_displacement = abs(fut_price - self._move_start_price)
            self._move_peak_displacement = max(self._move_peak_displacement, current_displacement)

        # ── ATR-normalized displacement check ────────────────────────────
        atr_disp_ratio = self._move_peak_displacement / atr_val if atr_val > 0.5 else 0.0

        # ── Debounce: don't switch phases too fast ───────────────────────
        ticks_in_phase = self._tick_count - self._phase_start_tick
        can_transition = ticks_in_phase >= self.PHASE_HOLD_TICKS

        # ── Phase detection (PRIORITY ORDER) ─────────────────────────────
        # Priority: EXHAUSTION > EXPANSION > TREND > DISTRIBUTION > ACCUMULATION
        # Exhaustion is highest priority because entering during exhaustion
        # is the MOST dangerous action — it must override everything.

        new_phase = self.phase  # default: stay in current phase

        # EXHAUSTION: large displacement + velocity decaying + past warmup
        # This is the "move is dying" detector.
        # v28 FIX: Requires EXHAUSTION_CONFIRM_TICKS consecutive qualifying ticks.
        #   Old code fired on the FIRST qualifying tick — a 3-tick pullback in a
        #   strong trend would trigger EXHAUSTION → tighten stops → stop out winner.
        #   Now the counter must build up (25 ticks ~8s).  Brief pullbacks reset it.
        _exhaustion_conditions_met = (
            atr_disp_ratio >= self.EXHAUSTION_DISP_MULT and
            vel_accel < self.EXHAUSTION_VEL_DECAY and
            self._tick_count > 100
        )
        if _exhaustion_conditions_met:
            self._exhaustion_confirm_count += 1
        else:
            self._exhaustion_confirm_count = 0  # hard reset — any break in decay = not exhausted

        if self._exhaustion_confirm_count >= self.EXHAUSTION_CONFIRM_TICKS:
            new_phase = self.EXHAUSTION

        # EXPANSION: BB squeeze fires + ATR expanding, OR strong SV pre-expansion.
        # This is the "breakout happening NOW" detector — THE moment to enter.
        elif sq_fire and is_hv:
            new_phase = self.EXPANSION
        elif straddle_vel < -0.02 and is_moving and adx > 15:
            # Strong pre-expansion: SV deeply negative (institutions loading)
            # + price actually moving + some trend strength
            new_phase = self.EXPANSION

        # TREND: sustained directional move, ADX confirms
        elif adx >= self.TREND_MIN_ADX and is_moving and not is_exhausted:
            new_phase = self.TREND

        # DISTRIBUTION: coming from exhaustion/trend, now choppy
        elif (self.phase in (self.EXHAUSTION, self.TREND) and
              not is_moving and adx < self.TREND_MIN_ADX):
            new_phase = self.DISTRIBUTION

        # ACCUMULATION: low energy, market coiling
        elif is_lv or (adx < self.ACCUM_MAX_ADX and not is_moving and abs(straddle_vel) < 0.01):
            new_phase = self.ACCUMULATION

        # ── Apply transition with debounce ───────────────────────────────
        if new_phase != self.phase and can_transition:
            self._prev_phase = self.phase
            self.phase = new_phase
            self._phase_start_tick = self._tick_count

            # Reset move tracking on phase transitions to expansion/accumulation
            if new_phase in (self.EXPANSION, self.ACCUMULATION):
                self._move_start_price = fut_price
                self._move_peak_displacement = 0.0

        return self.phase

    # ── Phase-based entry policy ─────────────────────────────────────────
    @property
    def allows_entry(self) -> bool:
        """Does the current phase allow new entries?"""
        return self.phase in (self.EXPANSION, self.TREND)

    @property
    def is_aggressive(self) -> bool:
        """Should entry gates be relaxed? True only during EXPANSION."""
        return self.phase == self.EXPANSION

    @property
    def should_tighten_exits(self) -> bool:
        """Should existing trailing stops be tightened?"""
        return self.phase == self.EXHAUSTION

    @property
    def entry_policy(self) -> str:
        """Human-readable entry policy for the current phase."""
        policies = {
            self.ACCUMULATION: "NO ENTRY — market coiling, wait for expansion",
            self.EXPANSION:    "AGGRESSIVE ENTRY — breakout in progress, relax gates",
            self.TREND:        "PULLBACK ENTRY — enter on retests, don't chase",
            self.EXHAUSTION:   "NO NEW ENTRY — move is dying, tighten exits",
            self.DISTRIBUTION: "NO ENTRY — choppy redistribution, wait",
            self.UNKNOWN:      "WAITING — insufficient data for phase detection",
        }
        return policies.get(self.phase, "UNKNOWN")

    def reset(self):
        """Reset all state. Called on session start or after trade exit."""
        self.phase = self.UNKNOWN
        self._prev_phase = self.UNKNOWN
        self._phase_start_tick = 0
        self._tick_count = 0
        self._move_start_price = 0.0
        self._move_peak_displacement = 0.0
        self._velocity_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# TREND EXHAUSTION GATE — v29
# First-principles problem: when a trend reverses, the system takes 2-3 SL trades
# BEFORE the correct reversal trade because the engine fires the new direction
# based on early momentum signals alone (PREM_VEL, MACD flipping) before structural
# confirmation exists. Those early signals are noise from the pullback, not the reversal.
#
# This class tracks three exhaustion conditions:
#   1. RSI divergence (price HH/LL not confirmed by RSI) — thesis is weakening
#   2. PREM_VEL deceleration — premium momentum is fading even as price extends
#   3. ATR contraction after a run — range compressing = trend running out of fuel
#
# When a trend is in EXHAUSTION state AND the engine fires the OPPOSITE direction,
# a DIR_FLIP_LOCKOUT is activated. The flip is only allowed after N candles of
# consistent new-direction votes — confirming the reversal is structural, not noise.
#
# Result: the system waits for the reversal to PROVE itself before entering,
# rather than paying 2 SLs to discover that the reversal was real.
# ─────────────────────────────────────────────────────────────────────────────
class TrendExhaustionGate:
    """
    Detects when the current trend is exhausted and blocks premature flip entries.

    State machine:
      TRENDING  → normal operation, no restriction on direction changes
      EXHAUSTED → trend has run far and is losing momentum; flip entries require confirmation
      CONFIRMED → new direction held for MIN_CONFIRM_CANDLES; flip entry now permitted

    Exhaustion is triggered by ANY TWO of:
      - RSI divergence (bearish or bullish)
      - PremVel deceleration (velocity falling in trend direction for 3+ ticks)
      - ATR contraction (ATR shrinking to < 80% of its peak during this run)

    Confirmation clears exhaustion and resets state.
    """
    MIN_CONFIRM_CANDLES  = 3    # new direction must persist this many 1-min candles before entry allowed
    PREM_VEL_DECEL_TICKS = 4    # consecutive ticks of falling prem_vel triggers decel flag
    ATR_CONTRACT_PCT     = 0.80 # ATR must fall to <80% of run peak to count as contracting

    def __init__(self):
        self._run_direction   = "NEUTRAL"  # direction of the current trend run
        self._run_candles     = 0          # candles in current direction
        self._peak_atr        = 0.0        # ATR high-water mark during this run
        self._new_dir_candles = 0          # candles engine has voted opposite to run_direction
        self._new_dir         = "NEUTRAL"  # what the engine is voting now

        # Exhaustion flag inputs
        self._rsi_divergence  = False
        self._prem_decel      = False
        self._atr_contraction = False
        self._prem_vel_decel_count = 0
        self._last_prem_vel   = 0.0

        self.state            = "TRENDING"  # TRENDING | EXHAUSTED | CONFIRMED
        self.block_reason     = ""

    def reset(self):
        self.__init__()

    def _exhaustion_conditions_met(self) -> int:
        """Count how many exhaustion conditions are active. Need >= 2 to trigger."""
        return sum([self._rsi_divergence, self._prem_decel, self._atr_contraction])

    def update(self, engine_direction: str, candle_direction: str,
               rsi_bearish_div: bool, rsi_bullish_div: bool,
               prem_vel_score: float, current_atr: float, candle_closed: bool) -> None:
        """
        Called every tick. candle_direction only advances the run counter on candle close.
        engine_direction is the current dominant vote direction from NiftyEngine.
        """
        # ── Track ATR high-water mark for current run ──────────────────────
        if current_atr > self._peak_atr:
            self._peak_atr = current_atr
        if self._peak_atr > 0 and current_atr < self._peak_atr * self.ATR_CONTRACT_PCT:
            self._atr_contraction = True

        # ── Track premium velocity deceleration ────────────────────────────
        # prem_vel_score > 0.5 means premium is rising (CE). < 0.5 means PE.
        # Deceleration = score drifting toward 0.5 (neutral) from an extreme.
        if self._run_direction == "CE":
            # In a CE run, watch for prem_vel_score falling (CE momentum fading)
            if prem_vel_score < self._last_prem_vel and prem_vel_score < 0.65:
                self._prem_vel_decel_count += 1
            else:
                self._prem_vel_decel_count = max(0, self._prem_vel_decel_count - 1)
        elif self._run_direction == "PE":
            # In a PE run, watch for prem_vel_score rising (PE momentum fading)
            if prem_vel_score > self._last_prem_vel and prem_vel_score > 0.35:
                self._prem_vel_decel_count += 1
            else:
                self._prem_vel_decel_count = max(0, self._prem_vel_decel_count - 1)
        self._last_prem_vel = prem_vel_score
        self._prem_decel = (self._prem_vel_decel_count >= self.PREM_VEL_DECEL_TICKS)

        # ── RSI divergence flags ────────────────────────────────────────────
        if self._run_direction == "CE" and rsi_bearish_div:
            self._rsi_divergence = True
        elif self._run_direction == "PE" and rsi_bullish_div:
            self._rsi_divergence = True
        else:
            self._rsi_divergence = False

        # ── Update run candle counter on candle close ───────────────────────
        if candle_closed:
            if candle_direction == self._run_direction:
                self._run_candles += 1
                self._new_dir_candles = 0
                self._new_dir = "NEUTRAL"
            elif candle_direction != "NEUTRAL":
                if self._run_direction == "NEUTRAL":
                    # First direction — start a new run
                    self._run_direction = candle_direction
                    self._run_candles   = 1
                    self._peak_atr      = current_atr
                    self._rsi_divergence = False; self._prem_decel = False
                    self._atr_contraction = False; self._prem_vel_decel_count = 0
                elif candle_direction != self._run_direction:
                    # Potential flip — track how many candles are going new direction
                    if candle_direction == self._new_dir:
                        self._new_dir_candles += 1
                    else:
                        self._new_dir = candle_direction
                        self._new_dir_candles = 1

        # ── State machine ───────────────────────────────────────────────────
        opposite = "PE" if self._run_direction == "CE" else "CE"
        is_flip_attempt = (engine_direction == opposite and self._run_direction != "NEUTRAL")

        if self.state == "TRENDING":
            # Enter EXHAUSTED if trend has run for >= 3 candles AND >= 2 exhaustion signals
            if self._run_candles >= 3 and self._exhaustion_conditions_met() >= 2:
                self.state = "EXHAUSTED"
                conditions = []
                if self._rsi_divergence: conditions.append("RSI_DIV")
                if self._prem_decel:     conditions.append("PREM_DECEL")
                if self._atr_contraction: conditions.append("ATR_CONTRACT")
                self.block_reason = f"TrendExhaustion({self._run_direction} run={self._run_candles}c {'+'.join(conditions)}): need {self.MIN_CONFIRM_CANDLES} confirm candles"

        elif self.state == "EXHAUSTED":
            if self._new_dir_candles >= self.MIN_CONFIRM_CANDLES:
                # Reversal confirmed — reset and start new run
                self.state = "CONFIRMED"
                self.block_reason = ""
                # Reset for the new run
                self._run_direction   = self._new_dir
                self._run_candles     = self._new_dir_candles
                self._new_dir_candles = 0
                self._new_dir         = "NEUTRAL"
                self._peak_atr        = current_atr
                self._rsi_divergence  = False; self._prem_decel = False
                self._atr_contraction = False; self._prem_vel_decel_count = 0
            elif not is_flip_attempt:
                # Engine went back to same direction or NEUTRAL — reset exhaustion
                self.state = "TRENDING"
                self.block_reason = ""
                self._new_dir_candles = 0; self._new_dir = "NEUTRAL"

        elif self.state == "CONFIRMED":
            # After confirmation, act as TRENDING but in new direction
            self.state = "TRENDING"
            self.block_reason = ""

    def is_flip_blocked(self, proposed_direction: str) -> bool:
        """
        Returns True if proposed_direction is a flip from the established run
        AND the reversal hasn't been confirmed yet.
        """
        if self.state != "EXHAUSTED":
            return False
        opposite = "PE" if self._run_direction == "CE" else "CE"
        return proposed_direction == opposite

    @property
    def exhaustion_summary(self) -> str:
        if self.state == "TRENDING":
            return ""
        conds = []
        if self._rsi_divergence: conds.append("RSI_DIV")
        if self._prem_decel:     conds.append("PREM_DECEL")
        if self._atr_contraction: conds.append("ATR_CONTR")
        confirmed = f" new_dir={self._new_dir}({self._new_dir_candles}/{self.MIN_CONFIRM_CANDLES}c)" if self._new_dir != "NEUTRAL" else ""
        return f"[EXHAUST:{self._run_direction}/{'+'.join(conds) or 'none'}{confirmed}]"


# ─────────────────────────────────────────────────────────────────────────────
# PULLBACK DETECTOR — enter on retracement within trend
# ─────────────────────────────────────────────────────────────────────────────
class PullbackDetector:
    """
    Detects pullbacks within an established trend for better entry timing.

    Instead of entering at breakout peaks, wait for a 25-65% retracement
    of the recent swing. This gives:
      - Better entry price (lower premium)
      - Tighter effective SL (closer to structure)
      - Confirmation that the trend survives a test

    Logic:
      1. Track recent swing high/low over last N candles
      2. If trending CE and price pulls back 25-65% from high → pullback entry zone
      3. If trending PE and price bounces 25-65% from low → pullback entry zone
      4. If price retraces > 65%, trend may be breaking — don't enter
    """
    SWING_LOOKBACK = 20  # candles to identify swing

    def __init__(self):
        self._price_history: deque = deque(maxlen=200)
        self._swing_high = float('nan')
        self._swing_low = float('nan')
        self._in_pullback = False
        self._pullback_dir = "NEUTRAL"
        self._pullback_depth = 0.0

    def reset(self):
        self._price_history.clear()
        self._swing_high = float('nan')
        self._swing_low = float('nan')
        self._in_pullback = False
        self._pullback_dir = "NEUTRAL"
        self._pullback_depth = 0.0

    def update(self, price: float, trend_dir: str) -> None:
        """Update with current price and detected trend direction."""
        self._price_history.append(price)

        if len(self._price_history) < self.SWING_LOOKBACK:
            return

        recent = list(self._price_history)[-self.SWING_LOOKBACK:]
        self._swing_high = max(recent)
        self._swing_low = min(recent)
        swing_range = self._swing_high - self._swing_low

        if swing_range < 1.0:
            self._in_pullback = False
            self._pullback_dir = "NEUTRAL"
            return

        if trend_dir == "CE":
            # In uptrend: pullback = price drops from swing high
            retracement = (self._swing_high - price) / swing_range
            if PULLBACK_DEPTH_MIN <= retracement <= PULLBACK_DEPTH_MAX:
                self._in_pullback = True
                self._pullback_dir = "CE"
                self._pullback_depth = retracement
            else:
                self._in_pullback = False
        elif trend_dir == "PE":
            # In downtrend: pullback = price rises from swing low
            retracement = (price - self._swing_low) / swing_range
            if PULLBACK_DEPTH_MIN <= retracement <= PULLBACK_DEPTH_MAX:
                self._in_pullback = True
                self._pullback_dir = "PE"
                self._pullback_depth = retracement
            else:
                self._in_pullback = False
        else:
            self._in_pullback = False
            self._pullback_dir = "NEUTRAL"

    @property
    def in_pullback_zone(self) -> bool:
        return self._in_pullback

    @property
    def pullback_direction(self) -> str:
        return self._pullback_dir

    @property
    def depth(self) -> float:
        return self._pullback_depth


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# WIN PROBABILITY CALCULATOR (v17)
# Uses first-passage-time formula for Brownian motion with drift.
# Given SL, target, current volatility, and estimated drift → real P(win).
# ─────────────────────────────────────────────────────────────────────────────
WIN_PROB_MIN_ENTRY = 0.0   # v17.3: DISABLED — P(win) formula is broken (62% for both W and L)
IMM_MOMENTUM_MIN  = -0.05
IMM_MOMENTUM_MAX  = 0.75  # v22: was 0.60 — data proved 0.75 is the chasing boundary (2:1 BAD:GOOD)
IMM_MOMENTUM_PENALTY_ZONE = 0.12
IMM_MOMENTUM_HEAVY_ZONE  = 0.20

class WinProbabilityCalculator:
    """Two-layer probability system:

    Layer 1 — Lifetime P(win): First-passage-time formula.
      "Over the entire trade, will I hit target before SL?"
      P = [1 - exp(-2μb/σ²)] / [exp(2μa/σ²) - exp(-2μb/σ²)]

    Layer 2 — Immediate Momentum Score (IMS): [-1 to +1]
      "Is price actively moving in my direction RIGHT NOW?"
      Surgical filter: low bar blocks only obvious losers (momentum opposed),
      lets through winners even during brief consolidation.
    """

    def __init__(self):
        self._price_history: deque = deque(maxlen=60)
        self._ce_history: deque    = deque(maxlen=20)  # premium tracking
        self._pe_history: deque    = deque(maxlen=20)
        self._vwap_distance: float = 0.0
        self._momentum_strength: float = 0.0
        self._prem_vel_alignment: float = 0.0
        self._atr_ratio: float = 1.0

    def reset(self):
        self._price_history.clear()
        self._ce_history.clear()
        self._pe_history.clear()
        self._vwap_distance = 0.0
        self._momentum_strength = 0.0
        self._prem_vel_alignment = 0.0
        self._atr_ratio = 1.0

    def update_price(self, price: float):
        self._price_history.append(price)

    def update_premium(self, ce_ltp: float, pe_ltp: float):
        """Track raw premium prices for immediate momentum scoring."""
        if ce_ltp > 0: self._ce_history.append(ce_ltp)
        if pe_ltp > 0: self._pe_history.append(pe_ltp)

    def update_context(self, vwap_distance: float, momentum: float,
                       prem_vel_aligned: bool, atr_ratio: float):
        self._vwap_distance = vwap_distance
        self._momentum_strength = momentum
        self._prem_vel_alignment = 1.0 if prem_vel_aligned else -0.5
        self._atr_ratio = max(0.5, min(2.0, atr_ratio))

    # ── Immediate Momentum Score ──────────────────────────────────────────
    def immediate_momentum(self, direction: str, vix_scale: float = 1.0) -> float:
        """Score from -1 (strongly opposed) to +1 (strongly favorable).

        Looks at last 5 ticks of futures + last 5 ticks of relevant premium.
        This answers: "Is price moving in my direction RIGHT NOW?"

        Design: low bar. Score > -0.10 passes. This only blocks entries
        where momentum is clearly moving AGAINST you — the obvious losers.
        Winners in brief consolidation (score ~0) still pass.

        v25-LIVE: vix_scale normalizes for VIX regime. At VIX 26 (HIGH),
        raw point moves are 2x VIX 13 (MID). Without scaling, IMS is
        permanently inflated, triggering CHASING blocks on normal ticks.
        """
        prices = list(self._price_history)
        if len(prices) < 10:
            return 0.0  # not enough data, allow entry

        # v28: Use FRACTIONAL velocity (percentage of price) instead of absolute points.
        # Old code: (prices[-1] - prices[-5]) / 5.0 → absolute NIFTY points.
        # A 2pt move at NIFTY 23000 is 0.0087%, at 25000 is 0.008%. The VIX scale
        # normalization was a band-aid. Fractional change is self-normalizing.
        base_price = prices[-5]
        if base_price <= 0:
            base_price = prices[-1]
        if base_price <= 0:
            return 0.0

        # --- Futures fractional velocity: last 5 ticks ---
        micro_vel_5 = (prices[-1] - prices[-5]) / base_price
        if direction == "PE":
            micro_vel_5 = -micro_vel_5

        # --- Futures fractional velocity: last 10 ticks ---
        base_10 = prices[-10] if prices[-10] > 0 else base_price
        micro_vel_10 = (prices[-1] - prices[-10]) / base_10
        if direction == "PE":
            micro_vel_10 = -micro_vel_10

        # --- Micro-acceleration: is the 5-tick move building or fading? ---
        if len(prices) >= 15:
            prev_base = prices[-11] if prices[-11] > 0 else base_price
            prev_vel_5 = (prices[-6] - prices[-11]) / prev_base
            if direction == "PE":
                prev_vel_5 = -prev_vel_5
            micro_accel = micro_vel_5 - prev_vel_5
        else:
            micro_accel = 0.0

        # --- Premium micro-velocity: is the option premium itself rising? ---
        prem_hist = list(self._ce_history) if direction == "CE" else list(self._pe_history)
        prem_vel = 0.0
        if len(prem_hist) >= 5 and prem_hist[-5] > 0:
            prem_vel = (prem_hist[-1] - prem_hist[-5]) / prem_hist[-5]

        # --- Micro price position: where in the last 10-tick range are we? ---
        recent_10 = prices[-10:]
        r_high = max(recent_10)
        r_low = min(recent_10)
        r_range = r_high - r_low
        if r_range > 0.01:
            position = (prices[-1] - r_low) / r_range
            if direction == "CE":
                pos_score = 0.3 - position * 0.6
            else:
                pos_score = position * 0.6 - 0.3
        else:
            pos_score = 0.0

        # --- Combine: fractional velocities × calibrated weights ---
        # v28: Fractional values are ~100x smaller than old absolute values.
        # Recalibrated weights: multiply by ~100 to keep same output range [-1, +1].
        # vix_scale is no longer needed — fractional change is self-normalizing.
        score = (
            micro_vel_5 * 300.0 +     # last 5 ticks: strongest signal
            micro_vel_10 * 150.0 +     # last 10 ticks: confirms direction
            micro_accel * 200.0 +      # acceleration: is move building?
            prem_vel * 5.0 +           # premium confirmation (already fractional)
            pos_score * 0.5            # where in micro-range (already normalized)
        )

        return max(-1.0, min(1.0, score))

    # ── Drift estimation (for lifetime P(win)) ───────────────────────────
    def estimate_drift(self, direction: str) -> float:
        """Estimate drift (μ) blending short-term and medium-term momentum.
        Short-term (5 ticks) weighted 50% — answers "what's happening NOW".
        Medium-term (20 ticks) weighted 30% — confirms sustained direction.
        Context factors 20% — structural edge."""

        if len(self._price_history) < 20:
            return 0.0

        prices = list(self._price_history)
        sign = -1.0 if direction == "PE" else 1.0

        # Short-term velocity: last 5 ticks
        short_vel = sign * (prices[-1] - prices[-5]) / 5.0

        # Medium-term velocity: last 20 ticks
        med_vel = sign * (prices[-1] - prices[-20]) / 20.0

        # Acceleration: is short-term building on medium-term?
        if len(prices) >= 40:
            vel_recent = sign * (prices[-1] - prices[-10]) / 10.0
            vel_older = sign * (prices[-20] - prices[-30]) / 10.0
            acceleration = vel_recent - vel_older
        else:
            acceleration = 0.0

        # VWAP tailwind/headwind
        vwap_factor = 0.0
        if direction == "CE" and self._vwap_distance < -0.3:
            vwap_factor = min(abs(self._vwap_distance) * 0.1, 0.3)
        elif direction == "PE" and self._vwap_distance > 0.3:
            vwap_factor = min(abs(self._vwap_distance) * 0.1, 0.3)
        elif direction == "CE" and self._vwap_distance > 1.5:
            vwap_factor = -0.15
        elif direction == "PE" and self._vwap_distance < -1.5:
            vwap_factor = -0.15

        # Premium confirmation
        prem_factor = self._prem_vel_alignment * 0.15

        # ATR regime
        atr_factor = 0.0
        if self._atr_ratio > 1.2:
            atr_factor = 0.1
        elif self._atr_ratio < 0.7:
            atr_factor = -0.1

        # Combine: short-term 50%, medium-term 30%, context 20%
        raw_drift = (
            short_vel * 0.50 +        # what's happening NOW (dominant)
            med_vel * 0.30 +           # sustained direction
            acceleration * 0.10 +     # momentum building?
            vwap_factor * 0.04 +      # structural
            prem_factor * 0.04 +      # premium confirmation
            atr_factor * 0.02         # vol regime
        )

        # Critical correction: futures drift ≠ premium drift
        # ATM delta ~0.5, and drift is noisy over short windows.
        # Apply conservative discount (0.15) and tight cap (±0.015).
        # The first-passage formula is exponentially sensitive to drift/sigma²,
        # so even small overestimation creates wildly optimistic probabilities.
        discounted = raw_drift * 0.15
        return max(-0.015, min(0.015, discounted))

    # ── Lifetime P(win) ───────────────────────────────────────────────────
    def calculate(self, direction: str, sl_pts: float, tgt_pts: float,
                  atr: float) -> float:
        """First-passage-time P(win): probability of hitting target before SL."""
        if atr <= 0 or sl_pts <= 0 or tgt_pts <= 0:
            return sl_pts / (sl_pts + tgt_pts) if (sl_pts + tgt_pts) > 0 else 0.5

        sigma_per_tick = (atr * 1.2533) / math.sqrt(216)
        # Floor sigma at 0.15 to prevent extreme probabilities from tiny ATR
        sigma_per_tick = max(sigma_per_tick, 0.15)
        sigma2 = sigma_per_tick ** 2

        if sigma2 < 1e-10:
            return sl_pts / (sl_pts + tgt_pts)

        mu = self.estimate_drift(direction)
        a = tgt_pts   # distance to target
        b = sl_pts    # distance to stop loss

        if abs(mu) < 1e-8:
            return b / (a + b)

        # First-passage-time: P(hit +a before -b | start at 0, drift μ)
        # = [1 - exp(2μb/σ²)] / [exp(-2μa/σ²) - exp(2μb/σ²)]
        # With positive μ (favorable drift): P approaches 1
        # With negative μ (opposing drift): P approaches 0
        try:
            alpha = 2.0 * mu / sigma2
            exp_a = max(-50, min(50, alpha * a))   # 2μa/σ²
            exp_b = max(-50, min(50, alpha * b))   # 2μb/σ²

            numerator = 1.0 - math.exp(exp_b)
            denominator = math.exp(-exp_a) - math.exp(exp_b)

            if abs(denominator) < 1e-12:
                return b / (a + b)

            prob = numerator / denominator
            return max(0.05, min(0.95, prob))

        except (OverflowError, ValueError):
            return b / (a + b)

    def expected_value(self, p_win: float, sl_pts: float, tgt_pts: float) -> float:
        return p_win * tgt_pts - (1.0 - p_win) * sl_pts


class NiftyEngine:
    def __init__(self):
        self.candles        = CandleAggregator()
        self.candles_5m     = FiveMinCandleAggregator()
        self.vwap_signal    = FuturesVWAPSignal()
        self.rsi_signal     = FuturesRSISignal()
        self.momentum       = FuturesMomentumSignal()
        self.opt_oi         = OptionsOIStructure()
        self.prem_vel       = PremiumVsForwardSignal()
        self.vwap_retest    = VWAPRetestSignal()
        self.max_pain       = MaxPainOIWall()
        self.structure      = FuturesPriceStructure()
        self.atr_calc       = ATRCalculator()
        self.transition     = ConvictionTransitionDetector()
        self.rsi_divergence = RSIDivergenceDetector()
        self.macd_hist      = MACDHistogramSignal()
        self.supertrend     = SupertrendSignal()
        self.bb_squeeze     = BollingerSqueezeSignal()
        self.adx_filter     = ADXTrendFilter()
        self.breakout15     = FifteenMinBreakout()
        self.orb            = OpeningRangeBreakout()
        self.fut_oi_buildup = FuturesOIBuildup()
        self.coi_pcr        = COIPCRSignal()
        self.pcp            = PutCallParitySignal()
        self.avwap          = AnchoredVWAPSignal()
        self.straddle_gamma = StraddleGammaSignal()
        self.fut_vel        = FuturesVelocitySignal()   # v20: signal #20 — is futures actually moving?
        self.fut_vel.set_atr_ref(self.atr_calc)   # v28 Innovation E: ATR-normalized thresholds
        self.vix_tracker    = VIXTracker()
        self.thesis_tracker = ThesisTracker()
        self.price_velocity = PriceVelocityFilter()
        self.momentum_burst = MomentumBurstDetector()  # v18.1: CUSUM change-point detection
        self.kaufman_er     = KaufmanEfficiencyRatio()   # v18.3: noise vs directional gate
        self.win_prob_calc  = WinProbabilityCalculator()
        self.market_regime  = MarketRegimeDetector()
        self.pullback       = PullbackDetector()
        self.prem_divergence = PremiumDivergenceFilter()  # v18: direction confirmation
        # v28 Innovation A: Market Phase State Machine — context layer above signals
        self.phase_tracker  = MarketPhaseTracker()
        # v29: Trend Exhaustion Gate — blocks premature direction-flip entries
        self.exhaustion_gate = TrendExhaustionGate()
        self.vwap_retest.set_vwap_ref(self.vwap_signal)
        self.market_regime.set_refs(self.adx_filter, self.atr_calc,
                                    self.bb_squeeze, self.straddle_gamma)
        self.phase_tracker.set_refs(self.atr_calc, self.adx_filter,
                                    self.bb_squeeze, self.straddle_gamma,
                                    self.fut_vel, self.market_regime)

        self._tick_count    = 0
        self._oi_snaps_taken = 0
        self._last_result: Optional[EngineResult] = None
        self._last_signals: list = []  # v25-LIVE: for gate_diagnostics()
        self._session_start = time.monotonic()

        self._last_rsi_vote        = SignalVote("RSI",        "NEUTRAL", 0.0, "cold")
        self._last_macd_vote       = SignalVote("MACD_HIST",  "NEUTRAL", 0.0, "cold")
        self._last_supertrend_vote = SignalVote("SUPERTREND", "NEUTRAL", 0.0, "cold")
        self._last_bb_squeeze_vote = SignalVote("BB_SQUEEZE", "NEUTRAL", 0.0, "cold")
        self._last_adx_vote        = SignalVote("ADX",        "NEUTRAL", 0.0, "cold")
        self._last_breakout15_vote = SignalVote("BREAKOUT15", "NEUTRAL", 0.0, "cold")
        self._last_prem_vel_reason = ""

    def reset_session(self):
        self.candles.reset(); self.candles_5m.reset()
        self.vwap_signal.reset(); self.rsi_signal.reset()
        self.momentum.reset(); self.opt_oi.reset(); self.prem_vel.reset()
        self.vwap_retest.reset(); self.max_pain.reset(); self.structure.reset()
        self.atr_calc.reset(); self.transition.reset(); self.rsi_divergence.reset()
        self.macd_hist.reset(); self.supertrend.reset()
        self.bb_squeeze.reset(); self.adx_filter.reset(); self.breakout15.reset()
        self.orb.reset(); self.fut_oi_buildup.reset(); self.coi_pcr.reset()
        self.pcp.reset(); self.avwap.reset(); self.straddle_gamma.reset(); self.fut_vel.reset()
        self.vix_tracker.reset(); self.thesis_tracker.reset(); self.price_velocity.reset()
        self.win_prob_calc.reset(); self.market_regime.reset(); self.pullback.reset()
        self.prem_divergence.reset(); self.momentum_burst.reset(); self.kaufman_er.reset()
        self.phase_tracker.reset()  # v28: reset phase state machine
        self.exhaustion_gate.reset()  # v29: reset trend exhaustion gate
        self._tick_count = 0; self._oi_snaps_taken = 0
        self._last_result = None; self._session_start = time.monotonic()
        self._last_rsi_vote        = SignalVote("RSI",        "NEUTRAL", 0.0, "cold")
        self._last_macd_vote       = SignalVote("MACD_HIST",  "NEUTRAL", 0.0, "cold")
        self._last_supertrend_vote = SignalVote("SUPERTREND", "NEUTRAL", 0.0, "cold")
        self._last_bb_squeeze_vote = SignalVote("BB_SQUEEZE", "NEUTRAL", 0.0, "cold")
        self._last_adx_vote        = SignalVote("ADX",        "NEUTRAL", 0.0, "cold")
        self._last_breakout15_vote = SignalVote("BREAKOUT15", "NEUTRAL", 0.0, "cold")
        self._last_prem_vel_reason = ""

    def _emit(self, result: EngineResult) -> EngineResult:
        """v28 Innovation A: Finalize every EngineResult before return.

        Centralizes result finalization so that market_phase is ALWAYS injected
        into every result — entry-allowed AND blocked. Without this, each of the
        16+ return paths would need a manual phase assignment (error-prone).

        Why a helper instead of modifying EngineResult.__post_init__:
          - Phase comes from engine state (self.phase_tracker), not from constructor args.
          - Post-init can't access engine instance. A method can.
        """
        result.market_phase = self.phase_tracker.phase
        self._last_result = result
        return result

    def update(self, snap, futures_price: float, futures_oi: float = 0.0,
               signal_weights: dict = None, vix_value: float = 0.0) -> EngineResult:
        self._tick_count += 1

        fut = futures_price if futures_price > 0 else snap.spot
        ce_ltp = snap.atm_ce_ltp if not math.isnan(snap.atm_ce_ltp) else 0.0
        pe_ltp = snap.atm_pe_ltp if not math.isnan(snap.atm_pe_ltp) else 0.0

        # v18.1: Feed premium divergence filter every tick (now with futures basis)
        self.prem_divergence.update(ce_ltp, pe_ltp, snap.spot, futures=fut)

        self.vix_tracker.update(vix_value=vix_value, spot=snap.spot,
                                 ce_ltp=ce_ltp, pe_ltp=pe_ltp)

        self.price_velocity.update(fut)
        self.momentum_burst.update(fut)  # CUSUM fed every tick
        self.kaufman_er.update(fut)      # Kaufman ER fed every tick
        self.win_prob_calc.update_price(fut)
        s20 = self.fut_vel.update(fut)   # v20: futures velocity signal — is price actually moving?
        self.win_prob_calc.update_premium(ce_ltp, pe_ltp)

        completed_candle = self.candles.update(fut)
        if completed_candle:
            self.atr_calc.on_candle(completed_candle)
            self._last_rsi_vote        = self.rsi_signal.on_candle(completed_candle)
            self._last_macd_vote       = self.macd_hist.on_candle(completed_candle)
            self._last_supertrend_vote = self.supertrend.on_candle(completed_candle)
            self._last_bb_squeeze_vote = self.bb_squeeze.on_candle(completed_candle)
            self._last_adx_vote        = self.adx_filter.on_candle(completed_candle)
            self.rsi_divergence.on_candle(completed_candle, self.rsi_signal.rsi)
            b15_vote = self.breakout15.on_1min_candle(completed_candle)
            if b15_vote is not None: self._last_breakout15_vote = b15_vote
            self.orb.on_1min_candle(completed_candle)
            self.avwap.on_candle(completed_candle)
            self.candles_5m.on_1min_candle(completed_candle)

        s1  = self.vwap_signal.update(fut)
        s2  = self._last_rsi_vote        or self.rsi_signal.current_vote()
        s3  = self.momentum.update(fut)
        s4, smart_money = self.opt_oi.update(snap)
        s5  = self.prem_vel.update(fut, ce_ltp, pe_ltp)
        s6  = self.vwap_retest.update(fut)
        s7  = self.max_pain.update(snap)
        s8  = self.structure.update(fut)
        s9  = self._last_macd_vote       or self.macd_hist.current_vote()
        s10 = self._last_supertrend_vote or self.supertrend.current_vote()
        s11 = self._last_bb_squeeze_vote or self.bb_squeeze.current_vote()
        s12 = self._last_adx_vote        or self.adx_filter.current_vote()
        s13 = self._last_breakout15_vote
        s14 = self.orb.update(fut)
        s15 = self.fut_oi_buildup.update(fut, futures_oi, ce_oi=snap.total_call_oi, pe_oi=snap.total_put_oi)
        s16 = self.coi_pcr.update(snap.total_call_oi, snap.total_put_oi)
        s17 = self.pcp.update(snap.atm_strike, ce_ltp, pe_ltp, fut)
        s18 = self.avwap.update(fut)
        s19 = self.straddle_gamma.update(ce_ltp, pe_ltp)

        # v17.2: Update market regime and pullback detector
        # Detect candle direction for choppiness tracking
        candle_dir = "NEUTRAL"
        if self.candles.closed_candles:
            lc = self.candles.closed_candles[-1]
            if lc.close > lc.open + 0.5:
                candle_dir = "CE"
            elif lc.close < lc.open - 0.5:
                candle_dir = "PE"
        market_regime = self.market_regime.update(candle_dir, self.candles.n_candles)
        # Pullback needs a trend direction hint — use ADX DI direction
        adx_dir = "CE" if self.adx_filter.di_plus > self.adx_filter.di_minus else "PE" if self.adx_filter.di_minus > self.adx_filter.di_plus else "NEUTRAL"
        self.pullback.update(fut, adx_dir if self.market_regime.is_trending else "NEUTRAL")

        # v28 Innovation A: Update market phase state machine BEFORE any entry logic.
        # Phase determines entry policy — EXPANSION relaxes gates, EXHAUSTION blocks.
        # Must run after: market_regime, atr_calc, adx_filter, straddle_gamma, fut_vel
        # (all of which have already been updated above).
        straddle_vel = self.straddle_gamma.straddle_velocity
        _market_phase = self.phase_tracker.update(fut, straddle_vel)

        # v29: Update TrendExhaustionGate — tracks whether current trend is exhausted
        # and blocks premature direction-flip entries until reversal is confirmed.
        # prem_vel score: s5.score if s5.direction=="CE" else (1-s5.score) gives CE-normalized velocity.
        _pv_score = s5.score if s5.direction == "CE" else (1.0 - s5.score) if s5.direction == "PE" else 0.5
        self.exhaustion_gate.update(
            engine_direction  = "NEUTRAL",   # will be filled after vote counting; update again below
            candle_direction  = candle_dir,
            rsi_bearish_div   = self.rsi_divergence.bearish_divergence,
            rsi_bullish_div   = self.rsi_divergence.bullish_divergence,
            prem_vel_score    = _pv_score,
            current_atr       = self.atr_calc.atr if self.atr_calc.atr > 0 else 5.0,
            candle_closed     = completed_candle is not None,
        )

        # v28 Innovation J: Store COI strength % for dashboard/diagnostics
        self._coi_strength_pct = self.coi_pcr.strength_pct

        # v28 Innovation K: IV Skew Proxy from strike prices
        # (Aristotle Phase 3 — Reconstruct from Atoms):
        #   Compare OTM PE vs equidistant OTM CE price.
        #   If PE is pricier → market expects downside → PE bias.
        #   If CE is pricier → market expects upside → CE bias.
        #   Uses strikes already in snapshot — zero additional data needed.
        #   Ratio > 1.2 = bearish skew, < 0.8 = bullish skew, else neutral.
        #   Access via _StrikesProxy: snap.call_strikes / snap.put_strikes
        self._iv_skew_ratio = 1.0  # default: neutral
        _skew_atm = snap.atm_strike
        if _skew_atm > 0 and hasattr(snap, 'call_strikes') and hasattr(snap, 'put_strikes'):
            # OTM CE = 2 steps above ATM, OTM PE = 2 steps below ATM
            # strike_step is typically 50 for NIFTY
            _step = 50  # NIFTY strike step
            _otm_ce_k = _skew_atm + _step * 2
            _otm_pe_k = _skew_atm - _step * 2
            _otm_ce_proxy = snap.call_strikes.get(_otm_ce_k)
            _otm_pe_proxy = snap.put_strikes.get(_otm_pe_k)
            _otm_ce_ltp = _otm_ce_proxy.ltp if _otm_ce_proxy else 0
            _otm_pe_ltp = _otm_pe_proxy.ltp if _otm_pe_proxy else 0
            if _otm_ce_ltp > 1.0 and _otm_pe_ltp > 1.0:
                self._iv_skew_ratio = _otm_pe_ltp / _otm_ce_ltp

        self._last_prem_vel_reason = s5.reason
        signals = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20]

        reversal_warnings = []
        if self.rsi_divergence.bearish_divergence: reversal_warnings.append(f"RSI_BEAR_DIV:{self.rsi_divergence.warning}")
        if self.rsi_divergence.bullish_divergence: reversal_warnings.append(f"RSI_BULL_DIV:{self.rsi_divergence.warning}")
        if "FutUp_CE_lag" in self._last_prem_vel_reason: reversal_warnings.append(f"PREM_BEAR_DIV")
        if "FutDn_PE_lag" in self._last_prem_vel_reason: reversal_warnings.append(f"PREM_BULL_DIV")
        if "LH_only" in s8.reason: reversal_warnings.append(f"STRUCT_LH_WARN")
        if "HL_only" in s8.reason: reversal_warnings.append(f"STRUCT_HL_WARN")
        if self.vix_tracker.is_vix_spike():
            reversal_warnings.append(f"VIX_SPIKE:{self.vix_tracker.vix:.1f}")

        signals = apply_freshness_decay(signals)
        self._last_signals = signals  # v25-LIVE: store for gate_diagnostics()
        vote_detail = [f"{s.name}:{s.direction}({s.score:.0%})[{s.reason}]" for s in signals]

        vix_sl_mult = self.vix_tracker.sl_multiplier()
        sl_pts  = self.atr_calc.sl_points(vix_mult=vix_sl_mult)
        sl_pct  = self.atr_calc.sl_pct_of_premium(vix_mult=vix_sl_mult)
        tgt_pts = sl_pts * 3.0

        if smart_money in ("CONDOR", "STRADDLE"):
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=0, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"SmartMoney:{smart_money}", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        if self._tick_count < 30:
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=0, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"Warming up ({self._tick_count}/30)", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        if self.candles.n_candles < MIN_CANDLES_BEFORE_ENTRY:
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=0, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"Candle warmup", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        if not snap.is_clean:
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=0, vote_detail=vote_detail, entry_allowed=False, blocked_reason="dirty_data", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        # v18.3-LIVE: Low-vol converted to soft penalty (was hard block killing all afternoon entries)
        if self.atr_calc.is_low_volatility():
            pass  # handled by regime penalty later

        if ce_ltp > 0 and pe_ltp > 0 and snap.spot > 0:
            straddle_ratio = (ce_ltp + pe_ltp) / snap.spot
            if straddle_ratio < 0.006:
                result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=0, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"Low-IV Regime: straddle/spot={straddle_ratio:.4f} < 0.006", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
                return self._emit(result)

        if self.vix_tracker.regime == VIXRegime.EXTREME and self.vix_tracker.is_vix_spike():
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=0, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"VIX_EXTREME+SPIKE: VIX={self.vix_tracker.vix:.1f}", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        # v17.3: Straddle velocity — soft conviction modifier, NOT hard block
        # Data from 69 trades:
        #   SV < 0:       +5.3pts/trade (best)
        #   SV 0-0.05:    +4.5pts/trade (good)
        #   SV 0.05-0.15: -1.1pts/trade (BAD — the real problem bucket)
        #   SV > 0.15:    +0.9pts/trade (marginal but positive)
        # Hard-blocking SV>0.15 would lose +21.7pts. Use conviction penalty instead.
        # (straddle_vel already computed above for phase_tracker — reuse, not recompute)

        # ── v19-LIVE: FAST ENTRY PATH ────────────────────────────────────────
        # THE CORE FIX for live trading: 19 signals take too long to agree.
        # By the time they do, the move already happened (IMS too high).
        #
        # FAST PATH: When the 2 fastest LEADING indicators (options flow)
        # strongly agree on direction + at least 1 TREND signal confirms,
        # enter immediately without waiting for full consensus.
        #
        # PREM_VEL (w=1.8): premium velocity — leads price by 10-30s
        # STRADDLE_GAMMA (w=1.8): straddle expansion — shows real flow
        # These are the BEST predictors (Trail SL winners avg 20.4pts).
        # When both fire together + a trend signal agrees, that IS the signal.
        # ─────────────────────────────────────────────────────────────────────
        FAST_ENTRY_MIN_SCORE = 0.55   # v19.2: back to 0.55 — 0.50 let in noise
        FAST_IMS_MAX = 0.60
        FAST_CONVICTION_FLOOR = 0.58

        # v28: Relaxed fast path SV gate — allow mild expansion (SV < 0.10)
        # v21 hard block at SV >= 0 contradicted STRADDLE_GAMMA (which fires on expansion).
        # STRADDLE_GAMMA + PREM_VEL both strong + SV < 0.10 = early expansion catch, not chasing.
        # Only block fast path when SV >= 0.10 (genuine mid-expansion = late entry risk).
        # v28 Innovation A: During EXPANSION phase, allow SV up to 0.15 (breakout is confirmed).
        #                    During EXHAUSTION phase, force sv_allows_entry = False to skip fast path.
        if _market_phase == MarketPhaseTracker.EXHAUSTION:
            sv_allows_entry = False  # EXHAUSTION = move dying → no fast entries, caught by block below
        else:
            FAST_SV_MAX = 0.15 if _market_phase == MarketPhaseTracker.EXPANSION else 0.10
            sv_allows_entry = straddle_vel < FAST_SV_MAX

        fast_leaders = []
        if sv_allows_entry:
            if s5.direction != "NEUTRAL" and s5.score >= FAST_ENTRY_MIN_SCORE:
                fast_leaders.append(s5)   # PREM_VEL
            if s19.direction != "NEUTRAL" and s19.score >= FAST_ENTRY_MIN_SCORE:
                fast_leaders.append(s19)  # STRADDLE_GAMMA

        fast_entry_triggered = False
        # v19.2: Need BOTH PREM_VEL and STRADDLE_GAMMA agreeing (not just 1)
        # 1 leader was too loose — let in 13 trades/day. Need both options-flow
        # indicators confirming = genuine institutional flow, not noise.
        if len(fast_leaders) >= 2:
            leader_dirs = [s.direction for s in fast_leaders]
            if leader_dirs[0] == leader_dirs[1]:
                fast_dir = leader_dirs[0]

                # Need at least 1 TREND signal confirming same direction
                # VWAP(s1), STRUCTURE(s8), BREAKOUT15(s13), SUPERTREND(s10)
                any_confirms = any(
                    s.direction == fast_dir
                    for s in [s1, s8, s13, s10]
                    if s.direction != "NEUTRAL"
                )

                if any_confirms:
                    # Compute fast entry score from leading signals
                    fast_score = sum(s.score for s in fast_leaders) / len(fast_leaders)

                    # CUSUM momentum burst boost
                    if self.momentum_burst.direction == fast_dir and self.momentum_burst.strength > 0.3:
                        fast_score = min(fast_score + 0.04, 0.99)

                    # Premium divergence boost/penalty
                    div_confirms, div_strength, _ = self.prem_divergence.confirms_direction(fast_dir)
                    if div_confirms and div_strength > 0.2:
                        fast_score = min(fast_score + 0.03, 0.99)
                    elif not div_confirms:
                        fast_score = max(0.0, fast_score - 0.03)

                    # Kaufman ER boost if directional and aligned
                    if self.kaufman_er.is_directional and len(self.kaufman_er._prices) > self.kaufman_er.PERIOD:
                        er_net = self.kaufman_er._prices[-1] - self.kaufman_er._prices[-1 - self.kaufman_er.PERIOD]
                        er_dir = "CE" if er_net > 0 else "PE" if er_net < 0 else "NEUTRAL"
                        if er_dir == fast_dir:
                            fast_score = min(fast_score + 0.03, 0.99)

                    # Regime checks — still respect dangerous regimes
                    if self.market_regime.is_low_vol:
                        fast_score = max(0.0, fast_score - 0.12)
                    elif self.market_regime.is_choppy and not self.straddle_gamma.is_expanding:
                        fast_score = max(0.0, fast_score - 0.10)

                    # Stale check
                    is_fresh, _ = self.transition.check(fast_dir, fast_score)

                    # IMS check — RELAXED for fast path (we're catching the move early)
                    # v26: premium-level scale — 220pt options have larger absolute moves than 80pt.
                    # Without this, IMS is inflated on high-premium options causing false CHASING blocks.
                    _prem_level = (ce_ltp + pe_ltp) / 200.0 if (ce_ltp > 0 and pe_ltp > 0) else 1.0
                    _ims_scale = self.vix_tracker.sl_multiplier() * max(_prem_level, 0.5)
                    ims = self.win_prob_calc.immediate_momentum(fast_dir, vix_scale=_ims_scale)

                    # Mild IMS penalty (but much more lenient than regular path)
                    if ims >= 0.25:
                        fast_score = max(0.0, fast_score - 0.02)
                    elif ims >= 0.15:
                        fast_score = max(0.0, fast_score - 0.01)

                    if (is_fresh and
                        fast_score >= FAST_CONVICTION_FLOOR and
                        ims >= IMM_MOMENTUM_MIN and
                        ims <= FAST_IMS_MAX):

                        fast_votes = sum(1 for s in signals if s.direction == fast_dir)
                        b15_tgt_mult = 3.5 if s13.direction == fast_dir else 3.0
                        fast_tgt = sl_pts * b15_tgt_mult

                        entry_thesis = {
                            "direction": fast_dir,
                            "entry_mode": "FAST_LEADING",
                            "leaders": [s.name for s in fast_leaders],
                            "leader_scores": [round(s.score, 2) for s in fast_leaders],
                            "imm_momentum": ims,
                            "kaufman_er": round(self.kaufman_er.er, 3),
                            "vix_regime": self.vix_tracker.regime,
                            "fut_vel_disp": round(self.fut_vel.displacement, 3),
                        }

                        result = EngineResult(
                            direction=fast_dir, score=fast_score, votes_for=fast_votes,
                            vote_detail=vote_detail, entry_allowed=True, blocked_reason="",
                            suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct,
                            suggested_tgt_pts=fast_tgt,
                            smart_money_bias=smart_money,
                            reversal_warnings=reversal_warnings,
                            vix_regime=self.vix_tracker.regime,
                            market_regime=self.market_regime.regime,
                            entry_thesis=entry_thesis, probability=0.0,
                            imm_momentum=ims,
                        )
                        # v28 Teardown fix: Use _emit() instead of manual assignment.
                        # _emit() centralizes result finalization (phase injection + _last_result).
                        # Manual duplication here was a maintenance hazard — if _emit() gains
                        # future logic (metrics, validation), this path would silently miss it.
                        self._emit(result)
                        fast_entry_triggered = True

        if fast_entry_triggered:
            return result
        # ── END FAST ENTRY PATH ──────────────────────────────────────────────

        SIGNAL_CATEGORIES = {
            "VWAP": "TREND", "VWAP_RETEST": "TREND", "STRUCTURE": "TREND",
            "AVWAP": "TREND", "SUPERTREND": "TREND", "BREAKOUT15": "TREND", "ORB": "TREND",
            "RSI": "MOMENTUM", "MOMENTUM": "MOMENTUM", "MACD_HIST": "MOMENTUM",
            "BB_SQUEEZE": "MOMENTUM", "ADX": "MOMENTUM",
            "OPT_OI": "OPTIONS", "PREM_VEL": "OPTIONS", "MAX_PAIN": "OPTIONS",
            "MKT_OI": "OPTIONS", "COI_PCR": "OPTIONS", "PCP": "OPTIONS",
            "STRADDLE_GAMMA": "OPTIONS",
            "FUT_VEL": "ENERGY",
        }

        CATEGORY_CAPS = {
            "MOMENTUM": 1.0,  # v28: tightened from 1.2 — RSI+MACD+Momentum heavily correlated
            "TREND":    2.5,  # v17.5: trend signals partially correlated
            "OPTIONS":  5.0,  # v28: raised from 4.5 — options flow is the real edge
            "ENERGY":   0.8,  # v28: single signal FUT_VEL now active
        }

        ce_votes = [(s.name, s.score) for s in signals if s.direction == "CE"]
        pe_votes = [(s.name, s.score) for s in signals if s.direction == "PE"]
        n_ce = len(ce_votes)
        n_pe = len(pe_votes)

        # v28 Innovation 1: Regime-adaptive category multipliers
        # In TRENDING markets, TREND signals matter more and MOMENTUM less.
        # In BREAKOUT, OPTIONS flow is king. In CHOPPY/LOW_VOL, everything is dampened.
        REGIME_MULTIPLIERS = {
            "TRENDING":  {"TREND": 1.3, "OPTIONS": 1.1, "MOMENTUM": 0.7, "ENERGY": 1.2},
            "BREAKOUT":  {"TREND": 1.1, "OPTIONS": 1.3, "MOMENTUM": 0.8, "ENERGY": 1.1},
            "RANGE":     {"TREND": 0.8, "OPTIONS": 1.0, "MOMENTUM": 0.8, "ENERGY": 0.9},
            "CHOPPY":    {"TREND": 0.7, "OPTIONS": 0.9, "MOMENTUM": 0.5, "ENERGY": 0.7},
            "LOW_VOL":   {"TREND": 0.7, "OPTIONS": 0.7, "MOMENTUM": 0.5, "ENERGY": 0.5},
        }
        _regime_mults = REGIME_MULTIPLIERS.get(self.market_regime.regime, {})

        def calculate_capped_weight(votes, weights_dict):
            raw_cats = {"MOMENTUM": 0.0, "TREND": 0.0, "OPTIONS": 0.0, "ENERGY": 0.0}
            for name, _ in votes:
                cat = SIGNAL_CATEGORIES.get(name)
                if cat:
                    base_w = weights_dict.get(name, 1.0)
                    regime_mult = _regime_mults.get(cat, 1.0)
                    raw_cats[cat] += base_w * regime_mult

            capped_weight = 0.0; active_cats = set()
            for cat, raw_val in raw_cats.items():
                if raw_val > 0:
                    capped_weight += min(raw_val, CATEGORY_CAPS[cat])
                    active_cats.add(cat)
            return capped_weight, active_cats

        effective_weights = dict(SIGNAL_WEIGHTS)
        if signal_weights: effective_weights.update(signal_weights)

        ce_weight, ce_categories = calculate_capped_weight(ce_votes, effective_weights)
        pe_weight, pe_categories = calculate_capped_weight(pe_votes, effective_weights)

        wt = self.vix_tracker.weight_threshold()
        # v25-LIVE: REMOVED HV multiplier — double penalty with VIX regime
        # VIX HIGH already raises threshold 4.0 → 4.5. HV multiplier pushed to 5.175.
        # 12 CE votes capped at ~4.8w couldn't clear 5.175. The VIX regime IS the vol adjustment.

        # v19-LIVE: REMOVED dynamic SV threshold multiplier
        # Was multiplying wt by 1.25-1.40 when SV positive, making threshold 5.6-7.2
        # This is the #1 reason for "Insufficient votes" on live data.
        # SV quality is already handled by conviction penalties later (lines 2860-2872).
        # Keep only the pre-expansion bonus — don't RAISE the bar when SV is positive.
        # ── v28 Innovation A: Phase-aware weight threshold modifier ──────────
        # CRITICAL FIX (Aristotle Phase 2 — Examine Every Assumption):
        #   Old code applied BOTH SV bonus (wt*=0.90) AND EXPANSION bonus (wt*=0.85),
        #   stacking to 0.765× which dropped threshold from 4.5 → 3.44.  But EXPANSION's
        #   own trigger requires straddle_vel < -0.02, which ALWAYS satisfies < -0.01,
        #   so both ALWAYS co-fired during EXPANSION.  That's not two independent signals
        #   agreeing — it's one signal counted twice.
        #
        # Fix (Aristotle Phase 3 — Reconstruct):
        #   Mutually exclusive elif chain.  EXPANSION subsumes SV pre-expansion because
        #   EXPANSION IS the confirmed version of what SV was detecting.  Only ONE
        #   modifier fires per tick — no double-stacking possible.
        #
        # Priority: EXPANSION (0.85) > ACCUMULATION/DISTRIBUTION (1.08) > SV fallback (0.90)
        # Validated (Phase 4): all 5 scenario combos produce correct single modifier.
        if _market_phase == MarketPhaseTracker.EXPANSION:
            wt *= 0.85   # 15% easier — breakout confirmed by phase machine
        elif _market_phase in (MarketPhaseTracker.ACCUMULATION, MarketPhaseTracker.DISTRIBUTION):
            wt *= 1.08   # 8% harder — need stronger conviction when market is quiet
        elif straddle_vel < -0.01:
            wt *= 0.90   # 10% easier — SV pre-expansion (phase not yet confirmed)

        # v28 Innovation I: CUSUM early detection — additional soft boost
        # (Aristotle Phase 3 — Reconstruct from Atoms):
        #   When CUSUM detects premium awakening but phase machine hasn't confirmed
        #   EXPANSION yet, give a 5% additional boost. This captures the 10-15 tick
        #   window between CUSUM fire and phase confirmation.
        #   NOT a gate — purely a boost. Cannot suffocate entries.
        #   Stacks with SV fallback (0.90 × 0.95 = 0.855 ≈ EXPANSION-level ease).
        if (self.phase_tracker._cusum_fired and
                _market_phase not in (MarketPhaseTracker.EXPANSION, MarketPhaseTracker.EXHAUSTION)):
            wt *= 0.95   # 5% extra ease — CUSUM sees premium flow starting

        dominant_dir = "NEUTRAL"; dominant_votes = 0; dominant_score = 0.0
        dominant_weight = 0.0; dominant_cats = set()

        if ce_weight >= wt and ce_weight > pe_weight:
            dominant_dir = "CE"; dominant_votes = n_ce; dominant_weight = ce_weight; dominant_cats = ce_categories
        elif pe_weight >= wt and pe_weight > ce_weight:
            dominant_dir = "PE"; dominant_votes = n_pe; dominant_weight = pe_weight; dominant_cats = pe_categories

        # ── v28 Innovation A: EXHAUSTION phase hard block ─────────────────────
        # If the market phase machine detects exhaustion (large displacement + velocity
        # decaying), entering now = buying the top/bottom. No new entries allowed.
        # This is the single most important phase gate — prevents the costliest entries.
        if _market_phase == MarketPhaseTracker.EXHAUSTION and dominant_dir != "NEUTRAL":
            result = EngineResult(
                direction=dominant_dir, score=0.0, votes_for=max(n_ce, n_pe),
                vote_detail=vote_detail, entry_allowed=False,
                blocked_reason=f"Phase:EXHAUSTION — move is dying ({self.phase_tracker._move_peak_displacement:.1f}pt displacement), no new entries",
                suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts,
                smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        # v28: Relax category requirement — allow 1 category when:
        #   (a) Market is trending/breakout (strong directional regime), OR
        #   (b) Both leading OPTIONS signals (PREM_VEL + STRADDLE_GAMMA) agree on direction
        #       with score > 0.55 — institutional flow doesn't need lagging confirmation
        #   (c) v28 Innovation A: during EXPANSION phase (breakout in progress)
        _min_cats = 2
        _leading_agree = (s5.direction == dominant_dir and s5.score >= 0.55 and
                          s19.direction == dominant_dir and s19.score >= 0.55)
        if self.market_regime.is_trending or _leading_agree or _market_phase == MarketPhaseTracker.EXPANSION:
            _min_cats = 1
        if dominant_dir != "NEUTRAL" and len(dominant_cats) < _min_cats:
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=dominant_votes, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"Need {_min_cats} categories: {dominant_dir} has {dominant_weight:.1f}w from {list(dominant_cats)}.", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        if dominant_dir == "NEUTRAL":
            dominant = "CE" if ce_weight >= pe_weight else "PE"
            result = EngineResult(direction="NEUTRAL", score=0.0, votes_for=max(n_ce, n_pe), vote_detail=vote_detail, entry_allowed=False, blocked_reason=(f"Insufficient votes: CE={n_ce}({ce_weight:.1f}w) PE={n_pe}({pe_weight:.1f}w) (need {wt:.1f}w). Leading: {dominant}."), suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        dom_votes_list = ce_votes if dominant_dir == "CE" else pe_votes
        weighted_score_sum = sum(effective_weights.get(name, 1.0) * score for name, score in dom_votes_list)
        raw_dom_weight = sum(effective_weights.get(name, 1.0) for name, _ in dom_votes_list)
        weighted_conviction = round(weighted_score_sum / raw_dom_weight, 3) if raw_dom_weight > 0 else 0.0

        conv_min = self.vix_tracker.conviction_floor()
        # v28 Innovation A: EXPANSION phase lowers conviction floor by 0.04
        # During breakouts, signals may look weak individually but the move is real.
        # The phase machine already confirmed expansion (ATR expanding + BB squeeze fired).
        if _market_phase == MarketPhaseTracker.EXPANSION:
            conv_min = max(0.40, conv_min - 0.04)  # floor at 0.40 — don't go below basic sanity
        if weighted_conviction < conv_min:
            result = EngineResult(direction=dominant_dir, score=weighted_conviction, votes_for=dominant_votes, vote_detail=vote_detail, entry_allowed=False, blocked_reason=(f"Weak conviction: w_score={weighted_conviction:.2f} < {conv_min:.2f} (VIX:{self.vix_tracker.regime})"), suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        dominant_score = weighted_conviction

        vwap_dir = s1.direction; struct_dir = s8.direction
        if (vwap_dir != "NEUTRAL" and struct_dir != "NEUTRAL" and vwap_dir != struct_dir):
            dominant_score = max(0.0, dominant_score - 0.05)

        # v17.4: 15-min breakout opposition → soft penalty (was hard block)
        b15_dir = s13.direction
        if b15_dir != "NEUTRAL" and b15_dir != dominant_dir:
            dominant_score = max(0.0, dominant_score - 0.07)  # heavy penalty — this is a strong contra

        htf_trend = self.candles_5m.trend
        if htf_trend != "NEUTRAL" and htf_trend != dominant_dir:
            dominant_score = max(0.0, dominant_score - 0.04)

        # v28: Straddle velocity — graduated penalty, not hard block
        # v21 data: SV < 0 is best (pre-expansion), SV > 0 degrades.
        # But v21's hard block at 0.05 killed ALL entries on live data (650pt trending day = 0 trades).
        # Fix: soft penalty 0.05-0.30 (scaled), hard block only at extreme 0.30+.
        SV_SOFT_THRESHOLD = 0.05
        SV_HARD_THRESHOLD = 0.30    # only block truly extreme expansion chasing
        if straddle_vel < -0.01:
            dominant_score = min(dominant_score + 0.04, 0.99)  # reward: pre-expansion
        elif straddle_vel >= SV_HARD_THRESHOLD:
            result = EngineResult(
                direction=dominant_dir, score=dominant_score, votes_for=dominant_votes,
                vote_detail=vote_detail, entry_allowed=False,
                blocked_reason=f"SV Block: SV={straddle_vel:.3f} (>= {SV_HARD_THRESHOLD} = extreme expansion)",
                suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts,
                smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                vix_regime=self.vix_tracker.regime)
            return self._emit(result)
        elif straddle_vel >= SV_SOFT_THRESHOLD:
            # Graduated penalty: linear from -0.02 (SV=0.05) to -0.12 (SV=0.30)
            sv_frac = (straddle_vel - SV_SOFT_THRESHOLD) / (SV_HARD_THRESHOLD - SV_SOFT_THRESHOLD)
            sv_penalty = 0.02 + sv_frac * 0.10
            dominant_score = max(0.0, dominant_score - sv_penalty)

        # v17.4: Category opposition penalty — if opposing side has strong category
        # agreement, it means the market is conflicted. Penalize conviction.
        opp_votes = pe_votes if dominant_dir == "CE" else ce_votes
        opp_weight = pe_weight if dominant_dir == "CE" else ce_weight
        opp_cats = pe_categories if dominant_dir == "CE" else ce_categories
        # If opposition has weight > 60% of dominant AND has 2+ categories, penalize
        if opp_weight > dominant_weight * 0.60 and len(opp_cats) >= 2:
            penalty = 0.04 if opp_weight > dominant_weight * 0.75 else 0.02
            dominant_score = max(0.0, dominant_score - penalty)

        # v17.4: PREM_VEL opposition penalty — premium velocity opposing direction
        # is the strongest contra-signal (leading indicator disagreeing with lagging)
        prem_vel_dir = s5.direction
        if prem_vel_dir != "NEUTRAL" and prem_vel_dir != dominant_dir:
            # Premium velocity actively opposes — harsh penalty
            dominant_score = max(0.0, dominant_score - 0.06)

        is_fresh, transition_reason = self.transition.check(dominant_dir, dominant_score)
        if not is_fresh:
            result = EngineResult(direction=dominant_dir, score=dominant_score * 0.5, votes_for=dominant_votes, vote_detail=vote_detail, entry_allowed=False, blocked_reason=f"Stale conviction: {transition_reason}", suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=tgt_pts, smart_money_bias=smart_money, reversal_warnings=reversal_warnings, vix_regime=self.vix_tracker.regime)
            return self._emit(result)

        # v17.5: High vote count penalty — on realistic data, 9+ votes = correlated noise
        # Data: Votes=6 is 60% WR +16.2pts, Votes=9+ is worst bucket
        # Many correlated signals agreeing = they're all reading the same noise
        if dominant_votes >= 9:
            dominant_score = max(0.0, dominant_score - 0.05)
        elif dominant_votes >= 8:
            dominant_score = max(0.0, dominant_score - 0.02)

        b15_tgt_mult = 3.5 if b15_dir == dominant_dir else 3.0
        final_score = min(dominant_score + (0.04 if b15_dir == dominant_dir else 0.0), 0.99)
        if htf_trend == dominant_dir:
            final_score = min(final_score + 0.03, 0.99)
        final_tgt   = sl_pts * b15_tgt_mult

        # v19-LIVE: Track pre-penalty score to cap total penalty deductions
        # Problem: ~15 penalties can stack and crush conviction from 0.70 to 0.20
        # Even good setups get killed. Cap total penalty at 0.15 (was unlimited).
        _pre_penalty_score = final_score
        MAX_TOTAL_PENALTY = 0.15  # v19: max conviction loss from ALL soft penalties combined

        vel_ok, vel_value = self.price_velocity.velocity_confirms(dominant_dir)
        if not vel_ok:
            final_score = max(0.0, final_score - 0.05)  # v18.3-LIVE: velocity opposes, penalty not block

        # ── v18.1: DEAD STRADDLE GATE — data-driven hard block ──────────────
        # Analysis of 78 trades across 3 seeds:
        #   |SV| < 0.008:  0% of big winners, 30% of stall kills
        #   Winners ALWAYS have |SV| >= 0.01 — real moves have premium movement
        #   When straddle is dead, signals are voting on noise, not real flow.
        # This single filter blocks 6+ bad trades per seed with ZERO winner casualties.
        #
        # v28 Innovation G: VIX-Regime-Scaled Dead Straddle Threshold
        # (Aristotle Phase 2 — Examine Every Assumption):
        #   SV is a PREMIUM-derived measure. When VIX is LOW (12-14), premiums are
        #   compressed — a genuine 50pt NIFTY move produces SV = 0.005-0.007 because
        #   the absolute premium change is tiny relative to spot.
        #   Fixed 0.008 blocks ALL entries on low-vol trending days.
        #   When VIX is HIGH (24-28), noise alone produces SV = 0.010+, so 0.008 is
        #   too loose — noise passes through.
        #   Fix: scale threshold with VIX regime. LOW=0.005, MID=0.008, HIGH=0.010,
        #   EXTREME=0.012. This adapts the gate to the premium environment.
        _DEAD_SV_BY_REGIME = {"LOW": 0.005, "MID": 0.008, "HIGH": 0.010, "EXTREME": 0.012}
        DEAD_STRADDLE_THRESHOLD = _DEAD_SV_BY_REGIME.get(self.vix_tracker.regime, 0.008)
        DEAD_STRADDLE_WARMUP = 1000   # ticks before hard block activates
        abs_sv = abs(straddle_vel)
        # v28: After warmup (1000 ticks), dead straddle = hard block.
        # During early session, straddle data is thin — use soft penalty.
        if abs_sv < DEAD_STRADDLE_THRESHOLD:
            if self._tick_count >= DEAD_STRADDLE_WARMUP:
                result = EngineResult(
                    direction=dominant_dir, score=final_score, votes_for=dominant_votes,
                    vote_detail=vote_detail, entry_allowed=False,
                    blocked_reason=f"Dead straddle: |SV|={abs_sv:.4f} < {DEAD_STRADDLE_THRESHOLD} [VIX:{self.vix_tracker.regime}] (no real premium flow)",
                    suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=final_tgt,
                    smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                    vix_regime=self.vix_tracker.regime)
                return self._emit(result)
            else:
                final_score = max(0.0, final_score - 0.04)

        # ── v18.3: Kaufman Efficiency Ratio — noise vs directional ─────────
        # ER < 0.12: market is noise — signals are voting on randomness
        #   Soft penalty -0.06 (not hard block: some winners start in noise as
        #   momentum builds, and contrarian entries look like noise initially)
        # ER > 0.35 + confirms direction: directional move — boost confidence
        er_val = self.kaufman_er.er
        if self.kaufman_er.is_noise:
            final_score = max(0.0, final_score - 0.01)  # noise penalty (minimal for live — winners start in noise too)
        elif self.kaufman_er.is_directional:
            # Only boost if ER direction aligns — check if net displacement is in our direction
            if len(self.kaufman_er._prices) > self.kaufman_er.PERIOD:
                er_net = self.kaufman_er._prices[-1] - self.kaufman_er._prices[-1 - self.kaufman_er.PERIOD]
                er_dir = "CE" if er_net > 0 else "PE" if er_net < 0 else "NEUTRAL"
                if er_dir == dominant_dir:
                    final_score = min(final_score + 0.03, 0.99)  # directional confirmation boost

        # ── v18.2: CUSUM Momentum Burst — confirm-only boost (no penalty) ──
        # CUSUM confirms direction but NEVER penalizes — best entries are
        # often contrarian (CE on dip). Only reward entries WITH momentum.
        burst_dir = self.momentum_burst.direction
        burst_str = self.momentum_burst.strength
        if burst_dir == dominant_dir and burst_str > 0.3:
            final_score = min(final_score + 0.04, 0.99)  # momentum confirms — boost

        # v18: Premium Divergence — the strongest direction confirmation
        # If CE is rising while PE falls, that's real CE flow (not noise)
        # If premiums diverge AGAINST our direction, block the entry
        div_confirms, div_strength, div_reason = self.prem_divergence.confirms_direction(dominant_dir)
        # v18.3-LIVE: converted from hard block to soft penalty
        # Hard block killed ALL entries on live data — premium layers rarely agree perfectly
        if not div_confirms:
            final_score = max(0.0, final_score - 0.04)  # oppose = penalty, not block
        elif div_strength > 0.3:
            final_score = min(final_score + 0.03, 0.99)  # strong confirmation boost

        candles = self.candles.closed_candles
        if len(candles) >= 5:
            recent_5 = candles[-5:]
            recent_high = max(c.high for c in recent_5)
            recent_low  = min(c.low  for c in recent_5)
            recent_range = recent_high - recent_low
            current_price = fut

            if recent_range > 0:
                position = (current_price - recent_low) / recent_range
                is_hv = self.atr_calc.is_high_volatility()
                max_ce_pos = 0.55 if is_hv else 0.80
                min_pe_pos = 0.45 if is_hv else 0.20

                if dominant_dir == "CE" and position > max_ce_pos:
                    final_score = max(0.0, final_score - 0.05)  # v18.3-LIVE: chasing penalty, not block
                elif dominant_dir == "PE" and position < min_pe_pos:
                    final_score = max(0.0, final_score - 0.05)  # v18.3-LIVE: chasing penalty, not block

        if self.market_regime.is_range:
            if not self.straddle_gamma.is_expanding:
                final_score = max(0.0, final_score - RANGE_CONVICTION_PENALTY - 0.06)
            else:
                final_score = max(0.0, final_score - RANGE_CONVICTION_PENALTY)

        if self.market_regime.is_choppy:
            if not self.straddle_gamma.is_expanding:
                final_score = max(0.0, final_score - 0.15)
            else:
                final_score = max(0.0, final_score - 0.08)

        if self.market_regime.is_low_vol:
            final_score = max(0.0, final_score - 0.18)

        if self.market_regime.is_trending:
            if self.pullback.in_pullback_zone and self.pullback.pullback_direction == dominant_dir:
                final_score = min(final_score + 0.05, 0.99)

        # v19-LIVE: PENALTY CAP — prevent conviction death by a thousand cuts
        # Without this, 15 soft penalties stack and crush 0.70 → 0.35 (unenterable)
        total_penalty = _pre_penalty_score - final_score
        if total_penalty > MAX_TOTAL_PENALTY:
            final_score = _pre_penalty_score - MAX_TOTAL_PENALTY

        # ── Real Win Probability Calculation (first-passage-time) ──────────
        atr_val = self.atr_calc.atr if not math.isnan(self.atr_calc.atr) else 5.0

        # VWAP z-score: how far price is from VWAP as fraction
        vwap_z = 0.0
        if not math.isnan(self.vwap_signal.vwap) and self.vwap_signal.vwap > 0:
            vwap_z = (fut - self.vwap_signal.vwap) / self.vwap_signal.vwap * 100  # in basis-point-like units

        # Momentum: EMA separation as proxy
        mom_strength = 0.0
        if not math.isnan(self.momentum._ema9) and not math.isnan(self.momentum._ema21) and self.momentum._ema21 > 0:
            mom_strength = (self.momentum._ema9 - self.momentum._ema21) / self.momentum._ema21 * 1000

        # Premium velocity alignment: does option premium confirm direction?
        prem_aligned = (s5.direction == dominant_dir and s5.score > 0.5)

        # ATR ratio: current ATR vs session median (expanding = momentum environment)
        atr_ratio = 1.0
        if len(self.atr_calc._atr_history) >= 5:
            median_atr = float(np.median(list(self.atr_calc._atr_history)))
            if median_atr > 0:
                atr_ratio = atr_val / median_atr

        self.win_prob_calc.update_context(vwap_z, mom_strength, prem_aligned, atr_ratio)
        prob = self.win_prob_calc.calculate(dominant_dir, sl_pts, final_tgt, atr_val)
        ev = self.win_prob_calc.expected_value(prob, sl_pts, final_tgt)

        # ── P(win) gate: block entry if probability too low ──────────────
        if prob < WIN_PROB_MIN_ENTRY:
            result = EngineResult(
                direction=dominant_dir, score=final_score, votes_for=dominant_votes,
                vote_detail=vote_detail, entry_allowed=False,
                blocked_reason=f"Low P(win): {prob:.0%} < {WIN_PROB_MIN_ENTRY:.0%} (EV={ev:+.1f}pts)",
                suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=final_tgt,
                smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                vix_regime=self.vix_tracker.regime, probability=prob,
            )
            return self._emit(result)

        # ── Immediate Momentum gate: is price moving in our direction NOW? ─
        # Two-sided filter:
        #   Floor (-0.10): blocks when clearly opposed (obvious losers)
        #   Ceiling (0.75): blocks when momentum is exhausted (chasing entries)
        # v26: IMS scale = VIX regime × premium level. A 220pt option moves 2.7× more
        # in absolute pts than an 80pt option. Without this, IMS is permanently
        # inflated on high-premium options, blocking entries and crushing conviction.
        _prem_level = (ce_ltp + pe_ltp) / 200.0 if (ce_ltp > 0 and pe_ltp > 0) else 1.0
        _vix_ims_scale = self.vix_tracker.sl_multiplier() * max(_prem_level, 0.5)
        ims = self.win_prob_calc.immediate_momentum(dominant_dir, vix_scale=_vix_ims_scale)
        if ims < IMM_MOMENTUM_MIN:
            result = EngineResult(
                direction=dominant_dir, score=final_score, votes_for=dominant_votes,
                vote_detail=vote_detail, entry_allowed=False,
                blocked_reason=f"Momentum opposed: IMS={ims:+.2f} (price moving against {dominant_dir})",
                suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=final_tgt,
                smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                vix_regime=self.vix_tracker.regime, probability=prob,
            )
            return self._emit(result)

        # v18.3: Two-tier IMS penalty — data shows IMS 0.12+ is garbage zone
        # IMS 0.12-0.20: mild penalty (some winners still possible)
        # IMS 0.20+: heavy penalty (nearly zero edge — stall kill territory)
        if IMM_MOMENTUM_HEAVY_ZONE <= ims <= IMM_MOMENTUM_MAX:
            final_score = max(0.0, final_score - 0.05)  # heavy penalty (reduced for live) — stall kill zone
        elif IMM_MOMENTUM_PENALTY_ZONE <= ims < IMM_MOMENTUM_HEAVY_ZONE:
            final_score = max(0.0, final_score - 0.03)  # mild penalty (reduced for live)

        if ims > IMM_MOMENTUM_MAX:
            result = EngineResult(
                direction=dominant_dir, score=final_score, votes_for=dominant_votes,
                vote_detail=vote_detail, entry_allowed=False,
                blocked_reason=f"Chasing momentum: IMS={ims:+.2f} > {IMM_MOMENTUM_MAX} (move already happened)",
                suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=final_tgt,
                smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                vix_regime=self.vix_tracker.regime, probability=prob,
            )
            return self._emit(result)

        entry_thesis = {
            "direction": dominant_dir,
            "vwap": self.vwap_signal.vwap if not math.isnan(self.vwap_signal.vwap) else 0.0,
            "supertrend": self.supertrend.direction,
            "structure": s8.direction,
            "htf_trend": htf_trend,
            "probability": prob,
            "expected_value": ev,
            "imm_momentum": ims,
            "vix_regime": self.vix_tracker.regime,
            "fut_vel_disp": round(self.fut_vel.displacement, 3),
            "drift_factors": {
                "vwap_z": round(vwap_z, 2),
                "momentum": round(mom_strength, 2),
                "prem_aligned": prem_aligned,
                "atr_ratio": round(atr_ratio, 2),
                "kaufman_er": round(er_val, 3),
            },
        }

        # v29: Update exhaustion gate with the now-known dominant direction
        # (first pass used NEUTRAL; now re-call with actual direction so state machine
        # knows which way the engine is voting this tick — needed for flip detection)
        self.exhaustion_gate.update(
            engine_direction  = dominant_dir,
            candle_direction  = candle_dir,
            rsi_bearish_div   = self.rsi_divergence.bearish_divergence,
            rsi_bullish_div   = self.rsi_divergence.bullish_divergence,
            prem_vel_score    = _pv_score,
            current_atr       = self.atr_calc.atr if self.atr_calc.atr > 0 else 5.0,
            candle_closed     = False,  # candle already processed above; don't double-count
        )
        # v29: Block direction-flip entries when trend is exhausted but reversal unconfirmed.
        # This prevents the "2 SL then correct trade" churn pattern.
        # The gate only fires when: (a) trend has been running >= 3 candles, (b) >= 2 exhaustion
        # signals are active, and (c) this entry is OPPOSITE to the established run direction.
        # It does NOT block continuation entries or NEUTRAL results.
        if self.exhaustion_gate.is_flip_blocked(dominant_dir):
            exhaust_sum = self.exhaustion_gate.exhaustion_summary
            result = EngineResult(
                direction=dominant_dir, score=final_score * 0.4, votes_for=dominant_votes,
                vote_detail=vote_detail, entry_allowed=False,
                blocked_reason=f"ExhaustionGate: {self.exhaustion_gate.block_reason} {exhaust_sum}",
                suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=final_tgt,
                smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
                vix_regime=self.vix_tracker.regime, market_regime=self.market_regime.regime,
                entry_thesis=entry_thesis, probability=prob, imm_momentum=ims,
            )
            return self._emit(result)

        result = EngineResult(
            direction=dominant_dir, score=final_score, votes_for=dominant_votes,
            vote_detail=vote_detail, entry_allowed=True, blocked_reason="",
            suggested_sl_pts=sl_pts, suggested_sl_pct=sl_pct, suggested_tgt_pts=final_tgt,
            smart_money_bias=smart_money, reversal_warnings=reversal_warnings,
            vix_regime=self.vix_tracker.regime, market_regime=self.market_regime.regime,
            entry_thesis=entry_thesis, probability=prob, imm_momentum=ims,
        )
        return self._emit(result)

    @property
    def last_result(self) -> Optional[EngineResult]: return self._last_result

    def reset_after_exit(self, exit_reason: str = "SL HIT", direction: str = "NEUTRAL"):
        # v17.3: Always mark exit direction for base-pattern requirement
        # This replaces arbitrary win cooldown — engine must see signal weaken
        # and strengthen again before re-entering same direction
        if exit_reason == "TRAIL SL":
            self.transition.soft_reset()
        else:
            self.transition.reset()
        self.transition.mark_exit(direction)
        self.thesis_tracker.reset()

    def soft_reset_after_trail(self): self.transition.soft_reset()

    def get_gate_diagnostics(self) -> dict:
        """v25-LIVE: Compute ALL gate values from current engine state.
        Called AFTER evaluate() — reads internal state to build a full diagnostic
        so the dashboard can show every gate at once, not just the first blocker."""
        r = self._last_result
        if r is None:
            return {}

        signals = [s for s in self._last_signals if s.direction != "NEUTRAL"] if hasattr(self, '_last_signals') else []

        # Vote weights (same logic as evaluate)
        SIGNAL_CATEGORIES = {
            "VWAP": "TREND", "VWAP_RETEST": "TREND", "STRUCTURE": "TREND",
            "AVWAP": "TREND", "SUPERTREND": "TREND", "BREAKOUT15": "TREND", "ORB": "TREND",
            "RSI": "MOMENTUM", "MOMENTUM": "MOMENTUM", "MACD_HIST": "MOMENTUM",
            "BB_SQUEEZE": "MOMENTUM", "ADX": "MOMENTUM",
            "OPT_OI": "OPTIONS", "PREM_VEL": "OPTIONS", "MAX_PAIN": "OPTIONS",
            "MKT_OI": "OPTIONS", "COI_PCR": "OPTIONS", "PCP": "OPTIONS",
            "STRADDLE_GAMMA": "OPTIONS",
            "FUT_VEL": "ENERGY",
        }
        CATEGORY_CAPS = {"MOMENTUM": 1.0, "TREND": 2.5, "OPTIONS": 5.0, "ENERGY": 0.8}
        ew = dict(SIGNAL_WEIGHTS)

        # v28: Apply same regime multipliers as evaluate() for consistent diagnostics
        REGIME_MULTIPLIERS = {
            "TRENDING":  {"TREND": 1.3, "OPTIONS": 1.1, "MOMENTUM": 0.7, "ENERGY": 1.2},
            "BREAKOUT":  {"TREND": 1.1, "OPTIONS": 1.3, "MOMENTUM": 0.8, "ENERGY": 1.1},
            "RANGE":     {"TREND": 0.8, "OPTIONS": 1.0, "MOMENTUM": 0.8, "ENERGY": 0.9},
            "CHOPPY":    {"TREND": 0.7, "OPTIONS": 0.9, "MOMENTUM": 0.5, "ENERGY": 0.7},
            "LOW_VOL":   {"TREND": 0.7, "OPTIONS": 0.7, "MOMENTUM": 0.5, "ENERGY": 0.5},
        }
        _diag_regime_mults = REGIME_MULTIPLIERS.get(self.market_regime.regime, {})

        def capped_w(votes):
            cats = {"MOMENTUM": 0.0, "TREND": 0.0, "OPTIONS": 0.0, "ENERGY": 0.0}
            for name, _ in votes:
                cat = SIGNAL_CATEGORIES.get(name)
                if cat:
                    base_w = ew.get(name, 1.0)
                    regime_mult = _diag_regime_mults.get(cat, 1.0)
                    cats[cat] += base_w * regime_mult
            w = 0.0; ac = set()
            for cat, rv in cats.items():
                if rv > 0: w += min(rv, CATEGORY_CAPS[cat]); ac.add(cat)
            return w, ac

        ce_v = [(s.name, s.score) for s in signals if s.direction == "CE"]
        pe_v = [(s.name, s.score) for s in signals if s.direction == "PE"]
        ce_w, ce_cats = capped_w(ce_v)
        pe_w, pe_cats = capped_w(pe_v)

        wt = self.vix_tracker.weight_threshold()
        # v25-LIVE: REMOVED HV multiplier (same as evaluate — no double penalty)
        sv = self.straddle_gamma.straddle_velocity
        # v28 FIX: Must match evaluate()'s mutually exclusive elif chain.
        # Old code only applied SV bonus, ignoring phase. Dashboard would show
        # wt_needed=3.6 during ACCUMULATION when engine actually requires 4.86.
        # Trader couldn't trust the dashboard to understand blocked entries.
        _diag_phase = self.phase_tracker.phase
        if _diag_phase == MarketPhaseTracker.EXPANSION:
            wt_eff = wt * 0.85   # phase-confirmed breakout
        elif _diag_phase in (MarketPhaseTracker.ACCUMULATION, MarketPhaseTracker.DISTRIBUTION):
            wt_eff = wt * 1.08   # quiet/choppy — higher bar
        elif sv < -0.01:
            wt_eff = wt * 0.90   # SV pre-expansion fallback
        else:
            wt_eff = wt
        # v28 Innovation I: CUSUM boost (must match evaluate)
        if (self.phase_tracker._cusum_fired and
                _diag_phase not in (MarketPhaseTracker.EXPANSION, MarketPhaseTracker.EXHAUSTION)):
            wt_eff *= 0.95

        # IMS — use same scale as the main evaluation path (vix × premium level)
        leading = "CE" if ce_w >= pe_w else "PE"
        if hasattr(self, 'win_prob_calc'):
            _diag_ce = self.win_prob_calc._ce_history[-1] if self.win_prob_calc._ce_history else 0.0
            _diag_pe = self.win_prob_calc._pe_history[-1] if self.win_prob_calc._pe_history else 0.0
            _diag_prem = (_diag_ce + _diag_pe) / 200.0 if (_diag_ce > 0 and _diag_pe > 0) else 1.0
            _diag_scale = self.vix_tracker.sl_multiplier() * max(_diag_prem, 0.5)
            ims = self.win_prob_calc.immediate_momentum(leading, vix_scale=_diag_scale)
        else:
            ims = 0.0

        # Transition freshness
        is_fresh, stale_reason = self.transition.check(leading, 0.5)

        # v28 Innovation A: Include phase state machine diagnostics
        _phase = self.phase_tracker.phase
        _phase_policy = self.phase_tracker.entry_policy
        _phase_disp = round(self.phase_tracker._move_peak_displacement, 1)

        # v28 Innovation I: CUSUM early detection state
        _cusum_fired = self.phase_tracker._cusum_fired
        _cusum_level = round(max(self.phase_tracker._cusum_pos, self.phase_tracker._cusum_neg), 4)

        # v28 Innovation J: COI Strength % from OI deltas
        _coi_strength = round(getattr(self, '_coi_strength_pct', 0.0), 1)

        # v28 Innovation K: IV Skew Proxy
        _skew_ratio = round(getattr(self, '_iv_skew_ratio', 1.0), 2)

        return {
            "ce_votes": len(ce_v), "ce_weight": round(ce_w, 2), "ce_cats": sorted(ce_cats),
            "pe_votes": len(pe_v), "pe_weight": round(pe_w, 2), "pe_cats": sorted(pe_cats),
            "wt_needed": round(wt_eff, 2), "wt_raw": round(wt, 2),
            "sv": round(sv, 4), "sv_soft": 0.05, "sv_hard": 0.30,
            "ims": round(ims, 3), "ims_min": -0.05, "ims_max": 0.75,
            "conv_floor": round(self.vix_tracker.conviction_floor(), 3),
            "is_fresh": is_fresh, "stale_reason": stale_reason if not is_fresh else "",
            "regime": self.market_regime.regime,
            "vix": round(self.vix_tracker.vix, 1),
            "vix_regime": self.vix_tracker.regime,
            # v28 Innovation A: Phase state machine
            "phase": _phase,
            "phase_policy": _phase_policy,
            "phase_displacement": _phase_disp,
            # v28.4 Innovation I: CUSUM early detection (fixed two-sided)
            "cusum_fired": _cusum_fired,
            "cusum_level": _cusum_level,
            "cusum_direction": self.phase_tracker._cusum_direction,
            # v28 Innovation J: COI Strength
            "coi_strength": _coi_strength,
            # v28 Innovation K: IV Skew
            "skew_ratio": _skew_ratio,
        }

    def summary(self) -> str:
        r = self._last_result
        if r is None: return "NiftyEngine: no data"
        hv  = " HV!" if self.atr_calc.is_high_volatility() else ""
        vix = f" VIX:{self.vix_tracker.regime}" if self.vix_tracker.regime != "MID" else ""
        if r.entry_allowed: return (f"Engine:{r.direction} {r.score:.0%} ({r.votes_for}/18) P(win):{r.probability:.0%} IMS:{r.imm_momentum:+.2f}{hv}{vix}")
        return (f"Engine:BLOCK {r.votes_for}/18 SM:{r.smart_money_bias} [{r.blocked_reason}]{hv}{vix}")

if __name__ == "__main__":
    print("hawk_engine.py v28.4 — FIX: two-sided CUSUM + snapshot alignment")
#!/usr/bin/env python3
"""
HAWK TRADER v18 — All-In, No Partials, Premium Divergence

  ENTRY:
  - CONVICTION: 0.64-0.80 (cap blocks chasing) | VOTES_MIN: 6 | Categories: 2 of 3
  - IMS gate: -0.05 to +0.40 (tightened ceiling — high IMS = chasing)
  - SIGNAL_PERSISTENCE: 3 out of 5 ticks (sliding window)
  - Premium Divergence Filter: blocks wrong-direction entries via CE/PE velocity divergence
  EXIT:
  - SL: 11pts | STALL KILL: exit dead trades (<1pt peak) at market after 120s
  - PROGRESSIVE CHANDELIER: 45-75% lock of peak profit
  - BREAKEVEN: SL -> entry+1.0 at 10pts profit
  - BLIND HOLD: 30s (escapes permanently at 8+ pts profit; hard SL still active)
  - ALL-IN: No partial booking. Capital compounds on every trade.
"""
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os, sys, time, threading, logging, logging.handlers
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from collections import deque

import multiprocessing as mp

from breeze_connect import BreezeConnect

try:
    import login as _login
except ImportError:
    print("FATAL: login.py not found"); sys.exit(1)

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich.align import Align
    from rich import box
    RICH = True
except ImportError:
    RICH = False

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Sound setup ──────────────────────────────────────────────────────────────
# v29: Two bugs fixed:
#   BUG 1 — Wrong filenames. Code looked for entry.mp3/win.mp3/loss.mp3 but the
#            actual files in the folder are alert_entry.mp3/alert_win.mp3/alert_loss.mp3.
#            None matched → always fell through to generated tones.
#   BUG 2 — pygame.mixer.Sound() cannot load MP3 on most platforms (Windows/Linux).
#            Sound() only supports WAV and OGG natively. MP3 requires pygame.mixer.music.
#            Even if filenames had matched, Sound(mp3) would throw an exception and
#            silently fall back to generated tones via the except block.
#   FIX   — Correct filenames + new _MusicPlayer class that uses pygame.mixer.music
#            for MP3 files and pygame.mixer.Sound for WAV/OGG. MP3s are played via
#            music.load()/music.play() in a non-blocking thread so they don't stall
#            the trading loop. WAV/OGG files still use Sound() for true concurrency.
import struct as _struct, math as _math, threading as _threading
_IS_MAIN_PROC = (mp.current_process().name == "MainProcess")
HAS_SOUND     = False

# v29: Corrected filenames — must match what is actually on disk
_MP3_ENTRY = os.path.join(_DIR, "alert_entry.mp3")
_MP3_WIN   = os.path.join(_DIR, "alert_win.mp3")
_MP3_LOSS  = os.path.join(_DIR, "alert_loss.mp3")

# Sound containers — either a pygame.mixer.Sound object (WAV/OGG)
# or a file path string (MP3, played via music channel)
_SND_ENTRY = None   # Sound object or mp3 path
_SND_WIN   = None
_SND_LOSS  = None
_SND_IS_MP3: dict = {"entry": False, "win": False, "loss": False}

# MP3 playback lock — pygame.mixer.music is a single sequential channel;
# serialise access so simultaneous alert calls don't corrupt each other
_MUSIC_LOCK = _threading.Lock()

def _generate_tone(freq: int, duration_ms: int, volume: float = 0.85) -> bytes:
    sr  = 22050
    n   = int(sr * duration_ms / 1000)
    buf = bytearray(n * 2)
    for i in range(n):
        fade = 1.0 - (i / n) ** 0.3
        val  = int(32767 * volume * fade * _math.sin(2 * _math.pi * freq * i / sr))
        _struct.pack_into("<h", buf, i * 2, max(-32767, min(32767, val)))
    return bytes(buf)

def _load_sound(path: str, fallback_freq: int, fallback_ms: int, volume: float):
    """
    Load a sound file. Returns (handle, source_label, is_mp3).
    For MP3: returns (path_string, label, True) — played via music channel.
    For WAV/OGG: returns (Sound_object, label, False) — played via Sound channel.
    Falls back to generated tone if file not found.
    """
    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mp3":
            # MP3 must go through pygame.mixer.music — do NOT use Sound()
            return path, os.path.basename(path), True
        else:
            # WAV / OGG — Sound() handles these natively
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(volume)
                return snd, os.path.basename(path), False
            except Exception as _ex:
                print(f"  [WARN] Sound: failed to load {path}: {_ex} — using tone")
    # Fallback: synthesised tone
    snd = pygame.mixer.Sound(buffer=_generate_tone(fallback_freq, fallback_ms))
    snd.set_volume(volume)
    return snd, f"tone({fallback_freq}Hz)", False

if _IS_MAIN_PROC:
    try:
        import pygame
        pygame.mixer.pre_init(22050, -16, 1, 2048)
        pygame.mixer.init()
        if pygame.mixer.get_init():
            has_any_mp3 = any(os.path.exists(p) for p in [_MP3_ENTRY, _MP3_WIN, _MP3_LOSS])
            if has_any_mp3:
                _SND_ENTRY, _src_e, _SND_IS_MP3["entry"] = _load_sound(_MP3_ENTRY, 1000, 600, 1.0)
                _SND_WIN,   _src_w, _SND_IS_MP3["win"]   = _load_sound(_MP3_WIN,   880,  500, 0.85)
                _SND_LOSS,  _src_l, _SND_IS_MP3["loss"]  = _load_sound(_MP3_LOSS,  330,  700, 0.95)
                print(f"  Sound: OK — entry={_src_e}  win={_src_w}  loss={_src_l}")
            else:
                # No mp3 files found — synthesise distinct tones
                _SND_ENTRY, _, _ = _load_sound("", 1000, 600, 0.90)
                _SND_WIN,   _, _ = _load_sound("",  880, 500, 0.85)
                _SND_LOSS,  _, _ = _load_sound("",  330, 700, 0.95)
                print(f"  Sound: OK (generated tones — place alert_entry.mp3 / alert_win.mp3 / alert_loss.mp3 in {_DIR})")
            HAS_SOUND = True
        else:
            print("  [WARN] Sound: pygame mixer failed to initialise — no audio")
    except Exception as _e:
        print(f"  [WARN] Sound init failed: {_e}")

def play_sound(kind: str = "entry"):
    """
    Play a named alert sound. kind ∈ {"entry", "win", "loss"}.

    v29: Dual-path playback:
      - MP3 files → pygame.mixer.music (the only pygame API that supports MP3).
        Played in a daemon thread so the trading loop is never blocked.
        _MUSIC_LOCK serialises concurrent calls since music is a single channel.
      - WAV/OGG files → pygame.mixer.Sound.play() (multi-channel, non-blocking).
      - Generated tones → pygame.mixer.Sound.play() (same as WAV path).

    If the mixer has died (underrun / device lost), it is silently re-initialised
    before playback so the system self-heals without operator intervention.
    """
    if not HAS_SOUND:
        return
    try:
        snd  = {"entry": _SND_ENTRY, "win": _SND_WIN, "loss": _SND_LOSS}.get(kind)
        if snd is None:
            return
        # Self-heal: re-init mixer if it has crashed (common on long-running processes)
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(22050, -16, 1, 2048)
            except Exception:
                return

        if _SND_IS_MP3.get(kind, False):
            # ── MP3 path: must use pygame.mixer.music ──────────────────────
            # music.load()+play() runs on the music channel (one at a time).
            # We fire it in a daemon thread so it never stalls the trading loop.
            def _play_mp3(path: str):
                with _MUSIC_LOCK:
                    try:
                        pygame.mixer.music.load(path)
                        pygame.mixer.music.set_volume(0.95)
                        pygame.mixer.music.play()
                    except Exception as _ex:
                        log.debug("play_sound MP3(%s) failed: %s", kind, _ex)
            _threading.Thread(target=_play_mp3, args=(snd,), daemon=True).start()
        else:
            # ── WAV/OGG/tone path: Sound.play() is non-blocking ────────────
            snd.play()

    except Exception as _e:
        log.debug("play_sound(%s) failed: %s", kind, _e)

from hawk_feed import (
    PriceStore, FeedManager, build_snapshot,
    _NEW_TICK_EVENT,
)

IST = timezone(timedelta(hours=5, minutes=30))
_IS_MAIN = (mp.current_process().name == "MainProcess")

# =============================================================================
# CONFIG
# =============================================================================
class CFG:
    STOCK_CODE        = "NIFTY"
    SPOT_FRAGMENT     = "NIFTY 50"
    EXCHANGE          = "NFO"
    STRIKE_STEP       = 50
    LOT_SIZE          = 25
    NUM_STRIKES       = 8
    EXPIRY_DATE       = "2026-05-05"
    FUTURES_EXPIRY    = "2026-05-26"

    @staticmethod
    def ws_expiry() -> str:
        d = datetime.strptime(CFG.EXPIRY_DATE, "%Y-%m-%d")
        return d.strftime("%d-%b-%Y")

    @staticmethod
    def futures_ws_expiry() -> str:
        d = datetime.strptime(CFG.FUTURES_EXPIRY, "%Y-%m-%d")
        return d.strftime("%d-%b-%Y")

    @staticmethod
    def futures_rest_expiry() -> str:
        d = datetime.strptime(CFG.FUTURES_EXPIRY, "%Y-%m-%d")
        return d.strftime("%Y-%m-%dT06:00:00.000Z")

    @staticmethod
    def rest_expiry() -> str:
        d = datetime.strptime(CFG.EXPIRY_DATE, "%Y-%m-%d")
        return d.strftime("%Y-%m-%dT06:00:00.000Z")

    STARTING_CAPITAL  = 100_000.0
    MAX_OPEN          = 1
    MIN_PREMIUM       = 40.0             # v17.2: widened from 50 — realistic data calibration
    MAX_PREMIUM       = 350.0            # v17.2: widened from 280 — capture higher-IV entries
    CONVICTION_MIN    = 0.62             # v29: raised 0.55→0.62 — 0.55 produced 34% WR (coin-flip); empirical break-even needs 43%+

    SL_MIN            = 11.0           # v17.3: sweep winner — 11pts room for realistic vol
    SIGNAL_PERSISTENCE_TICKS = 2       # v23: loosened from 3 — impulse confirmation replaces persistence
    SIGNAL_PERSIST_WINDOW    = 5       # v19.4: match backtest — need 2 out of 5 consistency

    VOTES_MIN         = 6              # v29: raised 4→6 — 4 was producing 34% WR; minimum credible signal cluster
    HIGH_QUALITY_VOTES = 10
    HIGH_QUALITY_CONV  = 0.75

    MARKET_OPEN_H     = 9
    MARKET_OPEN_M     = 15
    MARKET_CLOSE_H    = 15
    MARKET_CLOSE_M    = 30
    AVOID_FIRST_MINS  = 5
    AVOID_LAST_MINS   = 15
    NO_ENTRY_AFTER_H  = 15
    NO_ENTRY_AFTER_M  = 15
    TRADE_COOLDOWN    = 0              # v30: removed — smart conviction scaling is the gate, not arbitrary time locks
    SL_COOLDOWN       = 0              # v30: removed — consecutive SL conviction ramp handles post-loss quality
    MAX_SPREAD        = 2.0

    SNAP_INTERVAL     = 1.0

    # v17.3: Stall detection — kill dead trades or tighten SL
    ADAPTIVE_SL_SECONDS  = 180.0       # v29: raised 120→180s — 120s was killing trades that needed 2-3min to build momentum
    ADAPTIVE_SL_MIN_MOVE = 3.0         # unchanged — 3pts move needed before patience expires
    SL_TIGHT             = 9.0         # unchanged
    STALL_KILL           = True        # keep enabled
    STALL_KILL_THRESHOLD = 2.5         # v29: raised 1.0→2.5 — on 200pt+ premiums, 1pt threshold kills normal noise; 91% of STALL KILLs were losses

    # v28 Innovation H: Theta-Decay Adaptive Stall Patience
    # (Aristotle Phase 2 — Examine Every Assumption):
    #   ATM weekly option theta at 10:00 IST ≈ 0.3-0.5 pt/min (slow bleed).
    #   ATM weekly option theta at 14:30 IST ≈ 1.0-1.5 pt/min (3-4x faster).
    #   A stalling trade (peak < 1pt) is NOT making money — it's only bleeding theta.
    #   Waiting 120s at 14:30 bleeds 2-3pts; at 10:00 same wait bleeds 0.6-1pt.
    #   Fix: reduce patience in afternoon when theta cost of waiting is higher.
    #     Before 13:00 → 120s (standard morning patience)
    #     13:00-14:00  → 100s (theta accelerating, moderate urgency)
    #     After 14:00  → 75s  (theta 3-4x, aggressive exit of stalls)
    STALL_PATIENCE_AFTERNOON = 100.0   # 13:00-14:00 IST patience (seconds)
    STALL_PATIENCE_LATE      = 75.0    # after 14:00 IST patience (seconds)

    # v27: TIERED CHANDELIER — raise activation threshold, give small peaks room to breathe.
    # Root cause of "chillad chori": CHANDELIER_MIN_PROFIT=6 was too low. A 6-7pt peak
    # that arrives in 60-90 seconds is noise, not a trend. Firing the trail there locked
    # in 3-4pts and exited — trade had no room to develop into a 15-20pt runner.
    # Fix: don't trail until peak >= 8. Above 8, use tiered pct offset that is LOOSE
    # on small peaks (more room) and tightens only on genuine runners (>15pts).
    #
    #   Peak  8-12: offset = max(3.5, peak * 0.28)  → lock 60-72%  (breath room)
    #   Peak 12-20: offset = max(3.5, peak * 0.22)  → lock 72-80%  (protecting profit)
    #   Peak   >20: offset = max(4.0, peak * 0.18)  → lock 80-82%  (big runner, tighter)
    CHANDELIER_MIN_PROFIT = 8.0        # v27: raised 6→8 — stops trailing trivial spikes
    TRAIL_MIN_OFFSET      = 3.5        # v27: raised 2.5→3.5 — absolute floor; more breathing room
    TRAIL_PCT_OFFSET      = 0.28       # v27: 28% used for 8-12pt peak tier (see _check_exit)

    # v28 Innovation F: PREMIUM-SCALED CHANDELIER FLOOR
    # (Aristotle Phase 2 — Examine Every Assumption):
    #   The fixed TRAIL_MIN_OFFSET=3.5 assumes all options have the same pullback noise.
    #   WRONG: a 200pt option swings 5-8pts during normal trend continuation; 3.5 kills it.
    #   A 80pt option swings 1-3pts; 3.5 is fine.
    #   Fix: floor = max(TRAIL_MIN_OFFSET, entry_price * TRAIL_FLOOR_PCT).
    #   This makes the floor scale with option premium:
    #     80pt → max(3.5, 2.0) = 3.5 (unchanged — cheap options keep tight floor)
    #    150pt → max(3.5, 3.75) = 3.75 (marginal)
    #    200pt → max(3.5, 5.0) = 5.0 (survives normal 5pt noise during trends)
    #    300pt → max(3.5, 7.5) = 7.5 (deep ITM options need room)
    TRAIL_FLOOR_PCT       = 0.025      # 2.5% of entry premium = dynamic chandelier floor

    # v28: SOFT TRAIL — bridges the 4-7pt dead zone where trades had ZERO protection.
    # (Aristotle Phase 2): A trade peaking at 7pts and reverting to SL loses 18pts total
    # (7 peak + 11 SL). That's the costliest failure mode — winner becomes max loser.
    # This graduated trail reduces LOSS (not locks profit) as peak grows toward chandelier.
    # Cannot cause chillad chori — at peak 7 it only reaches breakeven, never locks profit.
    #   Peak 4: SL = entry - 6  (was -11, saves 5pts of max loss)
    #   Peak 5: SL = entry - 4  (saves 7pts)
    #   Peak 6: SL = entry - 2  (saves 9pts)
    #   Peak 7: SL = entry - 0  (breakeven, saves 11pts)
    #   Peak 8+: Chandelier takes over (entry + 4.5 min)
    SOFT_TRAIL_TRIGGER    = 4.0        # peak where soft trail begins
    SOFT_TRAIL_SL_START   = 6.0        # SL offset at trigger (entry - 6)
    SOFT_TRAIL_RATE       = 2.0        # SL improves by 2pts per 1pt of additional peak

    # v17.2: Breakeven move — once up by trigger, move SL to entry
    BREAKEVEN_TRIGGER     = 10.0
    BREAKEVEN_BUFFER      = 1.0

    # v24: THREE-PHASE RETEST — correct model
    # Phase 1: first move proves buying interest (P0 → high).
    # Phase 2: price returns NEAR P0 (signal price) — the retest.
    # Phase 3: fresh engine signal + price up + volume increasing → confirmed entry.
    RETEST_FIRST_MOVE     = {'LOW':3.0, 'MID':3.0, 'HIGH':4.0, 'EXTREME':5.0}
    # v28: Dynamic retest buffer — percentage of premium, clamped.
    # Old: fixed 1.0pt for all premiums. A 200pt option returning within 1pt is impossible.
    # New: 5-8% of premium depending on VIX, clamped [2.0, 8.0].
    RETEST_RETURN_PCT     = {'LOW': 0.05, 'MID': 0.06, 'HIGH': 0.07, 'EXTREME': 0.08}
    RETEST_RETURN_MIN     = 2.0       # minimum buffer in points
    RETEST_RETURN_MAX     = 8.0       # maximum buffer in points
    # v25-LIVE: VIX-scaled support — VIX 26 swings 2x wider than VIX 13
    # Bug fix: tighten-floor was killing valid setups (11 cancellations in one session)
    # HIGH/EXTREME VIX options oscillate 4-6 pts normally; old values of 3/4 were
    # too tight. Widened to match actual intraday option noise at VIX 25+.
    # v26-LIVE: Support floor is now PERCENTAGE of current premium, not fixed points.
    # Fixed points fail when premium changes (80pt yesterday vs 220pt today).
    # At 80pt: HIGH=3.2pt. At 220pt: HIGH=8.8pt. Scales with option noise naturally.
    RETEST_SUPPORT_BUF_PCT = {'LOW': 0.015, 'MID': 0.020, 'HIGH': 0.040, 'EXTREME': 0.055}
    RETEST_SUPPORT_BUF_MIN = {'LOW': 1.0,   'MID': 1.5,   'HIGH': 3.0,   'EXTREME': 4.0}
    RETEST_WINDOW_SECS    = 360.0     # Bug fix: was 200s — killed valid setups that needed time to retest
    # v25-LIVE: If Phase 1 impulse was very strong (>=15 pts), allow even more time.
    # Strong impulses prove conviction — they deserve extra retest time.
    RETEST_WINDOW_STRONG_IMPULSE = 480.0   # 8 min for impulses >= 15 pts
    RETEST_STRONG_IMPULSE_PTS    = 15.0    # threshold to qualify for extended window

    # v18: Blind hold reduced, with early escape
    MIN_HOLD_SECONDS      = 15.0       # v23: 15s blind hold (was 30s)
    HOLD_ESCAPE_PTS       = 8.0        # if trade hits 8+ pts, blind hold permanently off




    # v17.4: Dead zone REMOVED — engine quality gates handle all regimes
    # (Previously: 12:00-13:00 blocked. Data shows no consistent midday penalty.)

    # v30: No hard caps — smart conviction scaling prevents overtrading organically.
    # The engine's weighted vote threshold and conviction minimum are the real gates.
    # A system that needs external trade limits has broken internal signal quality.
    MAX_TRADES_PER_HOUR = 9999         # effectively disabled — conviction gate decides
    MAX_TRADES_PER_DAY  = 9999         # effectively disabled — conviction gate decides
    MAX_CONSEC_SL_SAME_DIR = 9999      # effectively disabled — conviction ramp is the organic brake

    CONFIG_JSON       = os.path.join(_DIR, "hawk_config.json")
    TRADE_LOG         = os.path.join(_DIR, "hawk_trades.json")
    LEARNING_FILE     = os.path.join(_DIR, "hawk_learning.json")
    LOG_FILE          = os.path.join(_DIR, "logs", "hawk_main.log")

# ── Logging setup ────────────────────────────────────────────────────────────
if _IS_MAIN:
    os.makedirs(os.path.dirname(CFG.LOG_FILE), exist_ok=True)
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    _fh  = logging.handlers.RotatingFileHandler(
        CFG.LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    _fh.setFormatter(_fmt); _fh.setLevel(logging.DEBUG)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(_fmt); _ch.setLevel(logging.CRITICAL)
    log = logging.getLogger("hawk")
    log.setLevel(logging.INFO)
    log.addHandler(_fh); log.addHandler(_ch)
    for _bl in ("WebsocketLogger", "APILogger"):
        logging.getLogger(_bl).propagate = False
else:
    log = logging.getLogger("hawk_child")
    log.addHandler(logging.NullHandler())

# =============================================================================
# PREMIUM TRACKER  (ported from backtest — gates entries on momentum)
# =============================================================================
class PremTracker:
    """Tracks recent ATM CE/PE premium history to gate entries.
    Ensures we only enter when premium is actively rising and not fading.
    Aligned with backtest PremTracker exactly."""
    def __init__(self):
        self.ce: deque = deque(maxlen=15)
        self.pe: deque = deque(maxlen=15)

    def update(self, ce_ltp: float, pe_ltp: float):
        if ce_ltp > 0: self.ce.append(ce_ltp)
        if pe_ltp  > 0: self.pe.append(pe_ltp)

    def rising(self, direction: str, n: int = 3) -> bool:
        """Latest premium is higher than n bars ago. True if not enough data (benefit of doubt)."""
        h = list(self.ce) if direction == "CE" else list(self.pe)
        if len(h) < n + 1: return True
        return h[-1] > h[-(n + 1)]

    def not_fading(self, direction: str, n: int = 10, mx: float = 0.02) -> bool:
        """Premium has not dropped more than mx% from its n-bar peak. True if insufficient data."""
        h = list(self.ce) if direction == "CE" else list(self.pe)
        if len(h) < n: return True
        pk = max(h[-n:])
        return ((pk - h[-1]) / pk) <= mx if pk > 0 else True

    def reset(self):
        self.ce.clear(); self.pe.clear()


# =============================================================================
# DECISION QUEUE
# =============================================================================
class DecisionQueue:
    def __init__(self):
        self._d = None
        self._lock = threading.Lock()
    def put(self, d: dict):
        with self._lock:
            self._d = d
    def peek(self) -> Optional[dict]:
        with self._lock:
            return self._d
    def get_latest(self) -> Optional[dict]:
        with self._lock:
            d = self._d
            self._d = None
            return d

# =============================================================================
# BRAIN BRIDGE
# =============================================================================
class BrainBridge:
    SNAP_Q_MAX     = 50
    DECISION_Q_MAX = 5
    RESULT_Q_MAX   = 200

    def __init__(self):
        self.snap_mp_q     = mp.Queue(maxsize=self.SNAP_Q_MAX)
        self.decision_mp_q = mp.Queue(maxsize=self.DECISION_Q_MAX)
        self.result_q      = mp.Queue(maxsize=self.RESULT_Q_MAX)
        self.decision_q    = DecisionQueue()
        self._process: Optional[mp.Process] = None
        self._bridge: Optional[threading.Thread] = None
        self._running = False
        self._snap_count = 0

    def start(self):
        self._running = True
        try:
            from hawk_brain import run_brain
        except ImportError:
            log.critical("BrainBridge: hawk_brain.py not found")
            raise
        self._process = mp.Process(
            target=run_brain,
            args=(self.snap_mp_q, self.decision_mp_q, self.result_q),
            daemon=True, name="HawkBrain",
        )
        self._process.start()
        log.info("BrainBridge: HawkBrain started (pid=%d)", self._process.pid)
        self._bridge = threading.Thread(
            target=self._drain, daemon=True, name="BrainBridge")
        self._bridge.start()

    def stop(self):
        self._running = False
        try:
            self.snap_mp_q.put_nowait(None)
        except Exception:
            pass
        if self._bridge and self._bridge.is_alive():
            self._bridge.join(timeout=1.0)
        if self._process and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def feed(self, snap: dict):
        try:
            self.snap_mp_q.put_nowait(snap)
        except Exception:
            try:
                self.snap_mp_q.get_nowait()
                self.snap_mp_q.put_nowait(snap)
            except Exception:
                pass

    def send_result(self, result: dict):
        try:
            self.result_q.put_nowait(result)
        except Exception:
            pass

    def _drain(self):
        _first = True
        _last_hb = time.monotonic()
        while self._running:
            try:
                d = self.decision_mp_q.get(timeout=0.5)
                self.decision_q.put(d)
                if _first:
                    log.info("BrainBridge: first decision received dir=%s score=%.2f",
                             d.get("entry_direction","?"), d.get("entry_conviction",0))
                    _first = False
                now = time.monotonic()
                if now - _last_hb > 60:
                    log.info("BrainBridge: drain alive, brain pid=%s alive=%s",
                             self._process.pid if self._process else "?",
                             self.is_alive())
                    _last_hb = now
            except Exception:
                pass
        log.warning("BrainBridge._drain: _running cleared, thread exiting")

# =============================================================================
# PAPER TRADER
# =============================================================================
@dataclass
class Trade:
    id:           int
    direction:    str
    strike:       float
    entry_price:  float
    entry_time:   str
    entry_epoch:  float
    stop_loss:    float
    target:       float
    sl_pts:       float
    tgt_pts:      float
    peak_price:   float      = 0.0
    trail_sl:     Optional[float] = None
    exit_price:   Optional[float] = None
    exit_time:    Optional[str]   = None
    exit_reason:  Optional[str]   = None
    pnl_pts:      Optional[float] = None
    status:       str             = "OPEN"
    conviction:   float           = 0.0
    smart_money:  str             = ""
    votes:        int             = 0
    entry_capital: float          = 0.0
    tightened:    bool            = False
    vix_regime:   str             = "MID"
    sl_adapted:   bool            = False
    probability:  float           = 0.0
    # v18: Chandelier exit fields (no partial booking)
    moved_to_breakeven: bool      = False
    hold_escaped:       bool      = False  # v18: blind hold auto-disabled at 8+ pts
    # v19.2: Conviction decay — track entry conviction to detect thesis collapse
    entry_conviction: float       = 0.0    # conviction at time of entry
    conv_tightened:   bool        = False   # v19.3: SL tightened due to conviction decay
    # v21: Premium history for momentum-aware chandelier
    ltp_history:      list        = field(default_factory=list)
    _ltp_hist_max:    int         = 40     # keep last ~40 readings

class PaperTrader:
    def __init__(self, prices: PriceStore):
        self._prices    = prices
        self._lock      = threading.Lock()
        self._next_id   = 1
        self.capital    = CFG.STARTING_CAPITAL
        self.open:  List[Trade] = []
        self.closed: List[Trade] = []
        self._last_trade_epoch: float = 0.0
        self._last_exit_epoch: float  = 0.0
        self._last_exit_was_sl: bool  = False  # True = loss, triggers SL_COOLDOWN
        self._consec_sl: dict = {"CE": 0, "PE": 0}
        self._dir_blocked: dict = {"CE": False, "PE": False}  # v17: hard block after MAX_CONSEC_SL
        self._recent_entry_epochs: deque = deque(maxlen=20)     # v17: for hourly rate limit
        self._last_trade_direction: str = "NEUTRAL"             # v29: track for flip-entry conviction bump
        self.reentry_block: str  = ""
        self.consecutive_ticks_ce = 0
        self.consecutive_ticks_pe = 0
        self._load()

    # ── Entry ─────────────────────────────────────────────────────────────────
    def enter(self, direction: str, conviction: float,
              votes: int = 0, smart_money: str = "", suggested_sl: float = 0.0,
              brain_decision: dict = None) -> Optional[Trade]:
        with self._lock:
            if len(self.open) >= CFG.MAX_OPEN:
                log.info("Entry rejected: MAX_OPEN reached (%d)", CFG.MAX_OPEN)
                return None

            now = time.monotonic()

            # v30: No time-based cooldowns, no hard blocks, no rate limits.
            # The ONLY gate is conviction. The conviction floor rises organically with
            # consecutive losses — if the signal is genuinely strong after 3 SLs, take it.
            # If it isn't strong enough to clear the raised bar, it shouldn't be taken.

            # v30: Removed hard direction block — _dir_blocked no longer gates entry.
            # Keeping the dict for dashboard display continuity (shows consecutive SL count).

            # v17: Hourly rate limit removed — no longer needed with raised conviction floor
            _ = now  # suppress unused warning; now used below for epoch recording

            consec_same = self._consec_sl.get(direction, 0)
            min_conv = CFG.CONVICTION_MIN

            # v30: Steepened conviction ramp — this is now the SOLE quality gate.
            # No hard blocks, no cooldowns. The ramp must be steep enough that a system
            # genuinely getting it wrong in one direction can't keep entering at low cost.
            #
            # Ramp schedule (base CONVICTION_MIN = 0.62):
            #   0 SLs:  0.62  — normal entry bar
            #   1 SL:   0.67  — market pushed back once; need slightly more confirmation
            #   2 SLs:  0.74  — two consecutive losses; conviction must be clear
            #   3 SLs:  0.82  — three consecutive losses; only very high confidence enters
            #   4 SLs:  0.88  — near-certain signal required
            #   5+ SLs: 0.93  — essentially: wait for a different setup entirely
            #
            # A genuine reversal signal after 5 consecutive SLs will score > 0.93 because:
            #   - RSI (weight 1.6) + BB_SQUEEZE (1.2) + STRUCTURE (1.5) + PREM_VEL (2.0)
            #     all voting together = wvote >> 6.0 = high conviction
            # A revenge/noise entry after 5 SLs will score 0.62–0.75 — correctly blocked.
            ramp = [0.0, 0.05, 0.12, 0.20, 0.26, 0.31]
            bump = ramp[min(consec_same, len(ramp) - 1)]
            min_conv += bump

            # v29: Direction-flip conviction bump.
            # If last trade was CE and this entry is PE (or vice versa), demand +0.06 extra.
            # Rationale: flip entries are entering AGAINST residual momentum — they need
            # stronger signal confirmation to overcome the existing trend inertia.
            # The engine's ExhaustionGate blocks premature flips at the signal level;
            # this is the trader-level second gate — even if engine passes a flip through,
            # the conviction must be meaningfully higher than a continuation entry.
            is_dir_flip = (self._last_trade_direction not in ("NEUTRAL", direction)
                           and self._last_trade_direction != "")
            if is_dir_flip:
                min_conv += 0.06
                log.debug("DirFlip bump: last=%s new=%s min_conv raised to %.2f",
                          self._last_trade_direction, direction, min_conv)

            if conviction < min_conv:
                flip_tag = " [DIR_FLIP]" if is_dir_flip else ""
                msg = (f"ReEntry gate{flip_tag}: {consec_same}×SL — need {min_conv:.0%} conv got {conviction:.0%}")
                log.info(msg)
                self.reentry_block = msg
                return None
            else:
                self.reentry_block = ""

            atm   = self._prices.atm
            right = "call" if direction == "CE" else "put"
            price = self._prices.get_option_price(atm, right)
            if price < CFG.MIN_PREMIUM:
                return None

            bid = self._prices.opt_bid.get((atm, right), 0.0)
            ask = self._prices.opt_ask.get((atm, right), 0.0)
            if bid > 0 and ask > 0:
                spread = ask - bid
                if spread > CFG.MAX_SPREAD:
                    return None

            now_ist = datetime.now(IST)
            cutoff  = now_ist.replace(hour=CFG.NO_ENTRY_AFTER_H, minute=CFG.NO_ENTRY_AFTER_M, second=0, microsecond=0)
            if now_ist >= cutoff:
                return None

            entry   = price
            if entry > CFG.MAX_PREMIUM:
                return None

            final_sl = CFG.SL_MIN
            final_tgt = final_sl * 3.0  # v17: 7 * 3 = 21pts target
            sl_lvl  = entry - final_sl
            tgt_lvl = entry + final_tgt

            vix_regime = brain_decision.get("vix_regime", "MID") if brain_decision else "MID"
            prob = float(brain_decision.get("probability", 0.0)) if brain_decision else 0.0

            t = Trade(
                id          = self._next_id,
                direction   = direction,
                strike      = atm,
                entry_price = entry,
                entry_time  = now_ist.strftime("%H:%M:%S"),
                entry_epoch = time.monotonic(),
                stop_loss   = sl_lvl,
                target      = tgt_lvl,
                sl_pts      = final_sl,
                tgt_pts     = final_tgt,
                peak_price  = entry,
                conviction  = conviction,
                smart_money = smart_money,
                votes       = votes,
                entry_capital = self.capital,
                tightened   = False,
                vix_regime  = vix_regime,
                sl_adapted  = False,
                probability = prob,
                entry_conviction = conviction,  # v19.2: track for conviction decay exit
            )
            self.open.append(t)
            self._next_id += 1
            self._last_trade_epoch = time.monotonic()
            self._last_trade_direction = direction          # v29: track for flip-entry conviction bump
            self._recent_entry_epochs.append(self._last_trade_epoch)  # v17: rate limit tracking
            log.info("ENTRY #%d %s %s @₹%.2f SL=%.1f TGT=%.1f P(win)=%.0f%% VIX=%s",
                     t.id, direction, atm, entry, final_sl, final_tgt, prob*100, vix_regime)
            self._save()
            return t

    # ── Exit checks ───────────────────────────────────────────────────────────
    def check_exits(self, brain_decision: Optional[dict] = None) -> List[Tuple[Trade, str]]:
        closed = []
        with self._lock:
            for t in list(self.open):
                result = self._check_exit(t, brain_decision=brain_decision)
                if result:
                    closed.append((t, result))
                    self.open.remove(t)
                    self.closed.append(t)
        if closed:
            self._save()
        return closed

    def _check_exit(self, t: Trade, brain_decision: Optional[dict] = None) -> Optional[str]:
        right = "call" if t.direction == "CE" else "put"
        price = self._prices.get_option_price(t.strike, right)
        if price <= 0:
            return None

        # v18: Blind hold — 30s, but escape permanently if trade hits 8+ pts
        elapsed_s = time.monotonic() - t.entry_epoch
        in_hold = elapsed_s < CFG.MIN_HOLD_SECONDS and not t.hold_escaped
        if in_hold:
            cp_hold = price - t.entry_price
            if cp_hold >= CFG.HOLD_ESCAPE_PTS:
                t.hold_escaped = True
                in_hold = False
            if in_hold:
                if price > t.peak_price:
                    t.peak_price = price
                # Hard SL still active during blind hold
                if price <= t.stop_loss:
                    t.exit_price = price
                    t.exit_time = datetime.now(IST).strftime("%H:%M:%S")
                    t.exit_reason = "SL HIT"
                    t.pnl_pts = price - t.entry_price
                    t.status = "CLOSED"
                    self._finalize_trade(t)
                    return "SL HIT"

                return None

        if price > t.peak_price:
            t.peak_price = price
        peak_profit = t.peak_price - t.entry_price
        current_profit = price - t.entry_price

        # ── Stage 1: Stall check — kill dead trades or tighten SL ──
        if not t.sl_adapted:
            elapsed = time.monotonic() - t.entry_epoch
            # v28 Innovation H: Theta-decay adaptive patience
            # (Aristotle Phase 3 — Reconstruct from Atoms):
            #   Theta bleeds 3-4x faster after 14:00 IST. A stalling trade in the
            #   afternoon costs more per second of waiting. Reduce patience window
            #   so we don't hemorrhage theta on dead trades.
            _now_h = datetime.now(IST).hour
            if _now_h >= 14:
                _stall_patience = CFG.STALL_PATIENCE_LATE       # 75s
            elif _now_h >= 13:
                _stall_patience = CFG.STALL_PATIENCE_AFTERNOON  # 100s
            else:
                _stall_patience = CFG.ADAPTIVE_SL_SECONDS       # 120s (morning)

            if elapsed >= _stall_patience and peak_profit < CFG.ADAPTIVE_SL_MIN_MOVE:
                if CFG.STALL_KILL and peak_profit < CFG.STALL_KILL_THRESHOLD:
                    log.info("STALL KILL #%d: dead trade (peak=%.2f<%.1f in %.0fs, patience=%.0fs) -> exit at market",
                             t.id, peak_profit, CFG.STALL_KILL_THRESHOLD, elapsed, _stall_patience)
                    t.exit_price = price
                    t.exit_time = datetime.now(IST).strftime("%H:%M:%S")
                    t.exit_reason = "STALL KILL"
                    t.pnl_pts = price - t.entry_price
                    t.status = "CLOSED"
                    self._finalize_trade(t)
                    return "STALL KILL"
                new_sl = t.entry_price - CFG.SL_TIGHT
                if new_sl > t.stop_loss:
                    t.stop_loss = new_sl
                    t.sl_adapted = True
                    log.info("ADAPTIVE SL #%d: stalled (peak=%.2f<%.1f in %.0fs, patience=%.0fs) -> ep-%.1f",
                             t.id, peak_profit, CFG.ADAPTIVE_SL_MIN_MOVE, elapsed, _stall_patience, CFG.SL_TIGHT)


        # ── Stage 1.3: v28 SOFT TRAIL — protect developing trades from catastrophic reversal ──
        # (Aristotle Phase 1): Trades peaking 4-7pts had ZERO trailing protection.
        # A 7pt winner could become an 11pt loser — 18pt round trip that destroys capital.
        # (Aristotle Phase 3): Graduated SL reduction, NOT profit locking.
        #   formula: soft_offset = max(0, SL_START - (peak - TRIGGER) × RATE)
        #   At peak 4: entry-6 | peak 5: entry-4 | peak 6: entry-2 | peak 7: entry-0
        # (Aristotle Phase 4): Cannot cause chillad chori — SL stays ≤ entry throughout.
        #   Chandelier (entry+4.5) takes over at peak 8 and dominates via max().
        # (Aristotle Phase 5): Safe during blind hold (entry-6 is still a loss stop).
        #   Interaction with conviction decay: conv_decay sets entry-5 at peak<5,
        #   soft trail sets entry-6 at peak=4. Conv_decay overrides (tighter), correct.
        if peak_profit >= CFG.SOFT_TRAIL_TRIGGER and peak_profit < CFG.CHANDELIER_MIN_PROFIT:
            _soft_offset = max(0.0, CFG.SOFT_TRAIL_SL_START -
                               (peak_profit - CFG.SOFT_TRAIL_TRIGGER) * CFG.SOFT_TRAIL_RATE)
            _soft_sl = t.entry_price - _soft_offset
            if _soft_sl > t.stop_loss:
                t.stop_loss = _soft_sl
                log.info("SOFT TRAIL #%d: peak=%.1f -> SL raised to entry%+.1f (max loss=%.1fpts)",
                         t.id, peak_profit, -_soft_offset, _soft_offset)

        # ── Stage 1.5: CONVICTION DECAY EXIT — thesis is collapsing ──────
        # v19.3: Conviction as EXIT INTELLIGENCE — tighten SL, DON'T panic exit.
        # Direction flip or conviction crash → tighten SL from 11pts to 5pts risk.
        # Bug fix: was firing within 60s of entry (62s in practice), halving the SL
        # before the trade had any chance to breathe. New guards:
        #   1. Minimum 180s hold (3 min) before this can activate.
        #   2. Only fires if peak_profit < 5 pts — if already in profit the
        #      chandelier trail handles protection; don't override it here.
        elapsed_s_decay = time.monotonic() - t.entry_epoch
        if brain_decision and elapsed_s_decay >= 180.0 and not t.conv_tightened and peak_profit < 5.0:
            live_conv = brain_decision.get("entry_conviction", 0.0)
            live_dir  = brain_decision.get("entry_direction", "NEUTRAL")
            entry_conv = t.entry_conviction
            conv_drop = entry_conv - live_conv
            dir_flipped = (live_dir != "NEUTRAL" and live_dir != t.direction)

            if dir_flipped or (conv_drop >= 0.20 and live_conv < 0.45):
                tight_sl = t.entry_price - 5.0  # tighten from 11 to 5pts risk
                if tight_sl > t.stop_loss:
                    t.stop_loss = tight_sl
                    t.conv_tightened = True
                    log.info("CONV TIGHTEN #%d: thesis weakening (conv %.2f->%.2f) -> SL tightened to entry-5",
                             t.id, entry_conv, live_conv)

        # ── Stage 2: Move to breakeven once in profit by 1× SL ──
        if not t.moved_to_breakeven and peak_profit >= CFG.BREAKEVEN_TRIGGER:
            be_sl = t.entry_price + CFG.BREAKEVEN_BUFFER
            if be_sl > t.stop_loss:
                t.stop_loss = be_sl
                t.moved_to_breakeven = True
                log.info("BREAKEVEN #%d: peak=%.1f → SL moved to entry+%.1f",
                         t.id, peak_profit, CFG.BREAKEVEN_BUFFER)

        # ── Stage 3: v21 Momentum-Aware Chandelier Exit ──
        # Base lock% from peak tier, adjusted by premium velocity.
        # Rising premium → loose trail (let winners run).
        # Falling premium → tight trail (protect before pullback kills profit).
        trail_sl = t.trail_sl if t.trail_sl is not None else t.stop_loss

        # Track premium history for velocity calculation
        t.ltp_history.append(price)
        if len(t.ltp_history) > t._ltp_hist_max:
            t.ltp_history = t.ltp_history[-t._ltp_hist_max:]

        if peak_profit >= CFG.CHANDELIER_MIN_PROFIT:
            # v28 Innovation F: Premium-scaled floor replaces fixed TRAIL_MIN_OFFSET.
            # (Aristotle Phase 3 — Reconstruct from Atoms):
            #   The floor must scale with option premium because pullback noise does.
            #   A 200pt option swings 5-8pts during a trend; fixed 3.5 floor kills it.
            #   Premium floor = max(3.5, entry_price * 0.025) adapts to the option price.
            #   This floor is used in ALL three tiers below instead of the fixed constant.
            _prem_floor = max(CFG.TRAIL_MIN_OFFSET, t.entry_price * CFG.TRAIL_FLOOR_PCT)

            # v27: TIERED offset — loose on small peaks, tighter on big runners.
            # Small peaks (8-12pts) need 28% room to breathe into larger moves.
            # Mid runners (12-20pts) tighten to 22% — already proven territory.
            # Big runners (>20pts) use 18% — lock the bulk of exceptional moves.
            if peak_profit <= 12.0:
                offset = max(_prem_floor, peak_profit * 0.28)
            elif peak_profit <= 20.0:
                offset = max(_prem_floor, peak_profit * 0.22)
            else:
                offset = max(_prem_floor, peak_profit * 0.18)

            # ── v28 Innovation C: Exhaustion Detector — tighten trail when move is dying ──
            # When the Market Phase State Machine detects EXHAUSTION, the move is losing
            # momentum (large displacement + velocity decaying). Reduce the chandelier
            # offset to lock more profit before the reversal hits.
            #
            # Additionally, when futures displacement opposes the trade direction,
            # that's a leading indicator the move is reversing — tighten further.
            #
            # The offset reduction is multiplicative (not additive) to preserve the
            # tiered structure above. Minimum offset of 3.0pts prevents over-tightening.
            if brain_decision:
                _exit_phase = brain_decision.get("market_phase", "UNKNOWN")
                _exit_fdc = brain_decision.get("fut_vel_disp", 0.0)

                # v28 Innovation F: Phase-based offset modifiers (mutually exclusive)
                # (Aristotle Phase 3 — Reconstruct from Atoms):
                #   TREND phase = phase machine confirmed sustained directional movement.
                #   Wider trail gives the move runway to develop into a 50-70pt runner.
                #   EXHAUSTION/DISTRIBUTION = move is dying or choppy, tighten to protect.
                #   These are mutually exclusive — a move cannot be simultaneously trending
                #   and exhausting. The phase machine guarantees this.
                if _exit_phase == "EXHAUSTION":
                    offset *= 0.70  # 30% tighter — move is dying, lock profit
                elif _exit_phase == "DISTRIBUTION":
                    offset *= 0.85  # 15% tighter — choppy redistribution
                elif _exit_phase == "TREND":
                    offset *= 1.35  # 35% wider — confirmed trend, let it run

                # FDC reversal tightening: if displacement opposes trade direction
                # (e.g., bought CE but futures pulling back), tighten by 15%
                if t.direction == "CE" and _exit_fdc < -0.3:
                    offset *= 0.85  # futures reversing against CE trade
                elif t.direction == "PE" and _exit_fdc > 0.3:
                    offset *= 0.85  # futures reversing against PE trade

                offset = max(3.0, offset)  # absolute floor — never choke tighter than 3pts

            chandelier_sl = t.entry_price + peak_profit - offset  # = peak_price - offset
            trail_sl = max(trail_sl, chandelier_sl)

        t.trail_sl = trail_sl
        effective_sl = max(t.stop_loss, t.trail_sl)

        reason = None
        now_ist = datetime.now(IST)
        current_mins = now_ist.hour * 60 + now_ist.minute

        EOD_CLOSE_MINS = 15 * 60 + 25
        if current_mins >= EOD_CLOSE_MINS:
            reason = "EOD CLOSE"
        elif price <= effective_sl:
            reason = "TRAIL SL" if t.trail_sl > t.stop_loss else "SL HIT"

        if reason is None and (time.monotonic() - t.entry_epoch) > 3600:
            reason = "TIMEOUT"

        if reason:
            EXEC_SLIP = 1.0
            if reason == "EOD CLOSE" or reason == "TIMEOUT":
                exit_px = price
            elif reason == "TRAIL SL":
                exit_px = max(price, t.trail_sl - EXEC_SLIP)
            else:  # SL HIT
                exit_px = max(price, t.stop_loss - EXEC_SLIP)

            t.exit_price  = exit_px
            t.exit_time   = now_ist.strftime("%H:%M:%S")
            t.exit_reason = reason
            t.pnl_pts     = t.exit_price - t.entry_price
            t.status      = "CLOSED"
            self._finalize_trade(t)
            return reason

        return None

    def _finalize_trade(self, t: Trade):
        # v18: All-in — simple PnL, no partial booking
        t.pnl_pts = t.exit_price - t.entry_price
        pnl_pct = t.pnl_pts / t.entry_price if t.entry_price > 0 else 0.0
        base_cap = t.entry_capital if t.entry_capital > 0 else self.capital
        pnl_cash = base_cap * pnl_pct
        self.capital = base_cap + pnl_cash
        self._last_exit_epoch = time.monotonic()

        # Backtest: loss (pnl<0) → SL cooldown; win (pnl>=0) → win cooldown
        is_loss = (t.pnl_pts or 0) < 0
        self._last_exit_was_sl = is_loss

        if is_loss:
            self._consec_sl[t.direction] = min(self._consec_sl.get(t.direction, 0) + 1, 6)
            opp = "PE" if t.direction == "CE" else "CE"
            self._consec_sl[opp] = 0
            # v30: No hard block. The consecutive SL counter feeds the conviction ramp
            # in enter() — the system naturally demands higher certainty after losses.
            # A hard block ignores a genuine 0.90+ conviction setup just because it's
            # the 4th trade. That is not intelligence, it's a circuit breaker.
            log.info("consec_sl[%s]=%d — conviction ramp active",
                     t.direction, self._consec_sl[t.direction])
        else:
            self._consec_sl[t.direction] = 0
            self._dir_blocked[t.direction] = False

        sl_str = f" consec_sl={self._consec_sl[t.direction]}" if is_loss else ""
        log.info("EXIT #%d %s %+.1fpts (%+.2f%%) ₹%+.0f→₹%.0f [%s] peak=%.1f sl=%.1f%s",
                 t.id, t.direction, t.pnl_pts, pnl_pct * 100,
                 pnl_cash, self.capital, t.exit_reason, t.peak_price - t.entry_price,
                 t.stop_loss, sl_str)

    def snapshot_open(self) -> list:
        with self._lock: return list(self.open)

    def snapshot_closed(self, n: int = 6) -> list:
        with self._lock: return list(self.closed[-n:])

    @property
    def wins(self) -> int:
        with self._lock: return sum(1 for t in self.closed if (t.pnl_pts or 0) >= 0)

    @property
    def losses(self) -> int:
        with self._lock: return sum(1 for t in self.closed if (t.pnl_pts or 0) < 0)

    @property
    def total_pts(self) -> float:
        with self._lock: return sum(t.pnl_pts or 0 for t in self.closed)

    @property
    def win_rate(self) -> float:
        with self._lock:
            n = len(self.closed)
            return (sum(1 for t in self.closed if (t.pnl_pts or 0) >= 0) / n * 100) if n > 0 else 0.0

    def _save(self):
        try:
            data = {"capital": self.capital, "next_id": self._next_id, "trades": [self._trade_dict(t) for t in self.closed]}
            tmp = CFG.TRADE_LOG + ".tmp"
            with open(tmp, "w") as f: json.dump(data, f, indent=2)
            os.replace(tmp, CFG.TRADE_LOG)
        except Exception as e:
            log.warning("Trade save failed: %s", e)

    def _load(self):
        try:
            if not os.path.exists(CFG.TRADE_LOG): return
            with open(CFG.TRADE_LOG) as f: data = json.load(f)
            self.capital  = data.get("capital", CFG.STARTING_CAPITAL)
            self._next_id = data.get("next_id", 1)
            for td in data.get("trades", []):
                t = Trade(**{k: td[k] for k in td if k in Trade.__dataclass_fields__})
                self.closed.append(t)
            log.info("Loaded %d past trades, capital=₹%.0f", len(self.closed), self.capital)
        except Exception as e:
            log.warning("Trade load failed: %s", e)

    @staticmethod
    def _trade_dict(t: Trade) -> dict:
        return {k: getattr(t, k) for k in Trade.__dataclass_fields__}

# =============================================================================
# DASHBOARD
# =============================================================================
class Dashboard:
    PHASE_WAITING    = "WAITING"
    PHASE_CONNECTING = "CONNECTING"
    PHASE_LIVE       = "LIVE"

    def __init__(self, prices: Optional[PriceStore] = None,
                 trader: Optional[PaperTrader] = None,
                 brain: Optional[BrainBridge] = None):
        self._p      = prices
        self._trader = trader
        self._brain  = brain
        self.events: deque = deque(maxlen=200)
        self._events_lock  = threading.Lock()
        self.status: str   = "Starting…"
        self.phase: str    = self.PHASE_WAITING
        self.phase_detail: str = ""
        self.connect_target: str = ""
        self._start = time.monotonic()
        self.pending_info: Optional[dict] = None  # v25-LIVE: retest phase visibility
        # v28 Innovation 3: Dashboard block counter + last discard tracking
        self._block_counts: dict = {}    # {reason_prefix: count}
        self._last_discard: Optional[dict] = None  # {dir, reason, ts}

    def set_components(self, prices: PriceStore, trader: PaperTrader, brain: BrainBridge):
        self._p = prices
        self._trader = trader
        self._brain = brain

    def record_block(self, reason: str):
        """v28: Track block reason counts for dashboard display."""
        prefix = reason.split(":")[0] if ":" in reason else reason[:20]
        self._block_counts[prefix] = self._block_counts.get(prefix, 0) + 1

    def record_discard(self, direction: str, reason: str):
        """v28: Track last discarded signal for dashboard display."""
        self._last_discard = {'dir': direction, 'reason': reason, 'ts': time.monotonic()}

    def event(self, msg: str):
        ts = datetime.now(IST).strftime("%H:%M:%S")
        with self._events_lock:
            self.events.appendleft(f"[{ts}] {msg}")

    def renderable(self):
        try: return self._build()
        except Exception as e: return Text(f"Render error: {e}")

    def _build(self):
        now_ist = datetime.now(IST)
        p = self._p
        tr = self._trader
        d = self._brain.decision_q.peek() if self._brain else None

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        layout["body"].split_row(
            Layout(name="left", ratio=5),
            Layout(name="right", ratio=5)
        )
        layout["left"].split_column(
            Layout(name="engine"),
        )
        layout["right"].split_column(
            Layout(name="prices", size=5),
            Layout(name="trades", size=9),
            Layout(name="events", size=8),
            Layout(name="recent")
        )

        cap = tr.capital if tr else CFG.STARTING_CAPITAL
        cap_pct = ((cap / CFG.STARTING_CAPITAL) - 1.0) * 100
        cap_col = "green" if cap_pct >= 0 else "red"
        conv = float(d.get("entry_conviction", 0.0)) if d else 0.0
        conv_col = "green" if conv >= CFG.CONVICTION_MIN else "yellow"
        vix_regime = str(d.get("vix_regime", "—")) if d else "—"
        vix_col = {"LOW": "green", "MID": "cyan", "HIGH": "yellow", "EXTREME": "red bold"}.get(vix_regime, "dim")
        prob = float(d.get("probability", 0.0)) if d else 0.0
        ims = float(d.get("imm_momentum", 0.0)) if d else 0.0
        ims_col = "green" if ims > 0.1 else "red" if ims < -0.05 else "yellow"
        prob_col = "green" if prob >= 0.55 else "red" if prob < 0.40 else "yellow"

        # v27: Determine trading phase (P0→P4) from current state
        _open_trades = tr.snapshot_open() if tr else []
        _pend = self.pending_info
        if _open_trades:
            _phase_label = "P4"
            _phase_desc  = "IN TRADE"
            _phase_col   = "bold green"
        elif _pend:
            _ph_map = {
                "first_move":      ("P1", "IMPULSE WAIT", "bold yellow"),
                "retest":          ("P2", "RETEST WAIT",  "bold cyan"),
                "confirmed_entry": ("P3", "CONFIRM WAIT", "bold magenta"),
            }
            _ph_raw = _pend.get("phase", "first_move")
            _phase_label, _phase_desc, _phase_col = _ph_map.get(_ph_raw, ("P1", _ph_raw.upper(), "yellow"))
        elif self.phase == self.PHASE_LIVE:
            _phase_label = "P0"
            _phase_desc  = "SCANNING"
            _phase_col   = "dim white"
        else:
            _phase_label = "—"
            _phase_desc  = self.phase
            _phase_col   = "dim"

        _phase_str = f"[{_phase_col}]{_phase_label}: {_phase_desc}[/{_phase_col}]"

        # v28 Innovation A: Market phase from state machine in header
        _mkt_phase = str(d.get("market_phase", "UNKNOWN")) if d else "UNKNOWN"
        _mkt_phase_cols = {
            "ACCUMULATION": "dim white", "EXPANSION": "bold green",
            "TREND": "bold cyan", "EXHAUSTION": "bold red",
            "DISTRIBUTION": "yellow", "UNKNOWN": "dim",
        }
        _mkt_phase_col = _mkt_phase_cols.get(_mkt_phase, "dim")

        head_text = (f"[bold cyan]🦅 HAWK v28[/bold cyan]  |  "
                     f"Phase: {_phase_str}  |  "
                     f"Mkt: [{_mkt_phase_col}]{_mkt_phase}[/{_mkt_phase_col}]  |  "
                     f"Capital: [{cap_col}]₹{cap:,.2f} ({cap_pct:+.2f}%)[/{cap_col}]  |  "
                     f"P(win): [{prob_col}]{prob:.0%}[/{prob_col}]  |  "
                     f"IMS: [{ims_col}]{ims:+.2f}[/{ims_col}]  |  "
                     f"VIX: [{vix_col}]{vix_regime}[/{vix_col}]")
        layout["header"].update(Panel(Align.center(Text.from_markup(head_text), vertical="middle"), style="bold blue", box=box.ROUNDED))

        if self.phase == self.PHASE_LIVE and d:
            direction = str(d.get("entry_direction", "NEUTRAL"))
            sm = str(d.get("smart_money", "UNCLEAR")[:15])
            blocked = str(d.get("blocked_reason", ""))

            bdr_style = "green" if d.get("entry_allowed") else "yellow"
            title_text = f"🧠 ENGINE ROOM | Leading: [bold]{direction}[/bold] | SM: {sm}"

            sig_table = Table(expand=True, show_lines=False, box=None, padding=(0, 1))
            sig_table.add_column("Vote", width=4, justify="center")
            sig_table.add_column("Indicator", style="cyan", width=15)
            sig_table.add_column("Reason", style="dim", overflow="fold")

            for v in (d.get("vote_detail", []) or []):
                vs = str(v)
                try:
                    name_part = vs.split(":")[0]
                    dir_part = vs.split(":")[1].split("(")[0]
                    reason = vs.split("[")[1].rstrip("]")
                    icon = "[green]▲[/green]" if dir_part == "CE" else "[red]▼[/red]" if dir_part == "PE" else "[dim]—[/dim]"
                    sig_table.add_row(icon, name_part, reason.replace("[", "\\[").replace("]", "\\]"))
                except (IndexError, ValueError, AttributeError):
                    # v28 Innovation D: Narrow exception — only catch parse failures, not bugs
                    sig_table.add_row("?", "Unknown", vs.replace("[", "\\[").replace("]", "\\]"))

            warns = d.get("reversal_warnings", [])
            if warns:
                sig_table.add_row("", "", "")
                for w in warns:
                    w_safe = str(w).replace("[", "\\[").replace("]", "\\]")
                    sig_table.add_row("[red]⚠[/red]", "[red]WARNING[/red]", f"[red]{w_safe}[/red]")

            # v25-LIVE: FULL GATE STATUS — show ALL gates at once, not just the first blocker
            g = d.get("gate_diagnostics", {})
            if g:
                sig_table.add_row("", "", "")
                sig_table.add_row("[bold cyan]━━[/bold cyan]", "[bold cyan]GATE STATUS[/bold cyan]", "[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

                # Votes & Weight
                ce_w = g.get("ce_weight", 0); pe_w = g.get("pe_weight", 0)
                wt = g.get("wt_needed", 5.0)
                lead_w = max(ce_w, pe_w); lead_dir = "CE" if ce_w >= pe_w else "PE"
                w_ok = lead_w >= wt
                w_icon = "[green]✓[/green]" if w_ok else "[red]✗[/red]"
                ce_cats_str = ",".join(g.get("ce_cats", []))
                pe_cats_str = ",".join(g.get("pe_cats", []))
                sig_table.add_row(w_icon, "Votes/Weight",
                    f"CE={g.get('ce_votes',0)}({ce_w:.1f}w [{ce_cats_str}])  PE={g.get('pe_votes',0)}({pe_w:.1f}w [{pe_cats_str}])  need {wt:.1f}w")

                # Category diversity — v28: show dynamic min (1 in trending/leading agree, else 2)
                lead_cats = g.get("ce_cats" if lead_dir == "CE" else "pe_cats", [])
                _regime_dash = g.get("regime", "?")
                _cat_min = 1 if _regime_dash in ("TRENDING", "BREAKOUT") else 2
                cat_ok = len(lead_cats) >= _cat_min
                c_icon = "[green]✓[/green]" if cat_ok else "[red]✗[/red]"
                sig_table.add_row(c_icon, "Categories", f"{lead_dir} has {len(lead_cats)} cats {lead_cats} (need {_cat_min}+)")

                # SV — v28: show soft/hard thresholds
                sv_val = g.get("sv", 0)
                sv_soft = g.get("sv_soft", 0.05)
                sv_hard = g.get("sv_hard", 0.30)
                sv_ok = sv_val < sv_soft
                sv_icon = "[green]✓[/green]" if sv_ok else "[red]✗[/red]" if sv_val >= sv_hard else "[yellow]~[/yellow]"
                sv_zone = "[green]CLEAN[/green]" if sv_val < sv_soft else "[red]HARD BLOCK[/red]" if sv_val >= sv_hard else "[yellow]PENALTY[/yellow]"
                sig_table.add_row(sv_icon, "SV (Straddle)", f"SV={sv_val:+.4f}  (soft>{sv_soft} hard>{sv_hard}) {sv_zone}")

                # IMS
                ims_val = g.get("ims", 0); ims_min = g.get("ims_min", -0.05); ims_max = g.get("ims_max", 0.75)
                ims_ok = ims_min <= ims_val <= ims_max
                ims_icon = "[green]✓[/green]" if ims_ok else "[red]✗[/red]"
                ims_zone = ""
                if ims_val < ims_min: ims_zone = " [red]OPPOSED[/red]"
                elif ims_val > ims_max: ims_zone = " [red]CHASING[/red]"
                elif ims_val > 0.20: ims_zone = " [yellow]HEAVY[/yellow]"
                elif ims_val > 0.12: ims_zone = " [yellow]PENALTY[/yellow]"
                else: ims_zone = " [green]CLEAN[/green]"
                sig_table.add_row(ims_icon, "IMS (Momentum)", f"IMS={ims_val:+.3f}  (need {ims_min} to {ims_max}){ims_zone}")

                # Conviction floor
                conv_val = float(d.get("entry_conviction", 0))
                conv_floor = g.get("conv_floor", 0.52)
                conv_ok = conv_val >= conv_floor
                conv_icon = "[green]✓[/green]" if conv_ok else "[red]✗[/red]"
                sig_table.add_row(conv_icon, "Conviction", f"Score={conv_val:.3f}  (floor={conv_floor:.3f} VIX:{g.get('vix_regime','?')})")

                # Freshness
                fresh_ok = g.get("is_fresh", True)
                f_icon = "[green]✓[/green]" if fresh_ok else "[red]✗[/red]"
                stale = g.get("stale_reason", "")
                sig_table.add_row(f_icon, "Freshness", "Fresh" if fresh_ok else f"Stale: {stale}")

                # Market regime
                regime = g.get("regime", "?")
                r_icon = "[green]✓[/green]" if regime == "TRENDING" else "[yellow]~[/yellow]"
                sig_table.add_row(r_icon, "Regime", f"{regime}  VIX={g.get('vix', 0):.1f} ({g.get('vix_regime','?')})")

                # v28 Innovation A: Market Phase State Machine
                _g_phase = g.get("phase", "UNKNOWN")
                _g_policy = g.get("phase_policy", "")
                _g_disp = g.get("phase_displacement", 0)
                _phase_ok = _g_phase in ("EXPANSION", "TREND")
                _phase_icon = "[green]✓[/green]" if _phase_ok else "[red]✗[/red]" if _g_phase == "EXHAUSTION" else "[yellow]~[/yellow]"
                _phase_style = {
                    "EXPANSION": "[bold green]", "TREND": "[bold cyan]",
                    "EXHAUSTION": "[bold red]", "ACCUMULATION": "[dim]",
                    "DISTRIBUTION": "[yellow]", "UNKNOWN": "[dim]",
                }.get(_g_phase, "[dim]")
                # v28.4: CUSUM early detection — now shows direction
                _cusum_fired = g.get("cusum_fired", False)
                _cusum_lvl = g.get("cusum_level", 0.0)
                _cusum_dir = g.get("cusum_direction", "")
                if _cusum_fired and _cusum_dir == "IV_SURGE":
                    _cusum_tag = "  [bold yellow]⚡IV_SURGE[/bold yellow]"
                elif _cusum_fired:
                    _cusum_tag = "  [bold magenta]⚡AWAKENING[/bold magenta]"
                elif _cusum_lvl > 0.005:
                    _cusum_tag = f"  [dim]cusum={_cusum_lvl:.3f}[/dim]"
                else:
                    _cusum_tag = ""
                sig_table.add_row(_phase_icon, "Phase", f"{_phase_style}{_g_phase}[/]  Disp={_g_disp:.0f}pt{_cusum_tag}")

                # v28 Innovation J: COI Strength — institutional positioning
                _coi = g.get("coi_strength", 0.0)
                if _coi > 40:
                    _coi_style = "[bold green]"; _coi_label = "BULLISH"
                elif _coi < -40:
                    _coi_style = "[bold red]"; _coi_label = "BEARISH"
                elif _coi > 15:
                    _coi_style = "[green]"; _coi_label = "Mild Bull"
                elif _coi < -15:
                    _coi_style = "[red]"; _coi_label = "Mild Bear"
                else:
                    _coi_style = "[dim]"; _coi_label = "Neutral"
                _coi_icon = "[green]↑[/green]" if _coi > 15 else "[red]↓[/red]" if _coi < -15 else "[dim]~[/dim]"
                sig_table.add_row(_coi_icon, "Institutions", f"{_coi_style}COI={_coi:+.0f}% {_coi_label}[/]")

                # v28 Innovation K: IV Skew Proxy — options market directional bias
                _skew = g.get("skew_ratio", 1.0)
                if _skew > 1.25:
                    _skew_style = "[bold red]"; _skew_label = "PE DEMAND (bearish skew)"
                elif _skew < 0.80:
                    _skew_style = "[bold green]"; _skew_label = "CE DEMAND (bullish skew)"
                elif _skew > 1.10:
                    _skew_style = "[red]"; _skew_label = "Mild PE bias"
                elif _skew < 0.90:
                    _skew_style = "[green]"; _skew_label = "Mild CE bias"
                else:
                    _skew_style = "[dim]"; _skew_label = "Balanced"
                _skew_icon = "[red]↓[/red]" if _skew > 1.10 else "[green]↑[/green]" if _skew < 0.90 else "[dim]=[/dim]"
                sig_table.add_row(_skew_icon, "IV Skew", f"{_skew_style}Ratio={_skew:.2f} {_skew_label}[/]")

            elif blocked:
                sig_table.add_row("", "", "")
                blocked_safe = blocked.replace("[", "\\[").replace("]", "\\]")
                sig_table.add_row("[yellow]⛔[/yellow]", "[yellow]BLOCKED[/yellow]", f"[yellow]{blocked_safe}[/yellow]")

            layout["engine"].update(Panel(sig_table, title=title_text, border_style=bdr_style))
        else:
            layout["engine"].update(Panel("[dim]Waiting for brain data...[/dim]", title="🧠 ENGINE ROOM"))

        spot = p.spot if p else 0.0; fut = p.futures if p else 0.0; atm = p.atm if p else 0.0
        ce_p = p.ce_ltp(atm) if p else 0; pe_p = p.pe_ltp(atm) if p else 0
        price_txt = (f"Spot: [bold]{spot:.2f}[/bold]  Fut: [bold]{fut:.2f}[/bold]  ATM: [bold yellow]{atm:.0f}[/bold yellow]\n"
                     f"[green]NIFTY {atm:.0f} CE: ₹{ce_p:.2f}[/green]  |  [red]NIFTY {atm:.0f} PE: ₹{pe_p:.2f}[/red]")
        layout["prices"].update(Panel(Text.from_markup(price_txt), title="📡 LIVE PRICES", border_style="green"))

        if tr and tr.open:
            open_tbl = Table(expand=True, show_lines=True, box=box.SIMPLE, padding=(0, 1))
            open_tbl.add_column("#", width=3, justify="right", style="dim")
            open_tbl.add_column("Option", style="bold cyan", min_width=22)
            open_tbl.add_column("Entry", justify="right", width=8)
            open_tbl.add_column("LTP", justify="right", width=8)
            open_tbl.add_column("PnL", justify="right", width=8)
            open_tbl.add_column("SL", justify="right", width=8)
            open_tbl.add_column("TGT", justify="right", width=8)
            open_tbl.add_column("Peak", justify="right", width=6)
            open_tbl.add_column("P(w)", justify="right", width=5)
            open_tbl.add_column("Flags", width=8)
            for t in tr.open:
                right = "call" if t.direction == "CE" else "put"
                curr = p.get_option_price(t.strike, right) if p else t.entry_price
                pts = curr - t.entry_price
                peak_pts = t.peak_price - t.entry_price
                col = "green" if pts >= 0 else "red"
                dir_col = "green" if t.direction == "CE" else "red"
                opt_name = f"[{dir_col}]NIFTY {t.strike:.0f} {t.direction}[/{dir_col}]"
                tsl_str = f"{t.trail_sl:.1f}" if t.trail_sl and t.trail_sl > t.stop_loss else f"{t.stop_loss:.1f}"
                # Status flags
                flags = []
                if t.moved_to_breakeven: flags.append("[green]BE[/green]")
                if t.hold_escaped: flags.append("[yellow]ESC[/yellow]")
                flags_str = " ".join(flags) if flags else "[dim]—[/dim]"
                open_tbl.add_row(
                    str(t.id),
                    opt_name,
                    f"₹{t.entry_price:.1f}",
                    f"[{col}]₹{curr:.1f}[/{col}]",
                    f"[{col}]{pts:+.1f}[/{col}]",
                    f"[red]{tsl_str}[/red]",
                    f"[cyan]{t.target:.1f}[/cyan]",
                    f"[yellow]{peak_pts:+.1f}[/yellow]",
                    f"{t.probability:.0%}",
                    flags_str,
                )
            layout["trades"].update(Panel(open_tbl, title=f"📈 OPEN ({len(tr.open)})", border_style="bold green"))
        else:
            layout["trades"].update(Panel("[dim]No active trades.[/dim]", title="📈 OPEN (0)", border_style="dim"))

        if tr and tr.closed:
            rec_tbl = Table(expand=True, show_lines=True, box=box.SIMPLE, padding=(0, 1))
            rec_tbl.add_column("", width=2, justify="center")
            rec_tbl.add_column("#", width=3, justify="right", style="dim")
            rec_tbl.add_column("Option", style="bold", min_width=22)
            rec_tbl.add_column("Entry", justify="right", width=8)
            rec_tbl.add_column("Exit", justify="right", width=8)
            rec_tbl.add_column("PnL", justify="right", width=8)
            rec_tbl.add_column("P(w)", justify="right", width=5)
            rec_tbl.add_column("Reason", width=12)
            rec_tbl.add_column("Time", width=11)
            for t in reversed(tr.snapshot_closed(8)):
                pnl = t.pnl_pts or 0
                col = "green" if pnl >= 0 else "red"
                icon = "✅" if pnl >= 0 else "❌"
                dir_col = "green" if t.direction == "CE" else "red"
                opt_name = f"[{dir_col}]NIFTY {t.strike:.0f} {t.direction}[/{dir_col}]"
                reason_safe = (t.exit_reason or "—").replace("[", "\\[").replace("]", "\\]")
                time_str = f"{t.entry_time}→{t.exit_time}" if t.exit_time else t.entry_time
                rec_tbl.add_row(
                    icon,
                    str(t.id),
                    opt_name,
                    f"₹{t.entry_price:.1f}",
                    f"₹{t.exit_price:.1f}" if t.exit_price else "—",
                    f"[{col}]{pnl:+.1f}[/{col}]",
                    f"{t.probability:.0%}",
                    reason_safe,
                    time_str,
                )
            layout["recent"].update(Panel(rec_tbl, title=f"📋 RECENT  W:{tr.wins} L:{tr.losses}  WR:{tr.win_rate:.0f}%  Total:{tr.total_pts:+.1f}pts", border_style="blue"))
        else:
            layout["recent"].update(Panel("[dim]No recent history.[/dim]", title="📋 RECENT"))

        # v25-LIVE: Pending signal / retest phase visibility
        pend = self.pending_info
        if pend:
            ph = pend['phase']
            dr = pend['dir']
            p0 = pend['sig_ltp']
            cl = pend['cur_ltp']
            hi = pend['high']
            fm = pend['first_move_req']
            el = pend['elapsed']
            wn = pend['window']
            pr_up = pend['price_up']

            ph_col = {"first_move": "yellow", "retest": "cyan", "confirmed_entry": "green"}.get(ph, "dim")
            ph_labels = {"first_move": "P1", "retest": "P2", "confirmed_entry": "P3"}
            ph_label = ph_labels.get(ph, "?")
            phase_txt = f"[bold {ph_col}]⏳ [{ph_label}] PENDING {dr}[/bold {ph_col}]  Phase: [bold]{ph}[/bold]  ({el:.0f}s / {wn:.0f}s)\n"

            if ph == 'first_move':
                move = hi - p0
                pct = move / fm * 100 if fm > 0 else 0
                bar_len = int(min(pct, 100) / 5)  # 20 chars max
                bar = "█" * bar_len + "░" * (20 - bar_len)
                phase_txt += f"  P0={p0:.1f}  High={hi:.1f}  Move={move:+.1f} / {fm:.1f} needed  [{bar}] {pct:.0f}%"
            elif ph == 'retest':
                dist = abs(cl - p0)
                # v28: show dynamic buffer
                _vr_dash = pend.get('vix_regime', 'MID')
                _pct_dash = CFG.RETEST_RETURN_PCT.get(_vr_dash, 0.06)
                buf = max(CFG.RETEST_RETURN_MIN, min(CFG.RETEST_RETURN_MAX, p0 * _pct_dash))
                phase_txt += f"  P0={p0:.1f}  LTP={cl:.1f}  Dist={dist:.1f} (need <={buf:.1f})"
            elif ph == 'confirmed_entry':
                p_icon = "[green]✓[/green]" if pr_up else "[red]✗[/red]"
                phase_txt += f"  {p_icon} Price>{p0+0.5:.1f} (LTP={cl:.1f})  Vol=waiting"

            pend_panel = Panel(Text.from_markup(phase_txt), border_style=ph_col, box=box.ROUNDED)
            with self._events_lock: evts = list(self.events)[:3]
            events_txt = "\n".join(evts) if evts else "[dim]—[/dim]"
            layout["events"].update(Group(pend_panel, Text.from_markup(events_txt)))
        else:
            # v28 Innovation 3: Show direction leaning, block counter, last discard when idle
            idle_parts = []

            # Direction leaning from gate diagnostics
            if d:
                _g = d.get("gate_diagnostics", {})
                _ce_w = _g.get("ce_weight", 0)
                _pe_w = _g.get("pe_weight", 0)
                _wt_n = _g.get("wt_needed", 5.0)
                if _ce_w > 0 or _pe_w > 0:
                    _lean_dir = "CE" if _ce_w >= _pe_w else "PE"
                    _lean_w = max(_ce_w, _pe_w)
                    _lean_pct = min(100, int(_lean_w / _wt_n * 100))
                    _lean_col = "green" if _lean_dir == "CE" else "red"
                    idle_parts.append(f"  Leaning: [{_lean_col}]{_lean_dir} ({_lean_w:.1f}/{_wt_n:.1f}w = {_lean_pct}%)[/{_lean_col}]")

            # Block counter (top 3)
            if self._block_counts:
                top_blocks = sorted(self._block_counts.items(), key=lambda x: -x[1])[:3]
                block_str = "  ".join(f"{k}:{v}" for k, v in top_blocks)
                idle_parts.append(f"  [dim]Blocks: {block_str}[/dim]")

            # Last discard
            if self._last_discard:
                _ld = self._last_discard
                _age = time.monotonic() - _ld['ts']
                if _age < 600:  # only show if < 10 min old
                    idle_parts.append(f"  [yellow]Last discard: {_ld['dir']} — {_ld['reason']} ({_age:.0f}s ago)[/yellow]")

            with self._events_lock: evts = list(self.events)[:4]
            all_parts = idle_parts + ["\n".join(evts) if evts else "[dim]—[/dim]"]
            layout["events"].update(Panel(Text.from_markup("\n".join(all_parts)), title="📢 EVENTS", border_style="dim"))

        return layout

# =============================================================================
# MARKET HOURS
# =============================================================================
def in_market_hours() -> Tuple[bool, str]:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False, "Weekend"
    o = now.replace(hour=CFG.MARKET_OPEN_H,  minute=CFG.MARKET_OPEN_M,  second=0, microsecond=0)
    c = now.replace(hour=CFG.MARKET_CLOSE_H, minute=CFG.MARKET_CLOSE_M, second=0, microsecond=0)
    if now < o:
        return False, "Pre-market"
    if now > c + timedelta(minutes=5):
        return False, "Closed"
    return True, "Open"

def in_safe_hours() -> bool:
    now = datetime.now(IST)
    o = now.replace(hour=CFG.MARKET_OPEN_H, minute=CFG.MARKET_OPEN_M + CFG.AVOID_FIRST_MINS, second=0)
    c = now.replace(hour=CFG.MARKET_CLOSE_H, minute=CFG.MARKET_CLOSE_M - CFG.AVOID_LAST_MINS, second=0)
    cutoff = now.replace(hour=CFG.NO_ENTRY_AFTER_H, minute=CFG.NO_ENTRY_AFTER_M, second=0)
    if not (o <= now <= c and now < cutoff):
        return False
    # v17.4: Dead zone removed — engine handles all regimes
    return True

# =============================================================================
# BOOTSTRAP
# =============================================================================
def bootstrap_prices(breeze: BreezeConnect, prices: PriceStore):
    try:
        r = breeze.get_quotes(
            stock_code=CFG.STOCK_CODE,
            exchange_code="NSE",
            product_type="cash",
        )
        if isinstance(r, dict) and r.get("Status") == 200:
            items = r.get("Success") or []
            if items:
                item = items[0] if isinstance(items, list) else items
                ltp = float(item.get("ltp") or 0)
                if ltp > 0:
                    prices.spot = ltp
                    log.info("Bootstrap spot: %.2f", ltp)
        else:
            log.warning("Bootstrap spot response: %s", r)
    except Exception as e:
        log.warning("bootstrap_prices spot: %s", e)

    futures_loaded = False
    try:
        r = breeze.get_quotes(
            stock_code=CFG.STOCK_CODE,
            exchange_code=CFG.EXCHANGE,
            product_type="futures",
            expiry_date=CFG.futures_rest_expiry(),
        )
        if isinstance(r, dict) and r.get("Status") == 200:
            items = r.get("Success") or []
            if items:
                item = items[0] if isinstance(items, list) else items
                ltp = float(item.get("ltp") or 0)
                if ltp > 0:
                    prices.futures = ltp
                    futures_loaded = True
                    log.info("Bootstrap futures (get_quotes): %.2f", ltp)
        else:
            log.warning("Bootstrap futures get_quotes response: %s",
                        str(r)[:200] if r else "None")
    except Exception as e:
        log.warning("bootstrap_prices futures get_quotes: %s", e)

    if not futures_loaded:
        try:
            r = breeze.get_option_chain_quotes(
                stock_code=CFG.STOCK_CODE,
                exchange_code=CFG.EXCHANGE,
                product_type="futures",
                expiry_date=CFG.futures_rest_expiry(),
            )
            if isinstance(r, dict) and r.get("Status") == 200:
                items = r.get("Success") or []
                if items:
                    item = items[0] if isinstance(items, list) else items
                    ltp = float(item.get("ltp") or 0)
                    if ltp > 0:
                        prices.futures = ltp
                        futures_loaded = True
                        log.info("Bootstrap futures (option_chain): %.2f", ltp)
            else:
                log.warning("Bootstrap futures option_chain response: %s",
                            str(r)[:200] if r else "None")
        except Exception as e:
            log.warning("bootstrap_prices futures option_chain: %s", e)

    if not futures_loaded and prices.spot > 0:
        prices.futures = prices.spot
        log.info("Bootstrap futures: using spot %.2f as proxy", prices.spot)

def fetch_vix(breeze: BreezeConnect, prices: PriceStore):
    """v25-LIVE: Fetch real INDIAVIX from REST API. Called at bootstrap + every OI refresh.
    Confirmed: stock_code='INDVIX', exchange_code='NSE', product_type='cash'.
    Bug fix: Breeze sometimes returns VIX as a decimal fraction (e.g. 0.25 instead of 25).
    We apply a sanity clamp: valid India VIX is always between 5 and 90. Values outside
    that range are corrected (×100 if too small) or discarded (if implausible)."""
    try:
        r = breeze.get_quotes(stock_code="INDVIX", exchange_code="NSE", product_type="cash")
        if isinstance(r, dict) and r.get("Status") == 200:
            items = r.get("Success") or []
            if items:
                item = items[0] if isinstance(items, list) else items
                ltp = float(item.get("ltp") or 0)
                if ltp > 0:
                    # Sanity: India VIX is always 5–90. Breeze occasionally
                    # returns a value 100× too small (decimal-shift bug).
                    if ltp < 5.0:
                        ltp = ltp * 100.0
                    if 5.0 <= ltp <= 90.0:
                        prices.real_vix = ltp
                        log.info("VIX fetch: %.2f", ltp)
                    else:
                        log.warning("VIX fetch: implausible value %.2f — ignored, keeping last=%.2f",
                                    ltp, prices.real_vix)
                    return
        log.warning("VIX fetch response: %s", r)
    except Exception as e:
        log.warning("VIX fetch failed: %s", e)

def bootstrap_oi(breeze: BreezeConnect, prices: PriceStore):
    print("  Bootstrapping OI from REST…")
    for right in ("call", "put"):
        try:
            r = breeze.get_option_chain_quotes(
                stock_code=CFG.STOCK_CODE, exchange_code=CFG.EXCHANGE,
                product_type="options", expiry_date=CFG.rest_expiry(),
                right=right,
            )
            if isinstance(r, dict) and r.get("Status") == 200:
                for item in (r.get("Success") or []):
                    try:
                        s   = float(item.get("strike_price") or 0)
                        ltp = float(item.get("ltp")          or 0)
                        oi  = float(item.get("open_interest") or 0)
                        rr  = str(item.get("right", "")).strip().lower()
                        if s > 0 and rr in ("call", "put"):
                            prices._write_option(s, rr, ltp, oi)
                    except Exception:
                        pass
        except Exception as e:
            log.warning("bootstrap_oi %s: %s", right, e)
    print(f"  Bootstrap done: {len(prices.opt_ltp)} option prices loaded")

# =============================================================================
# TRADING THREAD
# =============================================================================
def trading_thread(trader: PaperTrader, brain: BrainBridge,
                   dash: Dashboard, running: threading.Event):
    _last_acted_ts = -1.0
    _current_signal_ts = -1.0
    _current_signal_eligible = False
    _consecutive_ticks = 0
    _sig_window: deque = deque(maxlen=CFG.SIGNAL_PERSIST_WINDOW)
    _current_direction = "NEUTRAL"
    _prem = PremTracker()          # v16.1: premium momentum gate (ported from backtest)
    _last_day = ""                 # reset PremTracker on new day
    # v23: Impulse confirmation state
    _pending_signal = None         # dict with dir, sig_ltp, sig_time, decision info
    _pending_start = 0.0           # monotonic time when pending was created

    while running.is_set():
        try:
            # v25-LIVE: sync pending visibility to dashboard at top of every tick
            dash.pending_info = None  # cleared every tick, re-set below if still pending
            fired = _NEW_TICK_EVENT.wait(timeout=0.1)
            if fired:
                _NEW_TICK_EVENT.clear()

            # Update PremTracker every tick and reset on new day
            current_decision = brain.decision_q.peek()
            if current_decision:
                _today = datetime.now(IST).strftime("%Y-%m-%d")
                if _today != _last_day:
                    _prem.reset()
                    _last_day = _today
                _prem.update(
                    current_decision.get("ce_ltp", 0.0),
                    current_decision.get("pe_ltp", 0.0),
                )
            exits = trader.check_exits(brain_decision=current_decision)
            for t, reason in exits:
                is_win = (t.pnl_pts or 0) >= 0
                em = "✅" if is_win else "❌"
                play_sound("win" if is_win else "loss")
                dash.event(f"{em} #{t.id} {t.direction} {t.pnl_pts:+.1f}pts [{reason}]")
                try:
                    entry_dt = datetime.strptime(t.entry_time, "%H:%M:%S")
                    brain.send_result({
                        "pnl_pts":     t.pnl_pts or 0.0,
                        "direction":   t.direction,
                        "exit_reason": reason,
                        "entry_hour":  entry_dt.hour,
                        "entry_minute": entry_dt.minute,
                    })
                except Exception:
                    pass

            if not in_safe_hours():
                if _pending_signal:
                    log.info("TradingThread: DISCARD %s — outside safe hours", _pending_signal.get('dir', '?'))
                _sig_window.clear()
                _current_signal_ts = -1.0
                _pending_signal = None
                continue

            if len(trader.snapshot_open()) >= CFG.MAX_OPEN:
                if _pending_signal:
                    log.info("TradingThread: DISCARD %s — max open positions reached", _pending_signal.get('dir', '?'))
                _sig_window.clear()
                _current_signal_ts = -1.0
                _pending_signal = None
                continue

            # v24: CORRECT THREE-PHASE RETEST CONFIRMATION
            if _pending_signal and len(trader.snapshot_open()) == 0:
                elapsed = time.monotonic() - _pending_start
                pd = _pending_signal
                cur_decision = brain.decision_q.peek()
                if cur_decision:
                    cur_ltp = cur_decision.get("ce_ltp" if pd['dir'] == 'CE' else "pe_ltp", 0.0)
                    vr_sup = pd.get('decision', {}).get('vix_regime', 'MID')
                    # v26: percentage-based support floor — scales with premium level
                    _buf_pct = CFG.RETEST_SUPPORT_BUF_PCT.get(vr_sup, 0.025)
                    _buf_min = CFG.RETEST_SUPPORT_BUF_MIN.get(vr_sup, 2.0)
                    support = pd['sig_ltp'] - max(pd['sig_ltp'] * _buf_pct, _buf_min)

                    # ALWAYS CHECK: support breach → discard
                    if cur_ltp < support:
                        log.info("TradingThread: SUPPORT BREACHED %s @%.1f (support=%.1f) — discarded",
                                 pd['dir'], cur_ltp, support)
                        dash.record_discard(pd['dir'], f"Support breach @{cur_ltp:.1f} < {support:.1f}")
                        _pending_signal = None
                        continue

                    # Determine window: strong impulse (Phase 1 move >= threshold) gets more time
                    impulse_pts = pd.get('high', pd['sig_ltp']) - pd['sig_ltp']
                    retest_window = (
                        CFG.RETEST_WINDOW_STRONG_IMPULSE
                        if impulse_pts >= CFG.RETEST_STRONG_IMPULSE_PTS
                        else CFG.RETEST_WINDOW_SECS
                    )
                    if elapsed >= retest_window:
                        log.info("TradingThread: Retest window expired for %s after %.1fs — discarded",
                                 pd['dir'], elapsed)
                        dash.record_discard(pd['dir'], f"Window expired ({elapsed:.0f}s)")
                        _pending_signal = None
                        continue

                    # Track high watermark
                    if cur_ltp > pd.get('high', pd['sig_ltp']):
                        pd['high'] = cur_ltp

                    # Track recent prices for "price moving up" check
                    if 'recent_prices' not in pd:
                        pd['recent_prices'] = deque(maxlen=20)
                    pd['recent_prices'].append(cur_ltp)

                    phase = pd.get('phase', 'first_move')

                    if phase == 'first_move':
                        # Phase 1: first move proves buying interest
                        if pd['high'] - pd['sig_ltp'] >= pd['first_move_req']:
                            pd['phase'] = 'retest'
                            log.info("TradingThread: %s Phase 1 done — high=%.1f (+%.1f from P0=%.1f), waiting retest",
                                     pd['dir'], pd['high'], pd['high'] - pd['sig_ltp'], pd['sig_ltp'])

                    elif phase == 'retest':
                        # Phase 2: price returns NEAR P0 (signal price)
                        # v28: Dynamic buffer = percentage of premium, clamped [2.0, 8.0]
                        _vr_retest = pd.get('decision', {}).get('vix_regime', 'MID')
                        _ret_pct = CFG.RETEST_RETURN_PCT.get(_vr_retest, 0.06)
                        _ret_buf = max(CFG.RETEST_RETURN_MIN,
                                       min(CFG.RETEST_RETURN_MAX, pd['sig_ltp'] * _ret_pct))
                        near_p0 = abs(cur_ltp - pd['sig_ltp']) <= _ret_buf
                        if near_p0:
                            pd['phase'] = 'confirmed_entry'
                            log.info("TradingThread: %s Phase 2 done — price %.1f returned near P0=%.1f (buf=%.1f), waiting confirmation",
                                     pd['dir'], cur_ltp, pd['sig_ltp'], _ret_buf)

                    elif phase == 'confirmed_entry':
                        # v25-LIVE: Phase 3 SIMPLIFIED — engine already approved at persistence,
                        # market proved thesis (Phase 1 impulse + Phase 2 retest).
                        # Demanding engine re-approval here killed every live signal because
                        # by the time Phase 3 fires, SV/IMS/conviction have shifted.
                        # Just confirm the bounce: price up + volume OK.
                        #
                        # Engine direction check (soft): if engine now says OPPOSITE, abort.
                        # But don't require full entry_allowed — that's the gate that kills us.
                        cur_dir = cur_decision.get("entry_direction", "NEUTRAL")
                        if cur_dir != "NEUTRAL" and cur_dir != pd['dir']:
                            # Engine flipped direction — thesis is dead, discard
                            log.info("TradingThread: %s Phase 3 — engine flipped to %s, discarding",
                                     pd['dir'], cur_dir)
                            dash.record_discard(pd['dir'], f"Engine flipped to {cur_dir}")
                            _pending_signal = None
                            continue

                        # 1. Price moving up: above signal price (bounced off retest)
                        price_up = cur_ltp > pd['sig_ltp'] + 0.5

                        # 2. Volume not declining
                        vol_key = "atm_ce_vol" if pd['dir'] == 'CE' else "atm_pe_vol"
                        cur_vol = cur_decision.get(vol_key, 0)
                        if 'vol_history' not in pd:
                            pd['vol_history'] = deque(maxlen=500)
                        pd['vol_history'].append(cur_vol)
                        vh = list(pd['vol_history'])
                        if len(vh) >= 100:
                            recent_avg = sum(vh[-50:]) / 50
                            prev_avg = sum(vh[:-50]) / len(vh[:-50])
                            vol_increasing = prev_avg <= 0 or recent_avg >= prev_avg * 0.9
                        else:
                            vol_increasing = True  # not enough data, allow

                        if price_up and vol_increasing:
                            # Bug fix (Trade #2): verify the confirmation LTP is still fresh.
                            # Between Phase 3 firing and the actual enter() call, the market
                            # can gap. If the current LTP is already >SL_MIN pts BELOW the
                            # confirmation price, the SL will be hit on the very first tick.
                            # Abort instead of entering a trade that is already dead.
                            right_key = "ce_ltp" if pd['dir'] == 'CE' else "pe_ltp"
                            latest_ltp = cur_decision.get(right_key, cur_ltp)
                            stale_gap = pd['sig_ltp'] - latest_ltp   # positive = price fell
                            if stale_gap >= CFG.SL_MIN * 0.8:
                                log.warning(
                                    "TradingThread: %s STALE PRICE at entry — confirmation @%.1f "
                                    "but latest LTP=%.1f (gap=%.1f >= %.1f) — ABORTED",
                                    pd['dir'], pd['sig_ltp'], latest_ltp, stale_gap, CFG.SL_MIN * 0.8)
                                dash.record_discard(pd['dir'], f"Stale price gap {stale_gap:.1f}")
                                _pending_signal = None
                                continue

                            log.info("TradingThread: %s CONFIRMED ENTRY — engine agrees, price up, vol increasing @%.1f",
                                     pd['dir'], cur_ltp)
                            trade = trader.enter(pd['dir'], pd['conv'], pd['votes'], pd['smart'],
                                                 suggested_sl=pd['suggested_sl'], brain_decision=pd['decision'])
                            if trade:
                                play_sound("entry")
                                dash.event(
                                    f"🚀 #{trade.id} NIFTY {trade.strike:.0f} {pd['dir']} "
                                    f"@₹{trade.entry_price:.1f}  "
                                    f"Conv:{pd['conv']:.0%} "
                                    f"SL:{trade.sl_pts:.1f} [retest confirmed]"
                                )
                                _last_acted_ts = pd.get('decision_ts', -1.0)
                                _sig_window.clear()
                                _current_signal_ts = -1.0
                            _pending_signal = None
                            continue

                # v25-LIVE: Push pending state to dashboard for visibility
                if _pending_signal:
                    pd2 = _pending_signal
                    _elapsed = time.monotonic() - _pending_start
                    _cur_d = brain.decision_q.peek()
                    _cur_ltp2 = _cur_d.get("ce_ltp" if pd2['dir'] == 'CE' else "pe_ltp", 0.0) if _cur_d else 0.0
                    _imp_pts2 = pd2.get('high', pd2['sig_ltp']) - pd2['sig_ltp']
                    _retest_win = (
                        CFG.RETEST_WINDOW_STRONG_IMPULSE
                        if _imp_pts2 >= CFG.RETEST_STRONG_IMPULSE_PTS
                        else CFG.RETEST_WINDOW_SECS
                    )
                    dash.pending_info = {
                        'dir': pd2['dir'], 'phase': pd2.get('phase', 'first_move'),
                        'sig_ltp': pd2['sig_ltp'], 'cur_ltp': _cur_ltp2,
                        'high': pd2.get('high', pd2['sig_ltp']),
                        'first_move_req': pd2.get('first_move_req', 4.0),
                        'elapsed': _elapsed, 'window': _retest_win,
                        'price_up': _cur_ltp2 > pd2['sig_ltp'] + 0.5,
                        'vix_regime': pd2.get('decision', {}).get('vix_regime', 'MID'),
                    }
                else:
                    dash.pending_info = None
                continue  # while pending, skip signal detection

            decision = brain.decision_q.peek()
            if not decision:
                continue

            decision_ts = decision.get("ts", 0.0)

            if decision_ts == _last_acted_ts:
                continue

            if decision_ts != _current_signal_ts:
                _current_signal_ts = decision_ts

                direction = decision.get("entry_direction", "NEUTRAL")
                conviction = decision.get("entry_conviction", 0.0)
                votes = decision.get("votes_for", 0)

                sig_tick = None  # what this tick votes

                # v28: Record blocks for dashboard counter
                if not decision.get("entry_allowed"):
                    _br = decision.get("blocked_reason", "")
                    if _br:
                        dash.record_block(_br)

                if (decision.get("entry_allowed") and
                    conviction >= CFG.CONVICTION_MIN):  # v22: REMOVED 0.80 cap — stop blocking best signals

                    # v19-LIVE: REMOVED redundant PREM_VEL opposition check
                    # The engine already handles this as a soft penalty (-0.06).
                    # Having a HARD BLOCK here killed fast entry path entries.
                    # Also removed premium rising/fading hard blocks — engine handles
                    # premium quality via PREM_VEL signal score and PremiumDivergenceFilter.
                    # These 3 hard blocks were the #1 reason live entries were killed
                    # AFTER the engine approved them.

                    # v19-LIVE: Check entry_thesis for fast entry — skip prem checks entirely
                    entry_thesis = decision.get("entry_thesis", {})
                    is_fast_entry = entry_thesis.get("entry_mode") == "FAST_LEADING"

                    if is_fast_entry:
                        # v23: FAST ENTRY also goes through retest confirmation
                        # No more bypassing — the philosophy applies to ALL entries
                        trade_dir = direction
                        votes = decision.get("votes_for", 0)
                        smart = decision.get("smart_money", "")
                        suggested_sl = decision.get("suggested_sl", CFG.SL_MIN)
                        sig_ltp = decision.get("ce_ltp" if trade_dir == 'CE' else "pe_ltp", 0.0)
                        vr = decision.get("vix_regime", "MID")
                        first_move_req = CFG.RETEST_FIRST_MOVE.get(vr, 5.0)

                        log.info("TradingThread: FAST ENTRY → retest pending %s @%.1f (first_move=%.1f)",
                                 trade_dir, sig_ltp, first_move_req)

                        # v28: Slim decision — only store fields needed by enter() and retest logic
                        _slim_decision = {
                            'vix_regime': decision.get('vix_regime', 'MID'),
                            'probability': decision.get('probability', 0.0),
                        }
                        _pending_signal = {
                            'dir': trade_dir, 'sig_ltp': sig_ltp,
                            'conv': conviction, 'votes': votes, 'smart': smart,
                            'suggested_sl': suggested_sl, 'decision': _slim_decision,
                            'decision_ts': decision_ts,
                            'phase': 'first_move', 'high': sig_ltp,
                            'first_move_req': first_move_req,
                        }
                        _pending_start = time.monotonic()
                        _sig_window.clear()
                        _current_signal_ts = -1.0
                    else:
                        # Regular entry: only block on premium FADING (the worst signal)
                        passes_prem_fading = _prem.not_fading(direction, n=10, mx=0.03)
                        if passes_prem_fading:
                            sig_tick = direction
                        else:
                            log.debug("PremTracker: %s premium FADING — entry blocked", direction)

                _sig_window.append(sig_tick)

                # v17.1: sliding window persistence — need PERSIST out of last WINDOW ticks
                ce_count = sum(1 for s in _sig_window if s == 'CE')
                pe_count = sum(1 for s in _sig_window if s == 'PE')

                if ce_count >= CFG.SIGNAL_PERSISTENCE_TICKS or pe_count >= CFG.SIGNAL_PERSISTENCE_TICKS:
                    trade_dir = 'CE' if ce_count >= pe_count else 'PE'
                    conviction = decision.get("entry_conviction", 0.0)
                    votes = decision.get("votes_for", 0)
                    smart = decision.get("smart_money", "")
                    suggested_sl = decision.get("suggested_sl", CFG.SL_MIN)

                    # v23: Don't enter immediately — create pending, wait for impulse
                    sig_ltp = decision.get("ce_ltp" if trade_dir == 'CE' else "pe_ltp", 0.0)
                    log.info("TradingThread: Signal Persisted (%d/%d window). PENDING impulse confirm %s @%.1f",
                             max(ce_count, pe_count), CFG.SIGNAL_PERSIST_WINDOW, trade_dir, sig_ltp)

                    # v23: VIX-dependent first move threshold
                    vr = decision.get("vix_regime", "MID")
                    first_move_req = CFG.RETEST_FIRST_MOVE.get(vr, 5.0)
                    # v28: Slim decision — only store fields needed by enter() and retest logic
                    _slim_decision = {
                        'vix_regime': decision.get('vix_regime', 'MID'),
                        'probability': decision.get('probability', 0.0),
                    }
                    _pending_signal = {
                        'dir': trade_dir, 'sig_ltp': sig_ltp,
                        'conv': conviction, 'votes': votes, 'smart': smart,
                        'suggested_sl': suggested_sl, 'decision': _slim_decision,
                        'decision_ts': decision_ts,
                        'phase': 'first_move', 'high': sig_ltp,
                        'first_move_req': first_move_req,
                    }
                    _pending_start = time.monotonic()
                    _sig_window.clear()
                    _current_signal_ts = -1.0

        except Exception as e:
            log.error("TradingThread error: %s", e, exc_info=True)
            time.sleep(0.1)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n  🦅 HAWK TRADER v17 (sniper entries, strong holds) — launching dashboard…\n")

    try:
        breeze = BreezeConnect(api_key=_login.api_key)
        breeze.generate_session(api_secret=_login.api_secret,
                                session_token=_login.session_key)
    except Exception as e:
        print(f"  [FATAL] Session creation failed: {e}")
        print(f"  Check: 1) login.py credentials  2) session_key freshness  3) API key validity")
        sys.exit(1)

    console = Console() if RICH else None
    dash    = Dashboard()
    running = threading.Event()
    running.set()

    prices = PriceStore(spot_name_fragment=CFG.SPOT_FRAGMENT,
                        strike_step=CFG.STRIKE_STEP)

    try:
        bootstrap_prices(breeze, prices)
        fetch_vix(breeze, prices)
        dash.event(f"Prices loaded: Spot={prices.spot:.2f} Fut={prices.futures:.2f} "
                   f"VIX={prices.real_vix:.1f} Options={len(prices.opt_ltp)}")
    except Exception as e:
        dash.event(f"Price bootstrap warning: {e}")

    _pre_trader = PaperTrader(prices)
    dash._p = prices
    dash._trader = _pre_trader

    PRE_CONNECT_MINS = 15

    def next_connect_time(now: datetime) -> datetime:
        today_connect = now.replace(
            hour=CFG.MARKET_OPEN_H,
            minute=max(CFG.MARKET_OPEN_M - PRE_CONNECT_MINS, 0),
            second=0, microsecond=0)
        today_close = now.replace(
            hour=CFG.MARKET_CLOSE_H,
            minute=CFG.MARKET_CLOSE_M,
            second=0, microsecond=0) + timedelta(minutes=5)

        if now.weekday() >= 5:
            target = today_connect
            while target.weekday() >= 5:
                target += timedelta(days=1)
            return target

        if now < today_connect:
            return today_connect

        if now <= today_close:
            return now

        target = today_connect + timedelta(days=1)
        while target.weekday() >= 5:
            target += timedelta(days=1)
        return target

    now_ist = datetime.now(IST)
    connect_at = next_connect_time(now_ist)

    connect_is_now = (connect_at - now_ist).total_seconds() < 5
    if connect_is_now:
        dash.connect_target = "NOW"
    else:
        dash.connect_target = connect_at.strftime("%A %d-%b %H:%M IST")
    dash.connect_at_dt = connect_at

    _setup_done = threading.Event()
    _setup_error = [None]
    _components = {}

    def setup_thread():
        nonlocal connect_at
        try:
            now = datetime.now(IST)
            connect_at = next_connect_time(now)
            wait_needed = (connect_at - now).total_seconds() > 5

            if wait_needed:
                _, market_reason = in_market_hours()
                dash.phase = Dashboard.PHASE_WAITING
                dash.connect_target = connect_at.strftime("%A %d-%b %H:%M IST")
                dash.event(f"Market is {market_reason} — waiting for {dash.connect_target}")

                while running.is_set():
                    now = datetime.now(IST)
                    target = next_connect_time(now)
                    remaining = (target - now).total_seconds()
                    if remaining <= 0:
                        break
                    dash.connect_target = target.strftime("%A %d-%b %H:%M IST")
                    dash.connect_at_dt = target
                    time.sleep(1.0)

                if not running.is_set():
                    return

            dash.phase = Dashboard.PHASE_CONNECTING
            dash.phase_detail = "Connecting to ICICI WebSocket…"
            dash.event("Connecting WebSocket…")

            feed = FeedManager(breeze, prices,
                               expiry_ws=CFG.ws_expiry(),
                               stock_code=CFG.STOCK_CODE,
                               futures_expiry_ws=CFG.futures_ws_expiry())

            try:
                feed.connect(max_retries=5, base_wait=3.0)
            except ConnectionError as e:
                _setup_error[0] = str(e)
                dash.event(f"⚠ Connection failed: {e}")
                dash.phase_detail = f"FAILED: {e}"
                return

            dash.phase_detail = "Subscribing spot + futures…"
            dash.event("WebSocket connected")
            try:
                feed.subscribe_spot()
                feed.subscribe_futures(expiry_ws=CFG.futures_ws_expiry())
            except Exception as e:
                dash.event(f"Subscription warning: {e}")

            dash.phase_detail = "Waiting for first tick…"
            _NEW_TICK_EVENT.wait(timeout=10.0)
            _NEW_TICK_EVENT.clear()

            if prices.spot <= 0:
                dash.phase_detail = "Bootstrapping OI from REST…"
            bootstrap_oi(breeze, prices)

            if prices.spot <= 0:
                _setup_error[0] = "No spot price available after bootstrap"
                dash.event("⚠ No spot price — check feed")
                dash.phase_detail = "FAILED: no spot price"
                return

            atm = prices.atm
            dash.phase_detail = f"Subscribing {CFG.NUM_STRIKES*2+2} option strikes…"
            feed.subscribe_strikes(atm, n=CFG.NUM_STRIKES, step=CFG.STRIKE_STEP)
            time.sleep(3.0)

            dash.phase_detail = "Starting brain subprocess…"
            trader = PaperTrader(prices)
            brain  = BrainBridge()
            brain.start()

            _components['prices'] = prices
            _components['feed']   = feed
            _components['trader'] = trader
            _components['brain']  = brain

            dash.set_components(prices, trader, brain)
            dash.phase = Dashboard.PHASE_LIVE
            dash.event(f"✓ LIVE — Spot:{prices.spot:.2f} ATM:{atm:.0f} "
                       f"Options:{len(prices.opt_ltp)} Brain:pid={brain._process.pid}")

            _snap_count = [0]
            def snap_feeder():
                _last_hb = time.monotonic()
                while running.is_set():
                    try:
                        snap = build_snapshot(prices)
                        if snap:
                            try:
                                brain.snap_mp_q.put_nowait(snap)
                            except Exception:
                                try:
                                    brain.snap_mp_q.get_nowait()
                                    brain.snap_mp_q.put_nowait(snap)
                                except Exception:
                                    pass
                            _snap_count[0] += 1
                            if _snap_count[0] == 1:
                                log.info("FIRST snap sent spot=%.2f", snap.get("spot",0))
                            now = time.monotonic()
                            if now - _last_hb > 60:
                                log.info("SnapFeeder alive snaps=%d", _snap_count[0])
                                _last_hb = now
                    except Exception as e:
                        log.warning("snap_feeder: %s", e)
                    time.sleep(CFG.SNAP_INTERVAL)

            def status_loop():
                while running.is_set():
                    try:
                        _NEW_TICK_EVENT.wait(timeout=0.3)
                        market_ok, market_reason = in_market_hours()
                        d = brain.decision_q.peek()
                        feed.reconnect_if_needed()
                        if not market_ok:
                            dash.status = f"{market_reason} | Ticks:{prices.tick_count}"
                        elif d:
                            direction = str(d.get("entry_direction", "?"))
                            score = float(d.get("entry_conviction", 0.0))
                            votes = int(d.get("votes_for", 0))
                            vix_r = str(d.get("vix_regime", "?"))
                            dash.status = (
                                f"Ticks:{prices.tick_count} Snaps:{_snap_count[0]} "
                                f"Brain:{direction}({score:.0%},{votes}/18) "
                                f"VIX:{vix_r} "
                                f"{'✓' if brain.is_alive() else '✗'}")
                        else:
                            dash.status = (
                                f"Ticks:{prices.tick_count} Snaps:{_snap_count[0]} "
                                f"Brain:warming… {'✓' if brain.is_alive() else '✗'}")
                    except Exception as e:
                        log.error("StatusLoop: %s", e)
                        time.sleep(0.5)

            def management_loop():
                last_resub = 0.0
                last_brain_check = 0.0
                last_oi_refresh = time.monotonic()
                last_tick_count = 0
                last_tick_check = time.monotonic()
                OI_REFRESH_INTERVAL = 900.0
                while running.is_set():
                    try:
                        now = time.monotonic()
                        if now - last_resub > 60.0 and prices.spot > 0:
                            feed.subscribe_strikes(prices.atm, n=CFG.NUM_STRIKES,
                                                   step=CFG.STRIKE_STEP)
                            last_resub = now

                        if now - last_tick_check > 60.0:
                            current_ticks = prices.tick_count
                            if current_ticks == last_tick_count and current_ticks > 0:
                                market_ok, _ = in_market_hours()
                                if market_ok:
                                    log.warning("STALE FEED: no new ticks for 60s (count=%d) — forcing reconnect", current_ticks)
                                    dash.event("⚠ Stale feed detected — reconnecting WS")
                                    feed._reconnect_needed = True
                            last_tick_count = current_ticks
                            last_tick_check = now

                        if now - last_oi_refresh > OI_REFRESH_INTERVAL:
                            last_oi_refresh = now
                            try:
                                for right in ("call", "put"):
                                    r = breeze.get_option_chain_quotes(
                                        stock_code=CFG.STOCK_CODE,
                                        exchange_code=CFG.EXCHANGE,
                                        product_type="options",
                                        expiry_date=CFG.rest_expiry(),
                                        right=right)
                                    if isinstance(r, dict) and r.get("Status") == 200:
                                        for item in (r.get("Success") or []):
                                            try:
                                                s   = float(item.get("strike_price") or 0)
                                                oi  = float(item.get("open_interest") or 0)
                                                rr  = str(item.get("right","")).strip().lower()
                                                if s > 0 and oi > 0 and rr in ("call","put"):
                                                    prices.opt_oi[(s, rr)] = oi
                                            except Exception:
                                                pass
                                log.info("OI refresh: %d strikes updated",
                                         len(prices.opt_oi))
                                dash.event("OI chain refreshed")
                                # v25-LIVE: refresh real INDIAVIX alongside OI
                                fetch_vix(breeze, prices)
                            except Exception as oe:
                                log.warning("OI refresh failed: %s", oe)

                        if now - last_brain_check > 30.0:
                            last_brain_check = now
                            if not brain.is_alive():
                                log.critical("Brain died — restarting")
                                dash.event("⚠ Brain died — restarting")
                                try:
                                    brain.stop()
                                    brain._process = None
                                    brain._bridge = None
                                    brain.start()
                                    dash.event("✓ Brain restarted")
                                except Exception as re:
                                    log.critical("Brain restart failed: %s", re)
                    except Exception as e:
                        log.debug("Management: %s", e)
                    time.sleep(10.0)

            threading.Thread(target=snap_feeder, daemon=True, name="SnapFeeder").start()
            threading.Thread(target=status_loop, daemon=True, name="StatusLoop").start()
            threading.Thread(target=management_loop, daemon=True, name="Management").start()
            threading.Thread(target=trading_thread, args=(trader, brain, dash, running),
                             daemon=True, name="TradingThread").start()

            _setup_done.set()

        except Exception as e:
            _setup_error[0] = str(e)
            dash.event(f"⚠ Setup error: {e}")
            log.critical("Setup thread error: %s", e, exc_info=True)

    threading.Thread(target=setup_thread, daemon=True, name="Setup").start()

    try:
        if RICH and console:
            with Live(
                get_renderable=dash.renderable,
                screen=True,
                auto_refresh=True,
                refresh_per_second=4,
                console=console,
                vertical_overflow="visible",
            ):
                while running.is_set():
                    time.sleep(0.25)
        else:
            while running.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Shutting down…")
        running.clear()
        brain  = _components.get('brain')
        feed   = _components.get('feed')
        trader = _components.get('trader') or _pre_trader
        if brain:
            brain.stop()
        if feed:
            feed.disconnect()
        trader._save()
        cap_pct = ((trader.capital / CFG.STARTING_CAPITAL) - 1.0) * 100
        print(f"\n  Session complete")
        print(f"  Capital : ₹{trader.capital:,.2f}  ({cap_pct:+.2f}%)")
        print(f"  Trades  : {len(trader.closed)} ({trader.wins}W/{trader.losses}L)")
        print(f"  Win rate: {trader.win_rate:.0f}%")
        print()
        for t in trader.closed:
            pts = t.pnl_pts or 0
            pct = ((t.exit_price / t.entry_price) - 1.0) * 100 if (t.exit_price and t.entry_price) else 0.0
            print(f"  {'WIN' if pts>=0 else 'LOSS'} #{t.id} "
                  f"{t.direction} {pct:+.2f}% [{t.exit_reason}] "
                  f"{t.entry_time}→{t.exit_time}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
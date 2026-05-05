#!/usr/bin/env python3
"""
HAWK BRAIN v17.5 — Realistic data calibrated, tighter IMS/SV gates
"""
import os, sys, time, logging, logging.handlers, json
import multiprocessing as mp
from typing import Optional
from datetime import datetime, timezone, timedelta

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

IST = timezone(timedelta(hours=5, minutes=30))

# =============================================================================
# LOGGING
# =============================================================================
def _setup_logging() -> logging.Logger:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "hawk_brain.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt); ch.setLevel(logging.WARNING)
    log = logging.getLogger("hawk.brain")
    log.setLevel(logging.DEBUG); log.addHandler(fh); log.addHandler(ch)
    return log

# =============================================================================
# BRAIN PROCESS ENTRY POINT
# =============================================================================
def run_brain(snap_queue: mp.Queue, decision_queue: mp.Queue, result_queue: mp.Queue):
    log = _setup_logging()
    log.info("HawkBrain v17 started (pid=%d)", os.getpid())
    try:
        from hawk_engine import NiftyEngine
        engine = NiftyEngine()
    except ImportError as e:
        log.critical("HawkBrain: cannot import hawk_engine.py — %s", e)
        sys.exit(1)

    learning: dict = _load_learning()
    last_learning_save = time.monotonic()
    last_reset_date: Optional[str] = None
    session_reset_done = False
    tick = 0; last_hb = time.monotonic(); last_entry_signals: dict = {}

    # v28 Innovation D: Health counters — track silent failures for debugging.
    # Without these, a bug could silently kill the brain's decision output
    # while the dashboard keeps showing stale data (the zero-trade root cause).
    _health = {"loop_errors": 0, "decision_drops": 0, "result_errors": 0,
               "signal_parse_errors": 0, "last_error": ""}
    _last_health_log = time.monotonic()

    # v28 Innovation D: Import once at function level, not inside the loop
    import queue as _queue_mod

    while True:
        try:
            now_ist = datetime.now(IST); today_str = now_ist.strftime("%Y-%m-%d")
            if today_str != last_reset_date:
                if now_ist.hour == 9 and now_ist.minute >= 15:
                    engine.reset_session(); last_reset_date = today_str; session_reset_done = True
                    tick = 0; last_entry_signals = {}
                elif now_ist.hour > 9:
                    engine.reset_session(); last_reset_date = today_str; session_reset_done = True
                    tick = 0; last_entry_signals = {}

            pre_market = (now_ist.hour < 9 or (now_ist.hour == 9 and now_ist.minute < 15))

            # v28 Innovation D: Separate queue.Empty from real errors.
            # Before this fix, ANY exception (including _record_result crashes)
            # was silently swallowed — a bug in learning code would never surface.
            try:
                while True:
                    res = result_queue.get_nowait()
                    if res is None:
                        _save_learning(learning); return
                    _record_result(res, learning, last_entry_signals)
                    _save_learning(learning)
                    exit_reason = res.get("exit_reason", "SL HIT")
                    exit_dir = res.get("direction", "NEUTRAL")
                    engine.reset_after_exit(exit_reason=exit_reason, direction=exit_dir)
            except _queue_mod.Empty:
                pass  # expected — queue drained
            except Exception as e:
                _health["result_errors"] += 1
                _health["last_error"] = f"result_q: {e}"
                log.warning("HawkBrain: result_queue processing error: %s", e)

            try:
                snap = snap_queue.get(timeout=1.0)
            except _queue_mod.Empty:
                if time.monotonic() - last_hb > 30: last_hb = time.monotonic()
                continue
            except Exception as e:
                log.warning("HawkBrain: snap_queue error: %s", e)
                continue

            if snap is None:
                _save_learning(learning); return

            if pre_market and not session_reset_done:
                pre_decision = {
                    "entry_allowed": False, "entry_direction": "NEUTRAL", "entry_conviction": 0.0,
                    "blocked_reason": f"PRE-MARKET", "spot": snap.get("spot", 0.0),
                    "futures": snap.get("futures", 0.0), "atm": snap.get("atm", 0.0),
                    "ce_ltp": snap.get("ce_ltp", 0.0), "pe_ltp": snap.get("pe_ltp", 0.0),
                    "pcr": snap.get("pcr", 1.0), "is_clean": snap.get("is_clean", False),
                    "votes_for": 0, "vote_detail": [], "smart_money": "IDLE",
                    "suggested_sl": 0.0, "suggested_tgt": 0.0, "engine_summary": "",
                    "reversal_warnings": [], "ts": snap.get("ts", time.monotonic()),
                    "vix_regime": "MID", "probability": 0.0, "entry_thesis": {},
                    "market_phase": "UNKNOWN",  # v28 Innovation A
                }
                try:
                    decision_queue.put_nowait(pre_decision)
                except _queue_mod.Full:
                    _health["decision_drops"] += 1  # v28 Innovation D: count, don't swallow silently
                tick += 1; continue

            snap_obj = _SnapshotAdapter(snap)
            total_options_oi = snap.get("total_ce_oi", 0.0) + snap.get("total_pe_oi", 0.0)
            signal_weights = _compute_signal_weights(learning)
            vix_value = snap.get("vix", 0.0)

            result = engine.update(snap_obj, snap.get("futures", 0.0),
                                   futures_oi=total_options_oi,
                                   signal_weights=signal_weights,
                                   vix_value=vix_value)

            _apply_learning_modifier(result, learning)
            decision = _build_decision(snap, result)
            # v22: Pass fut_vel_disp EVERY tick (not just entries) for FDC exit intelligence
            decision["fut_vel_disp"] = round(engine.fut_vel.displacement, 3) if hasattr(engine, 'fut_vel') else 0.0
            # v25-LIVE: Pass ALL gate diagnostics so dashboard shows everything at once
            decision["gate_diagnostics"] = engine.get_gate_diagnostics()

            if result.entry_allowed:
                last_entry_signals = {}
                for detail in result.vote_detail:
                    try:
                        name = detail.split(":")[0]
                        direction = detail.split(":")[1].split("(")[0]
                        last_entry_signals[name] = direction
                    except (IndexError, ValueError):
                        _health["signal_parse_errors"] += 1  # v28 Innovation D: count parse failures

            try:
                decision_queue.put_nowait(decision)
            except _queue_mod.Full:
                _health["decision_drops"] += 1  # v28 Innovation D: decision DROPPED — dashboard will show stale data

            tick += 1
            if time.monotonic() - last_learning_save > 300:
                _save_learning(learning); last_learning_save = time.monotonic()

            # v28 Innovation D: Log health counters every 5 minutes — surfaces silent failures
            if time.monotonic() - _last_health_log > 300:
                if any(v for k, v in _health.items() if k != "last_error" and v > 0):
                    log.warning("HawkBrain health: %s", _health)
                _last_health_log = time.monotonic()

            if time.monotonic() - last_hb > 60: last_hb = time.monotonic()

        except Exception as e:
            # v28 Innovation D: Log the ACTUAL error instead of silently sleeping.
            # Before this fix, ANY bug in the main loop (engine crash, dict key error,
            # type error) was caught, swallowed, and the brain continued producing
            # stale/no decisions — the root cause of silent zero-trade sessions.
            _health["loop_errors"] += 1
            _health["last_error"] = f"main_loop: {type(e).__name__}: {e}"
            log.error("HawkBrain main loop error #%d: %s: %s",
                      _health["loop_errors"], type(e).__name__, e)
            time.sleep(0.1)


# =============================================================================
# SNAPSHOT ADAPTER
# =============================================================================
class _SnapshotAdapter:
    __slots__ = ("_d",)
    def __init__(self, d: dict): self._d = d
    @property
    def spot(self) -> float: return self._d.get("spot", 0.0)
    @property
    def atm_strike(self) -> float: return self._d.get("atm", 0.0)
    @property
    def atm_ce_ltp(self) -> float: v = self._d.get("ce_ltp", 0.0); return v if v > 0 else float("nan")
    @property
    def atm_pe_ltp(self) -> float: v = self._d.get("pe_ltp", 0.0); return v if v > 0 else float("nan")
    @property
    def total_call_oi(self) -> float: return self._d.get("total_ce_oi", 0.0)
    @property
    def total_put_oi(self) -> float: return self._d.get("total_pe_oi", 0.0)
    @property
    def pcr(self) -> float: return self._d.get("pcr", 1.0)
    @property
    def is_clean(self) -> bool: return self._d.get("is_clean", False)
    @property
    def call_strikes(self) -> '_StrikesProxy': return _StrikesProxy(self._d.get("strikes", {}), "call")
    @property
    def put_strikes(self) -> '_StrikesProxy': return _StrikesProxy(self._d.get("strikes", {}), "put")

class _StrikesProxy:
    def __init__(self, strikes: dict, right: str):
        self._s = {}
        for key, v in strikes.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                s, r = key
                if r == right: self._s[float(s)] = v
    def __contains__(self, item): return float(item) in self._s
    def __iter__(self): return iter(self._s)
    def __bool__(self): return len(self._s) > 0
    def items(self): return [(s, _StrikeDataProxy(v)) for s, v in self._s.items()]
    def get(self, key, default=None):
        v = self._s.get(float(key))
        return _StrikeDataProxy(v) if v is not None else default

class _StrikeDataProxy:
    def __init__(self, d: dict):
        self.oi = d.get("oi", 0.0); self.ltp = d.get("ltp", 0.0); self.vol = d.get("vol", 0.0)
        self.bid = d.get("bid", 0.0); self.ask = d.get("ask", 0.0)

# =============================================================================
# DECISION BUILDER
# =============================================================================
def _build_decision(snap: dict, result) -> dict:
    return {
        "entry_allowed": result.entry_allowed, "entry_direction": result.direction,
        "entry_conviction": result.score, "blocked_reason": result.blocked_reason,
        "spot": snap.get("spot", 0.0), "futures": snap.get("futures", 0.0),
        "atm": snap.get("atm", 0.0), "ce_ltp": snap.get("ce_ltp", 0.0),
        "pe_ltp": snap.get("pe_ltp", 0.0), "pcr": snap.get("pcr", 1.0),
        "is_clean": snap.get("is_clean", False), "votes_for": result.votes_for,
        "vote_detail": result.vote_detail, "smart_money": result.smart_money_bias,
        "suggested_sl": result.suggested_sl_pts, "suggested_tgt": result.suggested_tgt_pts,
        "engine_summary": "", "reversal_warnings": result.reversal_warnings if hasattr(result, 'reversal_warnings') else [],
        "ts": snap.get("ts", time.monotonic()),
        "vix_regime": result.vix_regime if hasattr(result, 'vix_regime') else "MID",
        "probability": result.probability if hasattr(result, 'probability') else 0.0,
        "imm_momentum": result.imm_momentum if hasattr(result, 'imm_momentum') else 0.0,
        "entry_thesis": result.entry_thesis if hasattr(result, 'entry_thesis') else {},
        # v28 Innovation A: Market phase from state machine (ACCUMULATION/EXPANSION/TREND/EXHAUSTION/DISTRIBUTION)
        "market_phase": result.market_phase if hasattr(result, 'market_phase') else "UNKNOWN",
    }

# =============================================================================
# LEARNING (unchanged from v16)
# =============================================================================
_LEARNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hawk_learning.json")

def _load_learning() -> dict:
    """Load learning data from disk. Returns empty defaults on any error.
    v28 Innovation D: Log errors instead of silent swallow — corrupt learning
    data caused inflated signal weights in previous versions."""
    log = logging.getLogger("hawk.brain")
    try:
        if os.path.exists(_LEARNING_FILE):
            with open(_LEARNING_FILE) as f: data = json.load(f)
            data.setdefault("time_slots", {}); data.setdefault("total_trades", 0); data.setdefault("signal_accuracy", {})
            return data
    except Exception as e:
        log.warning("Failed to load learning file: %s — starting fresh", e)
    return {"time_slots": {}, "total_trades": 0, "signal_accuracy": {}}

def _save_learning(learning: dict):
    """Persist learning data. v28 Innovation D: Log save failures."""
    try:
        with open(_LEARNING_FILE, "w") as f: json.dump(learning, f, indent=2)
    except Exception as e:
        logging.getLogger("hawk.brain").warning("Failed to save learning: %s", e)

def _record_result(result: dict, learning: dict, last_entry_signals: dict):
    try:
        pnl = result.get("pnl_pts", 0.0) or 0.0
        hour = result.get("entry_hour", 0); minute = result.get("entry_minute", 0)
        slot = f"{hour:02d}:{(minute // 15) * 15:02d}"
        is_win = pnl >= 0
        slots = learning.setdefault("time_slots", {})
        s = slots.setdefault(slot, {"wins": 0, "losses": 0, "total_pnl": 0.0})
        if is_win: s["wins"] += 1
        else: s["losses"] += 1
        s["total_pnl"] += pnl; learning["total_trades"] = learning.get("total_trades", 0) + 1

        # v29: Track direction-flip entries separately — flips have structurally different
        # win rates than continuation entries (they fight residual momentum).
        # Stored under "flip_accuracy" so the learning modifier can apply a separate penalty
        # when the flip track record is poor, independently of signal accuracy.
        is_flip = result.get("is_dir_flip", False)
        if is_flip:
            flip_acc = learning.setdefault("flip_accuracy", {"wins": 0, "losses": 0, "total_pnl": 0.0})
            if is_win: flip_acc["wins"] += 1
            else:      flip_acc["losses"] += 1
            flip_acc["total_pnl"] = flip_acc.get("total_pnl", 0.0) + pnl

        sig_acc = learning.setdefault("signal_accuracy", {})
        direction = result.get("direction", "?")
        # v28 FIX: Only credit signals that voted IN the trade direction.
        # Old code credited ALL signals (even those voting opposite), inflating total
        # and making accuracy = correct/total meaningless.
        for sig_name, sig_dir in last_entry_signals.items():
            if sig_dir != direction:
                continue  # skip signals that voted opposite — they don't deserve credit/blame
            sa = sig_acc.setdefault(sig_name, {"wins": 0, "losses": 0, "correct": 0, "total": 0})
            sa["total"] += 1
            if is_win:
                sa["wins"] += 1
                sa["correct"] += 1  # voted in direction AND trade won
            else:
                sa["losses"] += 1
    except Exception as e:
        # v28 Innovation D: Log trade recording failures — corrupted learning data
        # was a silent source of wrong signal weights in production
        logging.getLogger("hawk.brain").warning("_record_result failed: %s", e)

def _compute_signal_weights(learning: dict) -> dict:
    # v29: Overhauled — previous logic gave 1.0 weight to everything below 60% accuracy,
    # meaning SUPERTREND (37.9% WR), MAX_PAIN (37.8%), FUT_VEL (38.6%), ORB (39.4%)
    # all got weight=1.0 despite being near-contra-indicators.
    # First-principles rule: break-even WR for this system = ~43% (given avg win/loss ratio).
    # Any signal below 43% WR should have weight < 1.0 and below 38% should be near-zero.
    # Signals above 55%+ WR are genuinely predictive and deserve 1.5× amplification.
    weights = {}; sig_acc = learning.get("signal_accuracy", {})
    for sig_name, s in sig_acc.items():
        t = s.get("total", 0)
        if t < 15: weights[sig_name] = 1.0; continue  # insufficient data — neutral
        acc = s.get("correct", 0) / t
        if acc >= 0.62:   weights[sig_name] = 1.8   # genuinely excellent (RSI=68.8%)
        elif acc >= 0.55: weights[sig_name] = 1.5   # clearly above break-even
        elif acc >= 0.50: weights[sig_name] = 1.2   # slightly above coin-flip
        elif acc >= 0.43: weights[sig_name] = 1.0   # at or above system break-even — neutral
        elif acc >= 0.38: weights[sig_name] = 0.4   # below break-even — penalise
        else:             weights[sig_name] = 0.2   # near-contra-indicator — minimal weight
    return weights

def _apply_learning_modifier(result, learning: dict):
    if not result.entry_allowed: return
    try:
        STALE_PENALTY = 0.02
        if result.vote_detail:
            for detail in result.vote_detail:
                try:
                    parts = detail.split(":")
                    direction = parts[1].split("(")[0]
                    if direction == result.direction and "[stale]" in detail:
                        result.score = max(result.score - STALE_PENALTY, 0.0)
                except (IndexError, ValueError, AttributeError):
                    pass  # v28 Innovation D: narrowed — only catch string parse failures

        now = datetime.now(IST)
        hour = now.hour; minute = (now.minute // 15) * 15; slot = f"{hour:02d}:{minute:02d}"
        penalty = 0.0; boost = 0.0; reasons = []
        slots = learning.get("time_slots", {})
        s = slots.get(slot)
        if s is not None:
            total = s["wins"] + s["losses"]
            # v29: lowered sample threshold 20→6 — slots like 11:00 (0W/8L) and 14:45 (0W/6L)
            # were never penalised because they had < 20 samples. They are demonstrably toxic.
            # At 6+ samples a 0% WR slot is not noise — it is a real dead zone.
            if total >= 6:
                wr = s["wins"] / total
                # v29: HARD BLOCK — if a slot has 0 wins and 5+ losses, skip it entirely.
                # First-principles: 0% WR over 6+ attempts is not bad luck, it is structure.
                # Known toxic slots: 11:00 (0W/8L), 14:45 (0W/6L), 12:00 (0W/4L), 11:15 (0W/4L)
                if s["wins"] == 0 and total >= 5:
                    result.entry_allowed = False
                    result.blocked_reason = f"TimeSlot HARD BLOCK: {slot} has 0 wins in {total} trades (WR=0%)"
                    return
                elif wr < 0.30: penalty += 0.12; reasons.append(f"TimeSlot {slot} WR={wr:.0%}")  # v29: raised from 0.05
                elif wr < 0.40: penalty += 0.07; reasons.append(f"TimeSlot {slot} WR={wr:.0%}")  # v29: raised from 0.03
                elif wr < 0.45: penalty += 0.04; reasons.append(f"TimeSlot {slot} WR={wr:.0%}")
                elif wr > 0.60: boost += 0.05; reasons.append(f"TimeSlot {slot} WR={wr:.0%} BOOST")

        # v29: Direction-flip penalty — if this is a flip entry and the historical
        # flip win rate is poor, add an extra conviction penalty on top of the
        # trader-level +0.06 bump already applied in hawk_trader.py enter().
        # This creates a two-layer flip gate: engine (ExhaustionGate) → trader (conv bump) → brain (learning penalty).
        # Only fires once >= 8 flip trades have been recorded (statistical minimum).
        flip_acc = learning.get("flip_accuracy", {})
        flip_total = flip_acc.get("wins", 0) + flip_acc.get("losses", 0)
        if flip_total >= 8 and getattr(result, "is_dir_flip", False):
            flip_wr = flip_acc.get("wins", 0) / flip_total
            if flip_wr < 0.35:
                penalty += 0.10; reasons.append(f"FlipWR={flip_wr:.0%}(poor)")
            elif flip_wr < 0.42:
                penalty += 0.06; reasons.append(f"FlipWR={flip_wr:.0%}(weak)")
            elif flip_wr > 0.55:
                boost += 0.04  # flip trades are actually working — ease the gate slightly

        sig_acc = learning.get("signal_accuracy", {})
        if sig_acc and result.vote_detail:
            accuracy_scores = []
            for detail in result.vote_detail:
                try:
                    name = detail.split(":")[0]; direction = detail.split(":")[1].split("(")[0]
                    if direction == result.direction:
                        sa = sig_acc.get(name)
                        if sa and sa["total"] >= 15:  # v29: lowered 20→15 — more signals qualify sooner
                            acc = sa["correct"] / sa["total"]; accuracy_scores.append(acc)
                except (IndexError, ValueError, AttributeError):
                    pass
            if len(accuracy_scores) >= 2:
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                # v29: raised penalties — weak 0.06 penalty was insufficient to block bad signal combos
                if avg_accuracy < 0.35: penalty += 0.12; reasons.append(f"Hist_Signal_Acc={avg_accuracy:.0%}")
                elif avg_accuracy < 0.40: penalty += 0.08; reasons.append(f"Hist_Signal_Acc={avg_accuracy:.0%}")
                elif avg_accuracy < 0.43: penalty += 0.05; reasons.append(f"Hist_Signal_Acc={avg_accuracy:.0%}")
                elif avg_accuracy > 0.60: boost += 0.06  # v29: raised boost for genuinely good signal combos

        old_score = result.score
        result.score = min(max(result.score - penalty + boost, 0.0), 0.99)
        # v28: conviction floors match engine's VIXTracker.conviction_floor()
        vix_regime = getattr(result, 'vix_regime', 'MID')
        _floors = {"LOW": 0.48, "MID": 0.50, "HIGH": 0.52, "EXTREME": 0.55}
        conv_floor = _floors.get(vix_regime, 0.50)
        if result.score < conv_floor:
            result.entry_allowed = False
            result.blocked_reason = f"Learning Filter: Score dropped {old_score:.2f} -> {result.score:.2f} due to: {', '.join(reasons)}"
    except Exception as e:
        # v28 Innovation D: Log learning modifier failures — a bug here silently
        # disables the entire learning system without any visible symptom
        logging.getLogger("hawk.brain").warning("_apply_learning_modifier failed: %s", e)

if __name__ == "__main__":
    pass
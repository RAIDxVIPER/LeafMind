"""
src/weather/rules.py
====================
Weather Disease-Risk Rule Engine (Rules W01–W08) and Summary Statistics (Stage C4).

Provenance and rules:
  - All thresholds are read from config/weather_rules.yaml; NOT hardcoded.
  - Sourced from spec docs/spec/part2_ml_knowledge.md §2c-v and part6_build_spec.md §6d-v.
  - All rule thresholds are tagged [UNVERIFIED — I14] and require plant pathologist validation.
  - W08 is a general agronomic heuristic [GAP] and is omitted from active weather scoring
    per spec Part 4 §4a-iv and Part 6 §6d-v.
  - Leaf wetness: Open-Meteo has no direct leaf-wetness sensor. Derived as hours with
    RH >= 90% and/or precipitation > 0.1mm, tagged [ENG proxy] in code and output.
  - Weather score: Computed per Part 4 formula:
      applicable_flags = count of disease-applicable rules in W01–W07 (W08 omitted [GAP])
      weather_score = count(triggered applicable) / count(total applicable) in [0.0, 1.0]
      When no disease provided, returns fraction of triggered rules across W01–W07.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from src.schemas.schemas import (
    RuleTriggerResult,
    WeatherRiskFlag,
    WeatherSummaryStats,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RULES_PATH = REPO_ROOT / "config" / "weather_rules.yaml"


class WeatherRuleEngine:
    """Evaluates disease-risk rules W01–W08 against weather timeseries data."""

    def __init__(self, rules_path: Optional[Path | str] = None):
        cfg_file = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                self.rules_config: List[Dict[str, Any]] = raw.get("rules", [])
        else:
            logger.warning("Rules file %s not found; using empty rules list", cfg_file)
            self.rules_config = []

    def _isolate_past_24h_window(
        self,
        hourly_data: Dict[str, List[Any]],
        current_time_str: Optional[str] = None,
        query_datetime: Optional[datetime] = None,
    ) -> Tuple[int, int]:
        """
        Determine [start_idx, end_idx) corresponding strictly to the 24 hours ending at current time.
        Guarantees that forecast hours are never included, and that the window represents the
        rolling 24-hour period up to the current observation time (not a calendar day).
        """
        times = [str(t) for t in hourly_data.get("time", []) if t is not None]
        ref_len = len(times)
        if ref_len == 0:
            for k in ("temperature_2m", "relative_humidity_2m", "precipitation"):
                if k in hourly_data and hourly_data[k]:
                    ref_len = len(hourly_data[k])
                    break

        if ref_len <= 24:
            return 0, ref_len

        # When timestamps are present and length > 24:
        if times and len(times) == ref_len:
            curr_target = current_time_str
            if not curr_target and query_datetime:
                curr_target = query_datetime.isoformat()

            if curr_target:
                matching = [i for i, t in enumerate(times) if str(t) <= str(curr_target)]
                curr_idx = matching[-1] if matching else 0
            else:
                now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
                matching = [i for i, t in enumerate(times) if str(t) <= now_utc]
                if not matching:
                    now_local = datetime.now().strftime("%Y-%m-%dT%H:00")
                    matching = [i for i, t in enumerate(times) if str(t) <= now_local]
                curr_idx = matching[-1] if matching else min(168, len(times) - 1)

            start_idx = max(0, curr_idx - 23)
            end_idx = curr_idx + 1
            return start_idx, end_idx

        # If no timestamps provided but length > 24: take last 24
        return max(0, ref_len - 24), ref_len

    def compute_summary_statistics(
        self,
        hourly_data: Dict[str, List[Any]],
        rh_threshold_pct: float = 90.0,
        precip_threshold_mm: float = 0.1,
        current_time_str: Optional[str] = None,
        query_datetime: Optional[datetime] = None,
    ) -> WeatherSummaryStats:
        """
        Compute aggregate summary statistics from hourly weather data.
        Derives leaf-wetness proxy tagged [ENG proxy].
        When a full time series with timestamps is provided, computes over historical past window up to current time.
        """
        times = [str(t) for t in hourly_data.get("time", []) if t is not None]
        if times and len(times) > 24:
            _, end_idx = self._isolate_past_24h_window(
                hourly_data, current_time_str=current_time_str, query_datetime=query_datetime
            )
            # Use past data up to the current hour (excludes future forecast)
            data_slice = {k: v[:end_idx] if isinstance(v, list) else v for k, v in hourly_data.items()}
        else:
            data_slice = hourly_data

        temps = [float(v) for v in data_slice.get("temperature_2m", []) if v is not None]
        rhs = [float(v) for v in data_slice.get("relative_humidity_2m", []) if v is not None]
        precips = [float(v) for v in data_slice.get("precipitation", []) if v is not None]
        winds = [float(v) for v in data_slice.get("wind_speed_10m", []) if v is not None]

        mean_temp = sum(temps) / len(temps) if temps else 0.0
        max_temp = max(temps) if temps else 0.0
        min_temp = min(temps) if temps else 0.0

        mean_rh = sum(rhs) / len(rhs) if rhs else 0.0
        max_rh = max(rhs) if rhs else 0.0
        min_rh = min(rhs) if rhs else 0.0

        total_precip = sum(precips) if precips else 0.0
        precip_hours = sum(1 for p in precips if p >= precip_threshold_mm)

        # Hours with RH >= threshold [ENG proxy]
        hours_rh_above = sum(1 for r in rhs if r >= rh_threshold_pct)

        # Leaf wetness hours proxy: RH >= 90% OR precip >= 0.1mm [ENG proxy]
        n_hours = min(len(rhs), len(precips))
        leaf_wetness_hours = 0
        for i in range(n_hours):
            if rhs[i] >= rh_threshold_pct or precips[i] >= precip_threshold_mm:
                leaf_wetness_hours += 1

        max_wind = max(winds) if winds else 0.0

        return WeatherSummaryStats(
            mean_temperature_c=round(mean_temp, 2),
            max_temperature_c=round(max_temp, 2),
            min_temperature_c=round(min_temp, 2),
            mean_relative_humidity_pct=round(mean_rh, 2),
            max_relative_humidity_pct=round(max_rh, 2),
            min_relative_humidity_pct=round(min_rh, 2),
            total_precipitation_mm=round(total_precip, 2),
            precipitation_hours=precip_hours,
            hours_rh_above_threshold=hours_rh_above,
            leaf_wetness_hours_proxy=leaf_wetness_hours,
            max_wind_speed_kmh=round(max_wind, 2),
        )

    def evaluate_rules(
        self,
        hourly_data: Dict[str, List[Any]],
        summary_stats: Optional[WeatherSummaryStats] = None,
        rh_threshold_pct: float = 90.0,
        precip_threshold_mm: float = 0.1,
        current_time_str: Optional[str] = None,
        query_datetime: Optional[datetime] = None,
    ) -> List[RuleTriggerResult]:
        """
        Evaluate rules W01–W08 over the past 24 hours of data ending at current time.
        Never includes forecast hours; represents the rolling 24-hour period (not a calendar day).
        Returns a structured RuleTriggerResult for each rule.
        """
        if summary_stats is None:
            summary_stats = self.compute_summary_statistics(
                hourly_data,
                rh_threshold_pct,
                precip_threshold_mm,
                current_time_str=current_time_str,
                query_datetime=query_datetime,
            )

        # Isolate exactly the past 24 hours ending at current observation time
        start_idx, end_idx = self._isolate_past_24h_window(
            hourly_data, current_time_str=current_time_str, query_datetime=query_datetime
        )

        raw_temps = hourly_data.get("temperature_2m", [])[start_idx:end_idx]
        raw_rhs = hourly_data.get("relative_humidity_2m", [])[start_idx:end_idx]
        raw_precips = hourly_data.get("precipitation", [])[start_idx:end_idx]
        raw_winds = hourly_data.get("wind_speed_10m", [])[start_idx:end_idx]
        raw_dews = hourly_data.get("dew_point_2m", [])[start_idx:end_idx]

        w_temps = [float(v) for v in raw_temps if v is not None]
        w_rhs = [float(v) for v in raw_rhs if v is not None]
        w_precips = [float(v) for v in raw_precips if v is not None]
        w_winds = [float(v) for v in raw_winds if v is not None]
        w_dews = [float(v) if v is not None else None for v in raw_dews]

        w_min_temp = min(w_temps) if w_temps else 0.0
        w_max_temp = max(w_temps) if w_temps else 0.0
        w_max_rh = max(w_rhs) if w_rhs else 0.0
        w_min_rh = min(w_rhs) if w_rhs else 0.0
        w_precip_sum = sum(w_precips) if w_precips else 0.0
        w_precip_hours = sum(1 for p in w_precips if p >= precip_threshold_mm)
        w_max_wind = max(w_winds) if w_winds else 0.0

        # Leaf-wetness proxy in evaluation window [ENG proxy]
        w_leaf_wetness_hours = 0
        n_win = min(len(w_rhs), len(w_precips))
        for i in range(n_win):
            if w_rhs[i] >= rh_threshold_pct or w_precips[i] >= precip_threshold_mm:
                w_leaf_wetness_hours += 1

        results: List[RuleTriggerResult] = []

        for rule in self.rules_config:
            r_id = rule.get("id")
            name = rule.get("condition_name", f"Rule {r_id}")
            source = rule.get("threshold_source", "[UNVERIFIED — I14]")
            threshold_expr = rule.get("threshold_expression", "")
            time_window = rule.get("time_window", "past_24h")
            applicable_classes = rule.get("applicable_classes", [])
            omitted = bool(rule.get("omitted_from_scoring", False))
            criteria = rule.get("criteria", {})

            triggered = False
            measured_str = ""

            if r_id == "W01":
                # Late blight: RH > 90% AND temp_min < 18°C
                rh_min = float(criteria.get("rh_min", 90.0))
                temp_min_max = float(criteria.get("temp_min_max", 18.0))
                triggered = (w_max_rh > rh_min) and (w_min_temp < temp_min_max)
                measured_str = f"RH_max={w_max_rh:.1f}%, temp_min={w_min_temp:.1f}°C"

            elif r_id == "W02":
                # Powdery mildew: RH < 50% AND temp 20–30°C
                rh_max = float(criteria.get("rh_max", 50.0))
                t_low, t_high = criteria.get("temp_range", [20.0, 30.0])
                # Triggered if any hour meets condition
                matching_hours = [
                    i for i in range(min(len(w_rhs), len(w_temps)))
                    if w_rhs[i] < rh_max and float(t_low) <= w_temps[i] <= float(t_high)
                ]
                triggered = len(matching_hours) > 0
                if triggered:
                    best_h = matching_hours[0]
                    measured_str = f"RH={w_rhs[best_h]:.1f}%, temp={w_temps[best_h]:.1f}°C ({len(matching_hours)}h matched)"
                else:
                    measured_str = f"RH_min={w_min_rh:.1f}%, temp_range=[{w_min_temp:.1f}, {w_max_temp:.1f}]°C"

            elif r_id == "W03":
                # Bacterial spot: temp > 24°C AND precip > 0 AND wind > 15 km/h
                t_min = float(criteria.get("temp_min", 24.0))
                p_min = float(criteria.get("precip_min", 0.0))
                w_min = float(criteria.get("wind_min", 15.0))
                n_checks = min(len(w_temps), len(w_precips), len(w_winds))
                matching_hours = [
                    i for i in range(n_checks)
                    if w_temps[i] > t_min and w_precips[i] > p_min and w_winds[i] > w_min
                ]
                triggered = len(matching_hours) > 0
                if triggered:
                    best_h = matching_hours[0]
                    measured_str = (
                        f"temp={w_temps[best_h]:.1f}°C, precip={w_precips[best_h]:.1f}mm, "
                        f"wind={w_winds[best_h]:.1f}km/h ({len(matching_hours)}h matched)"
                    )
                else:
                    measured_str = f"temp_max={w_max_temp:.1f}°C, precip_sum={w_precip_sum:.1f}mm, wind_max={w_max_wind:.1f}km/h"

            elif r_id == "W04":
                # Early blight: temp > 24°C AND (precip_hours > 2 OR leaf_wetness_proxy > 2h)
                t_min = float(criteria.get("temp_min", 24.0))
                p_hrs_min = int(criteria.get("precip_hours_min", 2))
                lw_min = int(criteria.get("leaf_wetness_hours_min", 2))
                triggered = (w_max_temp > t_min) and (w_precip_hours > p_hrs_min or w_leaf_wetness_hours > lw_min)
                measured_str = (
                    f"temp_max={w_max_temp:.1f}°C, precip_hours={w_precip_hours}, "
                    f"leaf_wetness_proxy={w_leaf_wetness_hours}h [ENG proxy]"
                )

            elif r_id == "W05":
                # Septoria leaf spot: temp 15–25°C AND RH > 85%
                t_low, t_high = criteria.get("temp_range", [15.0, 25.0])
                rh_min = float(criteria.get("rh_min", 85.0))
                matching_hours = [
                    i for i in range(min(len(w_rhs), len(w_temps)))
                    if float(t_low) <= w_temps[i] <= float(t_high) and w_rhs[i] > rh_min
                ]
                triggered = len(matching_hours) > 0
                if triggered:
                    best_h = matching_hours[0]
                    measured_str = f"temp={w_temps[best_h]:.1f}°C, RH={w_rhs[best_h]:.1f}% ({len(matching_hours)}h matched)"
                else:
                    measured_str = f"temp_range=[{w_min_temp:.1f}, {w_max_temp:.1f}]°C, RH_max={w_max_rh:.1f}%"

            elif r_id == "W06":
                # Yellow leaf curl virus: temp > 28°C AND RH < 40%
                t_min = float(criteria.get("temp_min", 28.0))
                rh_max = float(criteria.get("rh_max", 40.0))
                matching_hours = [
                    i for i in range(min(len(w_rhs), len(w_temps)))
                    if w_temps[i] > t_min and w_rhs[i] < rh_max
                ]
                triggered = len(matching_hours) > 0
                if triggered:
                    best_h = matching_hours[0]
                    measured_str = f"temp={w_temps[best_h]:.1f}°C, RH={w_rhs[best_h]:.1f}% ({len(matching_hours)}h matched)"
                else:
                    measured_str = f"temp_max={w_max_temp:.1f}°C, RH_min={w_min_rh:.1f}%"

            elif r_id == "W07":
                # Rust: temp 15–22°C AND dew_point within 2°C of temp_min
                t_low, t_high = criteria.get("temp_range", [15.0, 22.0])
                dp_diff_max = float(criteria.get("dew_point_diff_max", 2.0))
                # Check hours where temp in range and abs(dew_point - temp) <= dp_diff_max
                n_dp = min(len(w_temps), len(w_dews))
                diffs = [
                    abs(w_dews[i] - w_temps[i])
                    for i in range(n_dp)
                    if w_dews[i] is not None and float(t_low) <= w_temps[i] <= float(t_high)
                ]
                if diffs:
                    min_diff = min(diffs)
                    triggered = min_diff <= dp_diff_max
                    if triggered:
                        measured_str = f"temp in [{t_low}, {t_high}]°C, dew_point_diff={min_diff:.1f}°C <= {dp_diff_max}°C"
                    else:
                        measured_str = f"temp_range=[{w_min_temp:.1f}, {w_max_temp:.1f}]°C, min_dew_point_diff={min_diff:.1f}°C"
                else:
                    triggered = False
                    measured_str = f"temp_range=[{w_min_temp:.1f}, {w_max_temp:.1f}]°C, min_dew_point_diff=null"

            elif r_id == "W08":
                # General epidemic risk: precipitation_sum > 10mm in last 24h [GAP]
                p_sum_min = float(criteria.get("precip_sum_min", 10.0))
                triggered = w_precip_sum > p_sum_min
                measured_str = f"precip_sum_24h={w_precip_sum:.1f}mm"

            else:
                logger.warning("Unknown rule id: %s", r_id)
                continue

            results.append(
                RuleTriggerResult(
                    rule_id=r_id,
                    condition_name=name,
                    triggered=triggered,
                    measured_values=measured_str,
                    threshold=threshold_expr,
                    time_window=time_window,
                    threshold_source=source,
                    disease_targets=applicable_classes,
                    tag="[UNVERIFIED — I14]",
                    omitted_from_scoring=omitted,
                )
            )

        return results

    def to_legacy_risk_flags(self, rule_results: List[RuleTriggerResult]) -> List[WeatherRiskFlag]:
        """Convert RuleTriggerResult list to backward-compatible WeatherRiskFlag objects."""
        return [
            WeatherRiskFlag(
                condition_name=f"{r.condition_name}",
                triggered=r.triggered,
                current_value=r.measured_values,
                threshold=r.threshold,
                threshold_source=r.threshold_source,
            )
            for r in rule_results
        ]

    def compute_weather_score(
        self,
        rule_results: List[RuleTriggerResult],
        diagnosed_disease: Optional[str] = None,
    ) -> float:
        """
        Compute deterministic weather_score in [0.0, 1.0] per Part 4 §4a-iv.

        Formula:
          1. Filter rules: W01–W07 only (W08 is excluded [GAP] per spec).
          2. If diagnosed_disease is specified:
             - Select rules applicable to this disease.
             - If count(applicable) == 0: return 0.0.
             - Else: return count(triggered applicable) / count(total applicable).
          3. If diagnosed_disease is None:
             - Return count(triggered in W01–W07) / 7.0.
        """
        # Step 1: Filter out omitted rules (e.g. W08)
        scoring_rules = [r for r in rule_results if not r.omitted_from_scoring and r.rule_id in ("W01", "W02", "W03", "W04", "W05", "W06", "W07")]
        if not scoring_rules:
            return 0.0

        # Step 2: Disease-specific scoring
        if diagnosed_disease:
            disease_clean = diagnosed_disease.strip()
            # Match rules where diagnosed_disease is in disease_targets or name matches
            applicable = [
                r for r in scoring_rules
                if disease_clean in r.disease_targets
                or any(t.lower() in disease_clean.lower() for t in r.disease_targets)
            ]
            if not applicable:
                logger.debug("No weather rules applicable to disease '%s'; weather_score=0.0", disease_clean)
                return 0.0

            triggered_count = sum(1 for r in applicable if r.triggered)
            score = triggered_count / len(applicable)
            return round(min(1.0, max(0.0, score)), 4)

        # Step 3: Generic scoring across W01–W07
        triggered_count = sum(1 for r in scoring_rules if r.triggered)
        score = triggered_count / len(scoring_rules)
        return round(min(1.0, max(0.0, score)), 4)

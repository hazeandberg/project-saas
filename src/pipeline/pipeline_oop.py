from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ✅ On réutilise tes fonctions "réelles" (Étape 3)
from src.pipeline.parse_clean import parse_clients, parse_subscriptions, parse_usage
from src.pipeline.group_aggregate import aggregate_by_client
from src.pipeline.sort_report import build_report


# -----------------------------
# Logging (simple)
# -----------------------------
log = logging.getLogger("saas_pipeline_oop")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    out_report_csv: Path = Path("data/processed/report_oop.csv")

    clients_csv: Path = Path("data/raw/clients.csv")
    subscriptions_csv: Path = Path("data/raw/subscriptions.csv")
    usage_csv: Path = Path("data/raw/usage.csv")


# -----------------------------
# Components
# -----------------------------
class Loader:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def load(self) -> dict[str, list[dict[str, Any]]]:
        log.info("Loader | parsing CSV (reusing parse_clean.py)")

        clients = parse_clients(self.cfg.clients_csv)
        subs = parse_subscriptions(self.cfg.subscriptions_csv)
        usage = parse_usage(self.cfg.usage_csv)

        log.info(
            "Loader | parsed: clients=%s subs=%s usage=%s",
            len(clients),
            len(subs),
            len(usage),
        )
        return {"clients": clients, "subscriptions": subs, "usage": usage}


class Cleaner:
    """
    Ici, le 'clean' est déjà fait dans parse_clean.py (filtrage + types).
    On garde la classe pour la structure OOP + évolutions futures (règles, warnings, etc.).
    """

    def clean(
        self, data: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        log.info("Cleaner | no-op (already cleaned by parse_clean.py)")
        return data


class Analyzer:
    def analyze(
        self, data: dict[str, list[dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        log.info("Analyzer | aggregating (reusing group_aggregate.py)")

        agg = aggregate_by_client(
            clients=data["clients"],
            subscriptions=data["subscriptions"],
            usage=data["usage"],
        )

        log.info("Analyzer | aggregated clients=%s", len(agg))
        return agg


class Reporter:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _to_csv_value(v: Any) -> Any:
        # csv module écrit mieux des str/numbers; on convertit les datetime.
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d")
        return v

    def build(self, agg: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        log.info("Reporter | building report (reusing sort_report.py)")
        report = build_report(agg)
        log.info("Reporter | report rows=%s", len(report))
        return report

    def write_csv(self, report: list[dict[str, Any]]) -> Path:
        if not report:
            raise ValueError("Reporter | empty report rows")

        self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)

        out = self.cfg.out_report_csv
        fieldnames = list(report[0].keys())

        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in report:
                w.writerow({k: self._to_csv_value(v) for k, v in row.items()})

        log.info("Reporter | wrote CSV: %s", out)
        return out

    def print_top(self, report: list[dict[str, Any]], n: int = 5) -> None:
        log.info("Reporter | top %s rows", n)
        for r in report[:n]:
            log.info(
                "client_id=%s | plan=%s | ville=%s | ca_total=%s | actions_total=%s | sessions_total=%s",
                r.get("client_id"),
                r.get("plan"),
                r.get("ville"),
                r.get("ca_total"),
                r.get("actions_total"),
                r.get("sessions_total"),
            )


# -----------------------------
# Orchestrator
# -----------------------------
def run_pipeline(cfg: PipelineConfig) -> Path:
    setup_logging()

    log.info("Pipeline | start")
    loader = Loader(cfg)
    cleaner = Cleaner()
    analyzer = Analyzer()
    reporter = Reporter(cfg)

    data = loader.load()
    data = cleaner.clean(data)
    agg = analyzer.analyze(data)
    report = reporter.build(agg)
    out = reporter.write_csv(report)
    reporter.print_top(report)

    log.info("Pipeline | done")
    return out


if __name__ == "__main__":
    run_pipeline(PipelineConfig())

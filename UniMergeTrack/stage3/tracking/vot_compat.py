"""Run VOT CLI with compatibility patches for mixed VOT/TraX versions."""

import os
import re
import sys
from pathlib import Path

import numpy as np
import trax.client
from vot.utilities.cli import main


if not hasattr(np, "int"):
    np.int = int

_original_initialize = trax.client.Client.initialize
_original_frame = trax.client.Client.frame


class _Properties(dict):
    def dict(self):
        return dict(self)


def _as_object_list(region_or_objects):
    if isinstance(region_or_objects, list):
        return region_or_objects
    return [(region_or_objects, {})]


def _single_object_result(result):
    objects, elapsed = result
    if len(objects) != 1:
        raise RuntimeError("Expected a single object response from tracker.")
    region, properties = objects[0]
    return region, _Properties(properties), elapsed


def _initialize_single_object(self, images, objects, properties):
    return _single_object_result(
        _original_initialize(self, images, _as_object_list(objects), properties)
    )


def _frame_single_object(self, images, properties=dict(), objects=None):
    object_list = [] if objects is None else _as_object_list(objects)
    return _single_object_result(_original_frame(self, images, properties, object_list))


trax.client.Client.initialize = _initialize_single_object
trax.client.Client.frame = _frame_single_object


def _latest_report(workspace):
    analysis_dir = workspace / "analysis"
    if not analysis_dir.is_dir():
        return None
    reports = [path / "report.html" for path in analysis_dir.iterdir()]
    reports = [path for path in reports if path.is_file()]
    if not reports:
        return None
    return max(reports, key=lambda path: path.stat().st_mtime)


def _extract_metrics(report_path):
    html = report_path.read_text(encoding="utf-8")
    rows = re.findall(
        r'<tr data-tracker="([^"]+)">(.*?)</tr>',
        html,
        flags=re.DOTALL,
    )
    metrics = []
    for tracker, row in rows:
        values = re.findall(r'data-value="([^"]+)"', row)
        if len(values) >= 3:
            metrics.append((tracker, values[0], values[1], values[2]))
    return metrics


def _write_metrics_after_analysis():
    report_path = _latest_report(Path(os.getcwd()))
    if report_path is None:
        print("No VOT analysis report found, metrics txt was not written.")
        return

    metrics = _extract_metrics(report_path)
    if not metrics:
        print("No metrics found in VOT report, metrics txt was not written.")
        return

    output_path = report_path.parent / "metrics.txt"
    
    rows = [("tracker", "EAO", "Acc", "Rob")]

    for tracker, eao, acc, rob in metrics:
        rows.append((
            str(tracker),
            f"{float(eao):.4f}",
            f"{float(acc):.4f}",
            f"{float(rob):.4f}",
        ))

    widths = [
        max(len(row[i]) for row in rows)
        for i in range(len(rows[0]))
    ]

    lines = []
    for row in rows:
        lines.append(
            f"{row[0]:<{widths[0]}}  "
            f"{row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  "
            f"{row[3]:>{widths[3]}}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


    print("Saved VOT metrics to {}".format(output_path))
    for tracker, eao, acc, rob in metrics:
        print(
            "{}: EAO={}, Acc={}, Rob={}".format(
                tracker, eao, acc, rob
            )
        )


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = exc.code
    if exit_code in (None, 0) and len(sys.argv) > 1 and sys.argv[1] == "analysis":
        _write_metrics_after_analysis()
    sys.exit(exit_code)

"""
Numerical precision characterization tools for GPT decoding.
Implements the reporting and artifacts defined in gpt_numerics_precision_spec.md.

Entry point: tools/numerics/precision_report.py
"""

from .precision_report import main as precision_report_main

__all__ = ["precision_report_main"]

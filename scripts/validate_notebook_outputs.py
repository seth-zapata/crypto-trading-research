#!/usr/bin/env python3
"""
Notebook Output Validation Script

Validates that Jupyter notebooks executed successfully and produced expected outputs.
Used as part of the phase completion checklist to ensure notebooks are working.

Usage:
    python scripts/validate_notebook_outputs.py notebooks/phase1/*.ipynb
    python scripts/validate_notebook_outputs.py notebooks/phase1/01_data_validation.ipynb

Exit codes:
    0 - All notebooks passed validation
    1 - One or more notebooks failed validation

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validating a single notebook."""
    notebook_path: Path
    passed: bool
    errors: List[str]
    warnings: List[str]
    has_plots: bool
    has_output: bool
    cell_count: int
    executed_cell_count: int


def validate_notebook(notebook_path: Path) -> ValidationResult:
    """
    Validate that a notebook executed successfully and has expected outputs.

    Checks for:
    - Execution errors in any cell
    - Presence of output (proves it ran)
    - Presence of plots/visualizations
    - All code cells have been executed

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        ValidationResult with pass/fail status and details
    """
    errors: List[str] = []
    warnings: List[str] = []
    has_plots = False
    has_output = False
    cell_count = 0
    executed_cell_count = 0

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            notebook_path=notebook_path,
            passed=False,
            errors=[f"Invalid JSON: {e}"],
            warnings=[],
            has_plots=False,
            has_output=False,
            cell_count=0,
            executed_cell_count=0
        )
    except FileNotFoundError:
        return ValidationResult(
            notebook_path=notebook_path,
            passed=False,
            errors=[f"File not found: {notebook_path}"],
            warnings=[],
            has_plots=False,
            has_output=False,
            cell_count=0,
            executed_cell_count=0
        )

    cells = nb.get('cells', [])

    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue

        cell_count += 1

        # Check if cell was executed (has execution_count)
        execution_count = cell.get('execution_count')
        if execution_count is not None:
            executed_cell_count += 1

        outputs = cell.get('outputs', [])

        for output in outputs:
            output_type = output.get('output_type', '')

            # Check for execution errors
            if output_type == 'error':
                ename = output.get('ename', 'Unknown')
                evalue = output.get('evalue', 'No message')
                errors.append(f"{ename}: {evalue}")

            # Check for plots (image output)
            if 'data' in output:
                data = output['data']
                if 'image/png' in data or 'image/jpeg' in data or 'image/svg+xml' in data:
                    has_plots = True

            # Check for any output (stream, execute_result, display_data)
            if output_type in ['stream', 'execute_result', 'display_data']:
                has_output = True

            # Also count error outputs as "output" (notebook ran, just had errors)
            if output_type == 'error':
                has_output = True

    # Generate warnings
    if cell_count > 0 and executed_cell_count < cell_count:
        warnings.append(
            f"Only {executed_cell_count}/{cell_count} code cells were executed"
        )

    if not has_output and cell_count > 0:
        warnings.append("Notebook has no output - did it actually run?")

    if not has_plots:
        warnings.append("Notebook has no plots/visualizations")

    # Determine pass/fail
    passed = len(errors) == 0 and has_output

    return ValidationResult(
        notebook_path=notebook_path,
        passed=passed,
        errors=errors,
        warnings=warnings,
        has_plots=has_plots,
        has_output=has_output,
        cell_count=cell_count,
        executed_cell_count=executed_cell_count
    )


def print_result(result: ValidationResult) -> None:
    """Print validation result in a formatted way."""
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"\n{status}: {result.notebook_path.name}")
    print(f"  Path: {result.notebook_path}")
    print(f"  Cells: {result.executed_cell_count}/{result.cell_count} executed")
    print(f"  Has output: {'Yes' if result.has_output else 'No'}")
    print(f"  Has plots: {'Yes' if result.has_plots else 'No'}")

    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for error in result.errors:
            # Truncate long error messages
            error_display = error[:100] + "..." if len(error) > 100 else error
            print(f"    - {error_display}")

    if result.warnings:
        print(f"  Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"    - {warning}")


def validate_notebooks(notebook_paths: List[Path]) -> Tuple[int, int]:
    """
    Validate multiple notebooks.

    Args:
        notebook_paths: List of paths to notebook files

    Returns:
        Tuple of (passed_count, failed_count)
    """
    passed = 0
    failed = 0

    for path in notebook_paths:
        result = validate_notebook(path)
        print_result(result)

        if result.passed:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if len(sys.argv) < 2:
        print("Usage: python validate_notebook_outputs.py <notebook.ipynb> [...]")
        print("\nExamples:")
        print("  python scripts/validate_notebook_outputs.py notebooks/phase1/*.ipynb")
        print("  python scripts/validate_notebook_outputs.py notebooks/phase1/01_data_validation.ipynb")
        return 1

    # Collect notebook paths
    notebook_paths: List[Path] = []

    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists():
            if path.is_file() and path.suffix == '.ipynb':
                notebook_paths.append(path)
            elif path.is_dir():
                notebook_paths.extend(path.glob('*.ipynb'))
        else:
            # Try glob pattern
            from glob import glob
            matches = glob(arg)
            for match in matches:
                p = Path(match)
                if p.suffix == '.ipynb':
                    notebook_paths.append(p)

    if not notebook_paths:
        print("No notebook files found.")
        return 1

    print(f"Validating {len(notebook_paths)} notebook(s)...")
    print("=" * 60)

    passed, failed = validate_notebooks(notebook_paths)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n⚠️  Some notebooks failed validation. Please fix errors before proceeding.")
        return 1
    else:
        print("\n✅ All notebooks validated successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

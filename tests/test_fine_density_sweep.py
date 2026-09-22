from wp1.run_agent4sr_fine_density_sweep import (
    classify,
    density_values,
)


GOLD = "sympy/core/_print_helpers.py::Printable"


def test_density_values_one_percent_steps():
    assert density_values(1.00, 0.97, 0.01) == [1.0, 0.99, 0.98, 0.97]


def test_classify_exact_hit():
    predictions = [
        GOLD,
        "sympy/core/basic.py::Basic",
        "sympy/core/symbol.py::Symbol",
        "sympy/core/expr.py::AtomicExpr",
        "sympy/core/expr.py::Expr",
    ]
    result = classify(predictions, GOLD)
    assert result["status"] == "HIT"
    assert result["exact_gold_rank"] == 1
    assert result["gold_file_hit"] is True


def test_classify_file_hit_but_exact_miss():
    predictions = [
        "sympy/core/_print_helpers.py::Printable.__str__",
        "sympy/core/basic.py::Basic",
        "sympy/core/symbol.py::Symbol",
        "sympy/core/expr.py::AtomicExpr",
        "sympy/core/expr.py::Expr",
    ]
    result = classify(predictions, GOLD)
    assert result["status"] == "MISS"
    assert result["exact_gold_rank"] is None
    assert result["gold_file_rank"] == 1


def test_classify_invalid_is_not_miss():
    result = classify(["sympy/core/basic.py::Basic"], GOLD)
    assert result["status"] == "INVALID"

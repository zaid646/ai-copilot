from evals.trajectory import score_trajectory


def test_exact_trajectory_scores_one():
    expected = [
        {"tool": "compute_kpi", "args": {"region": "WEST", "quarter": "2026Q3"}},
        {"tool": "run_sql", "args": {"query": "category breakdown"}},
    ]
    actual = [
        {"tool": "compute_kpi", "args": {"region": "west", "quarter": "2026Q3"}},
        {"tool": "run_sql", "args": {"query": "category breakdown by week"}},
    ]
    score = score_trajectory(expected, actual)
    assert score["tool_choice_accuracy"] == 1.0
    assert score["argument_validity"] == 1.0
    assert score["order_correctness"] == 1.0


def test_wrong_tool_and_order_are_penalised():
    expected = [
        {"tool": "compute_kpi", "args": {"region": "WEST"}},
        {"tool": "graphrag_search", "args": {"query": "merchandising"}},
    ]
    actual = [
        {"tool": "graphrag_search", "args": {"query": "merchandising"}},
        {"tool": "compute_kpi", "args": {"region": "WEST"}},
    ]
    score = score_trajectory(expected, actual)
    assert score["tool_choice_accuracy"] == 0.0
    assert score["order_correctness"] == 0.5


def test_empty_expectation_requires_no_calls():
    assert score_trajectory([], [])["tool_choice_accuracy"] == 1.0
    assert score_trajectory([], [{"tool": "run_sql", "args": {}}])[
        "tool_choice_accuracy"
    ] == 0.0


def test_query_argument_matches_token_subset_in_any_order():
    expected = [
        {"tool": "graphrag_search", "args": {"query": "capitalization threshold"}}
    ]
    actual = [
        {
            "tool": "graphrag_search",
            "args": {"query": "capitalization approval threshold"},
        }
    ]
    score = score_trajectory(expected, actual)
    assert score["argument_validity"] == 1.0


def test_multi_quarter_argument_matches_list():
    expected = [
        {
            "tool": "compute_kpi",
            "args": {"region": "WEST", "quarter": ["2026Q2", "2026Q3"]},
        }
    ]
    actual = [
        {
            "tool": "compute_kpi",
            "args": {"region": "WEST", "quarter": ["2026Q2", "2026Q3"]},
        }
    ]
    assert score_trajectory(expected, actual)["argument_validity"] == 1.0

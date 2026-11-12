"""PDDL dumps go to the run dir ($ADT4_RUN_DIR/omniplanner) under unique,
timestamped names; *_latest is only ever a symlink; and the solver reports
what it did (fd_stats, omniplanner.solve_stats). Fast Downward is faked."""

import os
import subprocess

import pytest
from dsg_pddl import dsg_pddl_grounding_multirobot as mr
from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_planning import resolve_dump_dir, solve_pddl

from omniplanner import solve_stats

DOMAIN = "(define (domain trivial) (:predicates (p) (q)))"
PROBLEM = "(define (problem t) (:domain trivial) (:init (p)) (:goal (q)))"
DUMP_ENVS = ("OMNIPLANNER_DUMP_DIR", "PDDL_DUMP_DIR", "ADT4_RUN_DIR", "ADT4_OUTPUT_DIR")


class _Domain:
    def to_string(self):
        return DOMAIN


def _problem():
    return GroundedPddlProblem(domain=_Domain(), problem_str=PROBLEM, symbols={})


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    for name in DUMP_ENVS:
        monkeypatch.delenv(name, raising=False)
    run = tmp_path / "runs" / "willow_1"
    monkeypatch.setenv("ADT4_RUN_DIR", str(run))
    monkeypatch.chdir(tmp_path)
    return run


def _fake_fd(monkeypatch, solved):
    def run(command, **kwargs):
        if solved:
            plan_fn = command[command.index("--plan-file") + 1]
            with open(plan_fn, "w") as fo:
                fo.write("(noop)\n; cost = 1 (unit cost)\n")
            return subprocess.CompletedProcess(command, 0, "Solution found!", "")
        return subprocess.CompletedProcess(command, 12, "Search stopped", "")

    monkeypatch.setattr(subprocess, "run", run)


def test_the_run_dir_is_the_dump_dir(run_dir):
    assert resolve_dump_dir() == str(run_dir / "omniplanner")


def test_an_explicit_dump_dir_wins(run_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", str(tmp_path / "mine"))
    assert resolve_dump_dir() == str(tmp_path / "mine")


def test_two_failed_solves_do_not_overwrite(run_dir, monkeypatch):
    _fake_fd(monkeypatch, solved=False)
    for _ in range(2):
        with pytest.raises(Exception):
            solve_pddl(_problem())
    dump = run_dir / "omniplanner"
    failed = sorted(dump.glob("pddl_problem_debugging_2*.pddl"))
    assert len(failed) == 2  # same second, different names
    assert all(f.read_text() == PROBLEM for f in failed)
    latest = dump / "pddl_problem_debugging_latest.pddl"
    assert latest.is_symlink() and latest.resolve() in failed
    assert len(list(dump.glob("omniplanner_problem_*.pddl"))) == 2
    assert not (run_dir.parent.parent / "pddl_problem_debugging.pddl").exists()


def test_without_a_run_dir_the_failed_problem_stays_where_it_was(tmp_path, monkeypatch):
    for name in DUMP_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ADT4_OUTPUT_DIR", str(tmp_path / "out"))
    os.makedirs(tmp_path / "out")
    _fake_fd(monkeypatch, solved=False)
    for _ in range(2):
        with pytest.raises(Exception):
            solve_pddl(_problem())
    assert len(list((tmp_path / "out").glob("pddl_problem_debugging_2*.pddl"))) == 2
    assert (tmp_path / "out" / "pddl_problem_debugging_latest.pddl").is_symlink()
    # the per-solve dumps keep their old default
    assert list((tmp_path / "home" / "adt4_output" / "omniplanner").iterdir())


def test_a_solve_reports_its_stats_and_files(run_dir, monkeypatch):
    _fake_fd(monkeypatch, solved=True)
    problem = _problem()
    solve_stats.start()
    plan = solve_pddl(problem)
    records, _ = solve_stats.finish()
    assert [tuple(a) for a in plan] == [("noop",)]
    stats = problem.fd_stats
    assert stats["seconds"] >= 0 and stats["solve_s"] >= stats["seconds"]
    for key in ("problem_file", "domain_file", "plan_file"):
        assert os.path.dirname(stats[key]) == str(run_dir / "omniplanner")
        assert os.path.exists(stats[key])
    (record,) = records
    assert record["engine"] == "FD" and record["ok"] is True
    assert record["problem_file"] == stats["problem_file"]


def test_a_failed_solve_reports_its_problem_file(run_dir, monkeypatch):
    _fake_fd(monkeypatch, solved=False)
    solve_stats.start()
    with pytest.raises(Exception):
        solve_pddl(_problem())
    (record,), _ = solve_stats.finish()
    assert record["ok"] is False and record["returncode"] == 12
    assert os.path.basename(record["failed_problem_file"]).startswith(
        "pddl_problem_debugging_"
    )


def test_nothing_is_recorded_without_a_collection(run_dir, monkeypatch):
    _fake_fd(monkeypatch, solved=True)
    solve_pddl(_problem())
    assert solve_stats.finish() == ([], {})


def test_region_problem_dumps_do_not_overwrite(run_dir):
    solve_stats.start()
    mr._dump_problem("(one)")
    mr._dump_problem("(two)")
    _, files = solve_stats.finish()
    dump = run_dir / "omniplanner"
    dumps = sorted(dump.glob("*_mr_region_problem.pddl"))
    assert len(dumps) == 2
    assert {d.read_text() for d in dumps} == {"(one)", "(two)"}
    latest = dump / "mr_region_problem_latest.pddl"
    assert latest.is_symlink() and latest.read_text() == "(two)"
    assert files["mr_region_problem"] == str(latest.resolve())


def test_region_problem_dump_dir_precedence(run_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("PDDL_DUMP_DIR", str(tmp_path / "pd"))
    mr._dump_problem("(x)")
    assert len(list((tmp_path / "pd").glob("*_mr_region_problem.pddl"))) == 1
    monkeypatch.delenv("PDDL_DUMP_DIR")
    monkeypatch.delenv("ADT4_RUN_DIR")
    mr._dump_problem("(y)")  # no env at all: <cwd>/pddl_dumps, as before
    assert (tmp_path / "pddl_dumps" / "mr_region_problem_latest.pddl").is_symlink()


def test_an_old_latest_file_is_replaced_by_a_link(run_dir):
    dump = run_dir / "omniplanner"
    dump.mkdir(parents=True)
    (dump / "mr_region_problem_latest.pddl").write_text("(old copy)")
    mr._dump_problem("(new)")
    latest = dump / "mr_region_problem_latest.pddl"
    assert latest.is_symlink() and latest.read_text() == "(new)"

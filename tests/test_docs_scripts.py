from pathlib import Path


def test_windows_launchers_are_non_global_and_create_project_local_run_dirs():
    root = Path(__file__).parents[1]
    ps = (root / "run-demo.ps1").read_text(encoding="utf-8")
    cmd = (root / "run-demo.cmd").read_text(encoding="utf-8")
    assert ".run" in ps
    assert "TEMP" in ps and "TMP" in ps
    assert "Set-ExecutionPolicy" not in ps
    assert "$Pid" not in ps and "$PID" not in ps
    assert "ExecutionPolicy Bypass" in cmd
    assert "run-demo.ps1" in cmd


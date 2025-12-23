from pathlib import Path


class TestFeatureFileLists:
    """Tests for file list population in FeatureStats."""

    def test_analyze_directory__accumulates_file_paths(self, tmp_path: Path) -> None:
        """Test that file lists are correctly populated with file paths."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # File 1: Orphan comment
        (src_dir / "orphan.py").write_text("# Just an orphan comment\ndef foo(): pass")

        # File 2: Untracked TODO
        (src_dir / "todo.py").write_text("# TODO: do something\ndef bar(): pass")

        # File 3: Inline import
        (src_dir / "inline.py").write_text("def baz():\n    import os\n    pass")

        # Initialize git repo so tracker works
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "test"], cwd=tmp_path, capture_output=True)

        from slopometry.core.python_feature_analyzer import PythonFeatureAnalyzer

        analyzer = PythonFeatureAnalyzer()
        stats = analyzer.analyze_directory(tmp_path)

        assert stats.orphan_comment_count == 1
        assert len(stats.orphan_comment_files) == 1
        assert str(src_dir / "orphan.py") in stats.orphan_comment_files

        assert stats.untracked_todo_count == 1
        assert len(stats.untracked_todo_files) == 1
        assert str(src_dir / "todo.py") in stats.untracked_todo_files

        assert stats.inline_import_count == 1
        assert len(stats.inline_import_files) == 1
        assert str(src_dir / "inline.py") in stats.inline_import_files

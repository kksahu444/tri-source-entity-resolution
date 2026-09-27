import pytest

from src import package


def test_refuses_doc_with_final_gaps(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("| final | [FINAL] | [FINAL run_id] |\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        package.main(["--team", "t", "--doc", str(doc), "--out", str(tmp_path / "t.zip")])
    assert not (tmp_path / "t.zip").exists()



def test_refuses_doc_without_team_header(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("**Team Name:** [team name]\n**Team Members:** [members]\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        package.main(["--team", "t", "--doc", str(doc), "--out", str(tmp_path / "t.zip")])
    assert not (tmp_path / "t.zip").exists()

from modab_root_finder import echo


def test_core():
    ans = echo("hello world")
    assert ans == 42

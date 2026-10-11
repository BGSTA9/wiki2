from wiki2shards.clean import filters


def test_filters():
    assert filters.min_words("one two three", 3) == ("min_words", True)
    assert filters.min_words("one two", 3) == ("min_words", False)
    assert filters.max_symbol_pct("ab@@", 50) == ("max_symbol_pct", True)
    assert filters.max_symbol_pct("a@@@", 50) == ("max_symbol_pct", False)
    assert filters.max_symbol_pct("नमस्ते", 0)[1]  # combining marks are not symbols
    assert filters.disambiguation(["Disambig"], ["disambig"]) == ("disambiguation", False)
    assert filters.disambiguation(["Template:Dab "], ["dab"])[1] is False
    assert filters.disambiguation(["Infobox"], ["dab"])[1] is True
    assert filters.list_page("List of rivers") == ("list_page", False)

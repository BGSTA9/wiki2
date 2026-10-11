import pytest
from helpers.tok import ADAPTERS
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.parametrize("kind", sorted(ADAPTERS))
@given(text=st.text())
def test_tokenizer_roundtrip(kind, text):
    tok = ADAPTERS[kind]()
    ids = tok.encode_batch([text])[0]
    assert tok.decode(ids) == text

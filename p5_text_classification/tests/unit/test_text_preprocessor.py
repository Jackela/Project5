import pytest
import torch
from p5_text_classification.src.data.text_preprocessor import TextPreprocessor

@pytest.fixture(scope="module")
def alphabet():
    return "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{} "

@pytest.fixture
def preprocessor(alphabet):
    """Returns a TextPreprocessor instance for testing."""
    # Using a shorter sequence length for easier testing
    return TextPreprocessor(alphabet=alphabet, sequence_length=10)

def test_init_builds_vocab_correctly(preprocessor, alphabet):
    """Tests that the vocabulary is built correctly upon initialization."""
    # +2 for PAD and UNK tokens
    assert preprocessor.vocab_size == len(alphabet) + 2
    assert preprocessor.char_to_id[preprocessor.PAD_TOKEN] == 0
    assert preprocessor.pad_id == 0
    assert preprocessor.char_to_id[preprocessor.UNK_TOKEN] == 1
    assert preprocessor.unk_id == 1
    assert preprocessor.char_to_id['a'] == 2

def test_encode_text_converts_correctly(preprocessor):
    """Tests that a simple string is encoded to the correct sequence of IDs."""
    text = "abc"
    expected_ids = [
        preprocessor.char_to_id['a'],
        preprocessor.char_to_id['b'],
        preprocessor.char_to_id['c'],
    ]
    # Padded to sequence_length of 10
    expected_padded = expected_ids + [preprocessor.pad_id] * 7
    
    encoded = preprocessor.encode_text(text, max_length=preprocessor.sequence_length)
    
    assert encoded == expected_padded

def test_encode_text_handles_unknown_chars(preprocessor):
    """Tests that characters not in the alphabet are mapped to the UNK token ID."""
    text = "a~b" # Assuming '~' is not in the limited alphabet for this test
    
    # Create a preprocessor with a limited alphabet
    limited_alphabet_preprocessor = TextPreprocessor(alphabet="abc", sequence_length=5)
    
    encoded = limited_alphabet_preprocessor.encode_text(text, max_length=5)
    
    expected_ids = [
        limited_alphabet_preprocessor.char_to_id['a'],
        limited_alphabet_preprocessor.unk_id, # '~' becomes <UNK>
        limited_alphabet_preprocessor.char_to_id['b'],
    ]
    expected_padded = expected_ids + [limited_alphabet_preprocessor.pad_id] * 2
    
    assert encoded == expected_padded

def test_encode_text_truncates_long_text(preprocessor):
    """Tests that text longer than sequence_length is truncated."""
    long_text = "abcdefghijklmnopqrstuvwxyz"
    assert len(long_text) > preprocessor.sequence_length
    
    encoded = preprocessor.encode_text(long_text, max_length=preprocessor.sequence_length)
    
    assert len(encoded) == preprocessor.sequence_length
    # Check that it's the beginning of the text that was kept
    expected_ids = [preprocessor.char_to_id[char] for char in long_text[:preprocessor.sequence_length]]
    assert encoded == expected_ids

def test_encode_text_handles_case_insensitivity(preprocessor):
    """Tests that uppercase text is converted to lowercase before encoding."""
    text = "ABC"
    expected_ids = [
        preprocessor.char_to_id['a'],
        preprocessor.char_to_id['b'],
        preprocessor.char_to_id['c'],
    ]
    expected_padded = expected_ids + [preprocessor.pad_id] * 7
    
    encoded = preprocessor.encode_text(text, max_length=preprocessor.sequence_length)
    
    assert encoded == expected_padded

def test_encode_text_exact_length(preprocessor):
    """Tests encoding a string that is exactly the sequence_length."""
    text = "1234567890" # sequence_length is 10
    assert len(text) == preprocessor.sequence_length
    
    expected_ids = [preprocessor.char_to_id[char] for char in text]
    encoded = preprocessor.encode_text(text, max_length=preprocessor.sequence_length)
    
    assert encoded == expected_ids
    assert len(encoded) == preprocessor.sequence_length

def test_encode_empty_string(preprocessor):
    """Tests that an empty string is encoded as a sequence of PAD tokens."""
    text = ""
    expected_padded = [preprocessor.pad_id] * preprocessor.sequence_length
    
    encoded = preprocessor.encode_text(text, max_length=preprocessor.sequence_length)
    
    assert encoded == expected_padded 
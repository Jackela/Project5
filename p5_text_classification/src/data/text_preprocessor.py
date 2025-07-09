"""
Character-level Text Preprocessor
"""

from typing import List, Dict
# Backward compatibility constants and functions
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{} "
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEQUENCE_LENGTH = 1014

def create_char_to_id_mapping() -> Dict[str, int]:
    """
    Creates a character-to-ID mapping for backward compatibility.
    Returns the same mapping as used by TextPreprocessor.
    """
    char_list = [PAD_TOKEN, UNK_TOKEN] + list(ALPHABET)
    return {char: i for i, char in enumerate(char_list)}

def tokenize_and_encode(text: str, char_to_id: Dict[str, int]) -> List[int]:
    """
    Tokenizes and encodes text to character IDs for backward compatibility.
    
    Args:
        text: Input text string
        char_to_id: Character to ID mapping
        
    Returns:
        List of character IDs with padding
    """
    text = str(text).lower()
    unk_id = char_to_id.get(UNK_TOKEN, 1)
    pad_id = char_to_id.get(PAD_TOKEN, 0)
    
    encoded = [char_to_id.get(char, unk_id) for char in text]
    
    if len(encoded) > SEQUENCE_LENGTH:
        encoded = encoded[:SEQUENCE_LENGTH]
    else:
        padding_needed = SEQUENCE_LENGTH - len(encoded)
        encoded.extend([pad_id] * padding_needed)
        
    return encoded

class TextPreprocessor:
    """
    Handles all text preprocessing tasks for character-level models.
    This class is responsible for:
    - Creating a character-to-ID mapping (vocabulary).
    - Encoding text strings into sequences of character IDs.
    - Padding sequences to a fixed length.
    """

    def __init__(self, alphabet: str, sequence_length: int):
        self.alphabet = alphabet
        self.sequence_length = sequence_length
        
        # Special tokens for padding and unknown characters.
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        
        self.char_to_id, self.id_to_char = self._build_vocab()
        
    def _build_vocab(self) -> (Dict[str, int], Dict[int, str]):
        """
        Builds the character vocabulary including special tokens.
        """
        char_list = [self.PAD_TOKEN, self.UNK_TOKEN] + list(self.alphabet)
        
        char_to_id = {char: i for i, char in enumerate(char_list)}
        id_to_char = {i: char for i, char in enumerate(char_list)}
        
        self.pad_id = char_to_id[self.PAD_TOKEN]
        self.unk_id = char_to_id[self.UNK_TOKEN]
        
        return char_to_id, id_to_char

    def encode_text(self, text: str) -> List[int]:
        """
        Encodes a single string into a list of character IDs, with padding.
        """
        text = str(text).lower()
        
        encoded = [self.char_to_id.get(char, self.unk_id) for char in text]
        
        if len(encoded) > self.sequence_length:
            encoded = encoded[:self.sequence_length]
        else:
            padding_needed = self.sequence_length - len(encoded)
            encoded.extend([self.pad_id] * padding_needed)
            
        return encoded

    @property
    def vocab_size(self) -> int:
        """Returns the total size of the character vocabulary."""
        return len(self.char_to_id) 
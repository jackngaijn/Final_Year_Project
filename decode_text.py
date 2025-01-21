from transformers import AutoTokenizer

class TextDecoder:
    def __init__(self, tokenizer_name="bert-base-uncased"):
        """
        Initialize the TextDecoder with a tokenizer.
        
        Args:
            tokenizer_name (str): Name of the tokenizer to use (default: 'bert-base-uncased').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def decode_text(self, token_ids, skip_special_tokens=True):
        """
        Decode a list of token IDs into human-readable text.
        
        Args:
            token_ids (list[int]): List of token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens like [CLS], [SEP], etc.
        
        Returns:
            str: The decoded human-readable text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_answer(self, context_ids, start_pos, end_pos, skip_special_tokens=True):
        """
        Decode the answer from a context given the start and end positions.
        
        Args:
            context_ids (list[int]): List of token IDs for the context.
            start_pos (int): Start position of the answer in the context.
            end_pos (int): End position of the answer in the context.
            skip_special_tokens (bool): Whether to skip special tokens like [CLS], [SEP], etc.
        
        Returns:
            str: The decoded human-readable answer.
        """
        answer_ids = context_ids[start_pos:end_pos + 1]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=skip_special_tokens)
    
if __name__ == '__main__':
    # Initialize the decoder
    decoder = TextDecoder(tokenizer_name="bert-base-uncased")
    # Example: Decoding a single tokenized text
    token_ids = [101, 7592, 2003, 1037, 6584, 102, 0]  # Example token IDs
    decoded_text = decoder.decode_text(token_ids)
    print("Decoded Text:", decoded_text)
def gen_embeddings(sentences, batch_size=8, max_length=128, use_gpu=True):
    """
    Generate sentence embeddings using a SentenceTransformer model with SapBERT weights.

    Args:
        sentences (list of str): A list of sentences to encode.
        batch_size (int): Number of sentences to process at a time (for batching).
        max_length (int): Maximum sequence length for tokenization.
        use_gpu (bool): Whether to use GPU for computation. Defaults to True.

    Returns:
        np.ndarray: The embeddings for the input sentences.
    """
    # Determine the device to use: GPU (if available and requested) or CPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Load the SentenceTransformer model with SapBERT weights
    sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens").to(device)
    sentence_transformer_model = sentence_model._first_module().auto_model  # Access underlying BERT model
    sentence_transformer_model.load_state_dict(sapbert_model.state_dict(), strict=False)

    # Tokenizer for the SentenceTransformer model
    tokenizer = sapbert_tokenizer  # Reuse SapBERT tokenizer

    # Store all embeddings here
    all_embeddings = []

    # Process the sentences in batches
    for i in range(0, len(sentences), batch_size):
        # Get the current batch
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize the batch with truncation to limit sequence length
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)

        # Generate embeddings without computing gradients (for efficiency)
        with torch.no_grad():
            model_output = sentence_transformer_model(**encoded_input)
            batch_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()

        # Append batch embeddings to the list
        all_embeddings.append(batch_embeddings)

        # Clear the GPU cache to free memory
        torch.cuda.empty_cache()

    # Concatenate all batch embeddings into a single array
    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings
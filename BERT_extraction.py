import pandas as pd
import torch
from transformers import T5Tokenizer, T5Model
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
from typing import List, Tuple
import sys
import logging
from pathlib import Path
import numpy as np
from typing import List, Tuple
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import sentencepiece
    except ImportError:
        missing_deps.append("sentencepiece")
    
    try:
        from transformers import T5Tokenizer, T5Model
    except ImportError:
        missing_deps.append("transformers")
    
    if missing_deps:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_deps)}\n"
            "Please install them using:\n"
            "pip install torch transformers sentencepiece"
        )

class ProteinEmbeddingGenerator:
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_uniref50", device: str = None):
        """
        Initialize the embedding generator with specified model and device.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        # Check dependencies first
        check_dependencies()
        
        # Import required libraries after checking
        import torch
        from transformers import T5Tokenizer, T5Model
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            
            logger.info("Loading model...")
            self.model = T5Model.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing model/tokenizer: {str(e)}")
            raise

    def process_sequence(self, sequence: str, max_length: int = 512) -> np.ndarray:
        """
        Generate embedding for a single protein sequence.
        
        Args:
            sequence: Protein sequence string
            max_length: Maximum sequence length to process
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        import torch
        
        # Clean sequence and check validity
        sequence = ''.join(char for char in sequence if char.isalpha())
        if not sequence:
            raise ValueError("Empty or invalid sequence")

        # Tokenize and generate embedding
        try:
            inputs = self.tokenizer(sequence, 
                                  return_tensors="pt", 
                                  truncation=True, 
                                  max_length=max_length)
            logger.info(f"WAZAAA DID I GOT HERE?...{inputs}")
            # Get the input token embeddings
            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()(inputs.input_ids.to(self.device))
                logger.info(f"LLAMA DID I GOT HERE?...{input_embeddings}")
                embedding = input_embeddings.mean(dim=1).squeeze().cpu().numpy()
                logger.info(f"LLAMA222 DID I GOT HERE?...{embedding}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing sequence: {str(e)}")
            raise

    def generate_embeddings(self, 
                          input_file: str, 
                          output_file: str,
                          sequence_col: str = 'seq',
                          id_col: str = 'uniprot',
                          batch_size: int = 32) -> None:
        """
        Generate embeddings for sequences in a CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save embeddings
            sequence_col: Name of column containing sequences
            id_col: Name of column containing protein IDs
            batch_size: Number of sequences to process at once
        """
        from tqdm import tqdm
        
        output_path = Path(output_file)
        
        try:
            # Read input file
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} sequences from {input_file}")
            
            # Process sequences in batches
            embeddings_list = []
            
            for i in tqdm(range(0, len(df), batch_size), desc="Generating embeddings"):
                batch_df = df.iloc[i:i + batch_size]
                
                for _, row in batch_df.iterrows():
                    try:
                        sequence = str(row[sequence_col])
                        protein_id = row[id_col]
                        
                        embedding = self.process_sequence(sequence)
                        embeddings_list.append([protein_id] + embedding.tolist())
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {protein_id}: {str(e)}")
                        continue
                
                # Periodically save progress
                if i % 1000 == 0 and i > 0:
                    self._save_embeddings(embeddings_list, output_path, append=True)
                    embeddings_list = []  # Clear list after saving
            
            # Save any remaining embeddings
            if embeddings_list:
                self._save_embeddings(embeddings_list, output_path, append=True)
            
            logger.info(f"Embeddings saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise

    def _save_embeddings(self, 
                        embeddings_list: List[List[float]], 
                        output_path: Path,
                        append: bool = False) -> None:
        """
        Save embeddings to CSV file.
        
        Args:
            embeddings_list: List of embeddings with protein IDs
            output_path: Path to save file
            append: Whether to append to existing file
        """
        if not embeddings_list:
            return
            
        embedding_columns = [f'emb_{i+1}' for i in range(len(embeddings_list[0]) - 1)]
        df = pd.DataFrame(embeddings_list, columns=['ID'] + embedding_columns)
        
        if append and output_path.exists():
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)


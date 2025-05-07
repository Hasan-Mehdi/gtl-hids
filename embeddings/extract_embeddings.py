import os
# Set specific GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
import argparse
from pathlib import Path

def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class EmbeddingExtractor:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct", device="cuda", layer=-1, output_dir="embeddings_output"):
        """
        Initialize the model and tokenizer
        Args:
            model_name: Name or path of the model
            device: Device to run the model on
            layer: Which layer to extract embeddings from (-1 for last layer)
            output_dir: Directory to save extracted embeddings and metadata
        """
        self.logger = setup_logging()
        self.device = device
        
        self.logger.info(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,  # Set pad token ID in model
            output_hidden_states=True  # Added this for embedding extraction
        )
        
        self.layer = layer
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def format_input(self, row):
        """Format a single data point for embedding extraction"""
        text = f"""Source IP: {row['Src IP']} (Port: {row['Src Port']})
Destination IP: {row['Dst IP']} (Port: {row['Dst Port']})
Protocol: {row['Protocol']}
Traffic Volume: {row['Total Length of Fwd Packet']} bytes in, {row['Total Length of Bwd Packet']} bytes out
Packets: {row['Total Fwd Packet']} packets in, {row['Total Bwd packets']} packets out
TCP Flags: {row.get('TCP Flags Count', row.get('FIN Flag Count', 0))}
Duration: {row['Flow Duration']} ms"""
        return text
    
    def extract_embeddings_from_batch(self, batch_texts):
        """Extract embeddings for a batch of texts"""
        batch_inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[self.layer]
            
            # Mean pooling over all tokens (excluding padding)
            attention_mask = batch_inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, 1)
            sum_mask = torch.sum(mask, 1)
            batch_embeddings = sum_embeddings / sum_mask
            
            return batch_embeddings.cpu().numpy()

    def process_dataset_files(self, data_dir, sample_size=25000, batch_size=16):
        """
        Process multiple dataset files, extracting embeddings with balanced sampling
        
        Args:
            data_dir: Directory containing dataset files
            sample_size: Number of samples to extract per category (benign/malicious)
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with embeddings and metadata for each file
        """
        results = {}
        
        # Define the five weekday files
        weekday_files = ['Monday.csv', 'Tuesday.csv', 'Wednesday.csv', 'Thursday.csv', 'Friday.csv']
        
        for file_name in weekday_files:
            file_path = os.path.join(data_dir, file_name)
            
            if not os.path.exists(file_path):
                self.logger.warning(f"File {file_path} not found, skipping...")
                continue
                
            self.logger.info(f"Processing {file_name}...")
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                self.logger.info(f"  Total rows in file: {len(df)}")
                
                # Split into benign and malicious
                benign_df = df[df['Label'].str.upper() == 'BENIGN']
                malicious_df = df[df['Label'].str.upper() != 'BENIGN']
                
                self.logger.info(f"  Found {len(benign_df)} benign and {len(malicious_df)} malicious samples")
                
                # Handle Monday specially (only benign)
                if file_name.lower() == 'monday.csv' or len(malicious_df) == 0:
                    self.logger.info(f"  {file_name} contains only benign traffic, sampling {sample_size * 2} benign samples")
                    if len(benign_df) >= sample_size * 2:
                        sample_df = benign_df.sample(n=sample_size * 2, random_state=42)
                    else:
                        sample_df = benign_df
                        self.logger.info(f"  Warning: {file_name} has fewer than {sample_size * 2} benign samples. Using all {len(benign_df)} samples.")
                else:
                    # Sample 25k benign
                    if len(benign_df) >= sample_size:
                        benign_sample = benign_df.sample(n=sample_size, random_state=42)
                    else:
                        benign_sample = benign_df
                        self.logger.info(f"  Warning: {file_name} has fewer than {sample_size} benign samples. Using all {len(benign_df)} samples.")
                    
                    # Sample 25k malicious
                    if len(malicious_df) >= sample_size:
                        malicious_sample = malicious_df.sample(n=sample_size, random_state=42)
                    else:
                        malicious_sample = malicious_df
                        self.logger.info(f"  Warning: {file_name} has fewer than {sample_size} malicious samples. Using all {len(malicious_df)} samples.")
                    
                    # Combine samples
                    sample_df = pd.concat([benign_sample, malicious_sample])
                
                # Store original indices and labels
                sample_indices = sample_df.index.tolist()
                sample_labels = sample_df['Label'].tolist()
                
                # Process in batches
                all_embeddings = []
                
                for i in tqdm(range(0, len(sample_df), batch_size), desc=f"Extracting embeddings from {file_name}"):
                    batch = sample_df.iloc[i:i+batch_size]
                    batch_texts = [self.format_input(row) for _, row in batch.iterrows()]
                    batch_embeddings = self.extract_embeddings_from_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                
                # Store results for this file
                results[file_name] = {
                    'indices': sample_indices,
                    'labels': sample_labels,
                    'embeddings': all_embeddings,
                    'file_path': file_path
                }
                
                self.logger.info(f"  Successfully extracted {len(all_embeddings)} embeddings from {file_name}")
                
                # Save embeddings for this specific day
                output_path = os.path.join(self.output_dir, f"embeddings_{file_name.replace('.csv', '')}.npy")
                np.save(output_path, np.array(all_embeddings))
                
                # Save metadata separately for this day
                metadata_df = pd.DataFrame({
                    'index': sample_indices,
                    'label': sample_labels,
                    'file': file_name
                })
                metadata_path = os.path.join(self.output_dir, f"metadata_{file_name}")
                metadata_df.to_csv(metadata_path, index=False)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return results

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Extract embeddings from Llama model for network traffic data")
    parser.add_argument("--data_dir", default="./",  help="Directory containing dataset files")
    parser.add_argument("--output_dir", default="finetuned_B-B-embeddings-weekdays-25k", help="Directory to save embeddings and metadata")
    parser.add_argument("--sample_size", type=int, default=25000, help="Number of samples per category (benign/malicious)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--model_name", default="cicids_finetuned/checkpoint-585/", help="Name or path of the model")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to extract embeddings from (-1 for last layer)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting embedding extraction with model {args.model_name}")
    logger.info(f"Processing weekday files in directory {args.data_dir}")
    logger.info(f"Extracting {args.sample_size} samples per category (benign/malicious) with batch size {args.batch_size}")
    logger.info(f"Saving results to {args.output_dir}")
    
    # Initialize the embedding extractor
    extractor = EmbeddingExtractor(
        model_name=args.model_name,
        layer=args.layer,
        output_dir=args.output_dir
    )
    
    # Process all dataset files
    results = extractor.process_dataset_files(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )
    
    # Save a summary file
    summary_path = os.path.join(args.output_dir, "extraction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Embedding Extraction Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n\n")
        f.write(f"Processing parameters:\n")
        f.write(f"- Sample size per category: {args.sample_size}\n")
        f.write(f"- Batch size: {args.batch_size}\n\n")
        f.write(f"Files processed:\n")
        for file_name, data in results.items():
            f.write(f"- {file_name}: {len(data['embeddings'])} embeddings extracted\n")
    
    logger.info(f"Finished processing. Summary saved to: {summary_path}")
    logger.info(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
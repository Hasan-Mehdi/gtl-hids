import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import TrainingArguments
from trl import SFTTrainer
import logging
from typing import List, Dict
import json
from sklearn.model_selection import train_test_split

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def split_csv_file(input_file, train_output, test_output, test_size=0.3, random_state=42):
    """
    Split a CSV file into training and testing sets.
    Args:
        input_file (str): Path to the input CSV file
        train_output (str): Path to save the training set
        test_output (str): Path to save the testing set
        test_size (float): Proportion of data to include in the test set (default: 0.3)
        random_state (int): Random seed for reproducibility
    """
    logger = setup_logging()
    logger.info(f"Loading data from {input_file}...")
    try:
        # Try to read the CSV file with comma delimiter
        df = pd.read_csv(input_file, sep=',')
        # If there's only one column and it contains commas, the file might be using a different delimiter
        if len(df.columns) == 1 and ',' in str(df.iloc[0, 0]):
            logger.info("Detected possible comma-separated values in a tab-delimited file...")
            # Try reading with comma delimiter
            df = pd.read_csv(input_file, sep=',')
    except Exception as e:
        logger.error(f"Error reading with standard delimiters: {e}")
        logger.info("Trying to read with custom parsing...")
        # Read the file as text and parse manually
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Find the delimiter by inspecting the first line
        first_line = lines[0].strip()
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        else:
            logger.warning("Could not determine delimiter. Defaulting to comma.")
            delimiter = ','
        # Parse the header
        header = lines[0].strip().split(delimiter)
        # Parse the data
        data = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                values = line.strip().split(delimiter)
                if len(values) == len(header):  # Only add rows with correct number of columns
                    data.append(values)
                else:
                    logger.warning(f"Skipping row with {len(values)} values (header has {len(header)} columns)")
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    
    # Split the data
    logger.info(f"Splitting data into {100-test_size*100}% training and {test_size*100}% testing...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Testing set shape: {test_df.shape}")
    
    # Save the split datasets
    logger.info(f"Saving training set to {train_output}...")
    train_df.to_csv(train_output, index=False)
    
    logger.info(f"Saving testing set to {test_output}...")
    test_df.to_csv(test_output, index=False)
    
    logger.info("Files saved successfully!")
    return train_df, test_df

def sample_balanced_dataset(df, max_samples_per_class=25000):
    """Sample a balanced subset of the data (50% benign, 50% malicious)"""
    sampled_dfs = []
    
    # Handle 'benign' class separately
    benign_df = df[df['Label'].str.upper() == 'BENIGN']
    attack_df = df[df['Label'].str.upper() != 'BENIGN']
    
    # Sample benign data
    if len(benign_df) > max_samples_per_class:
        benign_sampled = benign_df.sample(n=max_samples_per_class, random_state=42)
        sampled_dfs.append(benign_sampled)
    else:
        sampled_dfs.append(benign_df)
    
    # Sample malicious data (combine all attack types)
    if len(attack_df) > max_samples_per_class:
        attack_sampled = attack_df.sample(n=max_samples_per_class, random_state=42)
        sampled_dfs.append(attack_sampled)
    else:
        sampled_dfs.append(attack_df)
    
    return pd.concat(sampled_dfs, ignore_index=True)

class NetworkFlowDataProcessor:
    def __init__(self):
        self.logger = setup_logging()
        self.scaler = MinMaxScaler()
        
        # Map columns from second dataset to first dataset format where needed
        self.numerical_features = [
            'Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
            'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 
            'Total Fwd Packet', 'Total Bwd packets',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
        ]
        self.categorical_features = ['Src IP', 'Dst IP']
        
    def process_ip_address(self, ip: str) -> str:
        """Convert IP address to a more descriptive format"""
        if pd.isna(ip) or not isinstance(ip, str):
            return "unknown_ip"
        
        parts = ip.split('.')
        if len(parts) != 4:
            return f"invalid_ip_{ip}"
            
        if parts[0] == '192' and parts[1] == '168':
            return f"internal_network_{parts[2]}_{parts[3]}"
        elif parts[0] == '172' and parts[1] == '16':
            return f"internal_network_{parts[2]}_{parts[3]}"
        return f"external_network_{ip}"

    def format_flow_data(self, row: pd.Series) -> str:
        """Format network flow data into a descriptive text"""
        # Get TCP flags (use individual flag counts if available)
        tcp_flags = row.get('TCP Flags Count', 0)
        if pd.isna(tcp_flags):
            tcp_flags = sum([
                row.get('FIN Flag Count', 0) or 0,
                row.get('SYN Flag Count', 0) or 0,
                row.get('RST Flag Count', 0) or 0,
                row.get('PSH Flag Count', 0) or 0,
                row.get('ACK Flag Count', 0) or 0,
                row.get('URG Flag Count', 0) or 0
            ])
        
        return f"""Network Flow Description:
Source: {self.process_ip_address(row['Src IP'])} (Port: {row['Src Port']})
Destination: {self.process_ip_address(row['Dst IP'])} (Port: {row['Dst Port']})
Protocol Information:
- Protocol ID: {row['Protocol']}
- TCP Flags: {tcp_flags}
Traffic Metrics:
- Bytes: {row['Total Length of Fwd Packet']} inbound, {row['Total Length of Bwd Packet']} outbound
- Packets: {row['Total Fwd Packet']} inbound, {row['Total Bwd packets']} outbound
- Duration: {row['Flow Duration']} milliseconds"""

    def get_attack_description(self, attack_type: str) -> str:
        """Get detailed description of attack type"""
        descriptions = {
            "benign": "This is normal network traffic with no malicious intent.",
            "ftp-patator": "An attack using brute force methods to guess FTP credentials.",
            "ssh-patator": "An attack using brute force methods to guess SSH credentials.",
            "dos slowloris": "A DoS attack that holds connections open by sending partial HTTP requests.",
            "dos slowhttptest": "A DoS attack exploiting HTTP headers to keep connections open.",
            "dos hulk": "A DoS attack sending high-volume HTTP GET requests to overwhelm servers.",
            "dos goldeneye": "A DoS attack targeting HTTP servers with multiple concurrent sessions.",
            "heartbleed": "An attack exploiting a vulnerability in the OpenSSL library.",
            "web attack - brute force": "An attack attempting to gain access to web services through password guessing.",
            "web attack - xss": "A Cross-Site Scripting attack injecting malicious code into web pages.",
            "web attack - sql injection": "An attack attempting to execute SQL commands through web inputs.",
            "infiltration": "Malicious activity attempting to penetrate network defenses.",
            "botnet": "Traffic indicating command and control communication with compromised devices.",
            "portscan": "Reconnaissance activity scanning for open network ports and services.",
            "ddos": "A Distributed Denial of Service attack from multiple sources.",
            # Add generic fallbacks for any other categories
            "dos": "A Denial of Service attack targeting system availability.",
            "web attack": "An attack targeting web application vulnerabilities."
        }
        
        # Convert to lowercase for case-insensitive matching
        attack_lower = attack_type.lower()
        
        # Try to find the most specific match
        for key, description in descriptions.items():
            if attack_lower == key:
                return description
                
        # If no exact match, try partial matches
        for key, description in descriptions.items():
            if key in attack_lower:
                return description
                
        # Default description for unknown attacks
        return f"A malicious network activity classified as {attack_type}."

    def prepare_training_text(self, row: pd.Series) -> str:
        """Prepare single training example in LLaMA-3 chat format"""
        flow_text = self.format_flow_data(row)
        
        # Determine if traffic is benign or malicious
        is_benign = row['Label'].upper() == 'BENIGN'
        attack_type = 'benign' if is_benign else row['Label'].lower()
        attack_desc = self.get_attack_description(attack_type)
        
        # Get traffic volume
        total_bytes = (row['Total Length of Fwd Packet'] or 0) + (row['Total Length of Bwd Packet'] or 0)
        
        # Simplify TCP flags display
        tcp_flags = row.get('TCP Flags Count', 
                          sum([
                              row.get('FIN Flag Count', 0) or 0,
                              row.get('SYN Flag Count', 0) or 0,
                              row.get('RST Flag Count', 0) or 0,
                              row.get('PSH Flag Count', 0) or 0,
                              row.get('ACK Flag Count', 0) or 0,
                              row.get('URG Flag Count', 0) or 0
                          ]))
        
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Analyze this network flow for potential security threats:

{flow_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
This network flow is classified as {attack_type}. {attack_desc}

Key indicators from the flow data:
- Traffic volume: {total_bytes} total bytes
- Flow duration: {row['Flow Duration']} ms
- Protocol behavior: {tcp_flags} TCP flags<|eot_id|>"""

def load_and_process_data(train_path: str, processor: NetworkFlowDataProcessor, max_samples=50000):
    """Load and process the training data"""
    logger = setup_logging()
    logger.info(f"Loading data from {train_path}")
    
    # Read CSV file
    df = pd.read_csv(train_path)
    logger.info(f"Original dataset size: {len(df)}")
    
    # Sample balanced dataset (50% benign, 50% malicious)
    df = sample_balanced_dataset(df, max_samples_per_class=max_samples//2)
    logger.info(f"Sampled dataset size: {len(df)}")
    
    # Log distribution of attack types
    attack_counts = df['Label'].value_counts()
    logger.info(f"Attack type distribution:\n{attack_counts}")
    
    # Handle missing values in numerical features
    for col in processor.numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Create dataset with text field
    texts = []
    for _, row in df.iterrows():
        try:
            text = processor.prepare_training_text(row)
            texts.append(text)
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            continue
    
    logger.info(f"Created {len(texts)} training examples")
    dataset = Dataset.from_pandas(pd.DataFrame({'text': texts}))
    
    return dataset

def main():
    logger = setup_logging()
    
    # Define file paths
    input_file = "finetune_dataset_A.csv"
    train_output = "finetune_train_A.csv"
    test_output = "finetune_test_A.csv"
    
    # Create train-test split
    logger.info("Creating train-test split...")
    split_csv_file(input_file, train_output, test_output, test_size=0.3, random_state=42)
    
    # Initialize data processor
    processor = NetworkFlowDataProcessor()
    
    # Load and process data (using only the train split)
    train_dataset = load_and_process_data(train_output, processor, max_samples=50000)
    
    # Model initialization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # Configure tokenizer for LLaMA-3
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="cicids_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,  # Use bfloat16 for A100
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    model.save_pretrained("cicids_finetuned")
    tokenizer.save_pretrained("cicids_finetuned")

if __name__ == "__main__":
    main()
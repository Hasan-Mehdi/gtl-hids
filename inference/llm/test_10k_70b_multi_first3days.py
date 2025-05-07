
import os
# Set specific GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import os
import random

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def format_flow_prompt(row):
    """Format a single network flow into a prompt for the model"""
    
    # Enhanced system prompt with examples of all attack types
    system_prompt = system_prompt = """You are a cybersecurity expert analyzing network flows from the CIC-IDS-2017 dataset to identify specific security threats.

Here are examples of different network traffic patterns:

BENIGN EXAMPLE:
Source IP: 8.6.0.1 (Port: 0)
Destination IP: 8.0.6.4 (Port: 0)
Protocol: 0
Traffic Volume: 0 bytes in, 0 bytes out
Packets: 231 packets in, 0 packets out
TCP Flags: 0
Duration: 119719148 ms
This is BENIGN traffic with normal communication patterns and no suspicious indicators.

FTP-Patator EXAMPLE:
Source IP: 172.16.0.1 (Port: 52146) 
Destination IP: 192.168.10.50 (Port: 21)
Protocol: 6
Traffic Volume: 30 bytes in, 76 bytes out
Packets: 6 packets in, 6 packets out
TCP Flags: Not specified
Duration: 4008190 ms
This is FTP-Patator traffic showing FTP brute force password attack patterns.

FTP-Patator - Attempted EXAMPLE:
Source IP: 172.16.0.1 (Port: 52108)
Destination IP: 192.168.10.50 (Port: 21)
Protocol: 6
Traffic Volume: 31 bytes in, 96 bytes out
Packets: 9 packets in, 8 packets out
TCP Flags: Not specified
Duration: 4475174 ms
This is FTP-Patator - Attempted traffic showing unsuccessful FTP brute force attempts.

SSH-Patator EXAMPLE:
Source IP: 172.16.0.1 (Port: 52032)
Destination IP: 192.168.10.50 (Port: 22)
Protocol: 6
Traffic Volume: 1304 bytes in, 2153 bytes out
Packets: 14 packets in, 15 packets out
TCP Flags: Not specified
Duration: 4755497 ms
This is SSH-Patator traffic showing SSH brute force password attack patterns.

SSH-Patator - Attempted EXAMPLE:
Source IP: 172.16.0.1 (Port: 52068)
Destination IP: 192.168.10.50 (Port: 22)
Protocol: 6
Traffic Volume: 24 bytes in, 0 bytes out
Packets: 3 packets in, 3 packets out
TCP Flags: Not specified
Duration: 3187 ms
This is SSH-Patator - Attempted traffic showing unsuccessful SSH brute force attempts.

DoS Slowloris EXAMPLE:
Source IP: 172.16.0.1 (Port: 53816)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 239 bytes in, 0 bytes out
Packets: 4 packets in, 3 packets out
TCP Flags: Not specified
Duration: 17072865 ms
This is DoS Slowloris traffic showing DoS attack targeting web servers by slowly opening connections.

DoS Slowloris - Attempted EXAMPLE:
Source IP: 172.16.0.1 (Port: 54116)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 0 bytes in, 0 bytes out
Packets: 3 packets in, 2 packets out
TCP Flags: Not specified
Duration: 5964859 ms
This is DoS Slowloris - Attempted traffic showing unsuccessful slowloris attack attempts.

DoS Slowhttptest EXAMPLE:
Source IP: 172.16.0.1 (Port: 33664)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 5200 bytes in, 0 bytes out
Packets: 18 packets in, 2 packets out
TCP Flags: Not specified
Duration: 83400021 ms
This is DoS Slowhttptest traffic showing slow HTTP POST DoS attack patterns.

DoS Slowhttptest - Attempted EXAMPLE:
Source IP: 172.16.0.1 (Port: 37654)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 0 bytes in, 1964 bytes out
Packets: 0 packets in, 3 packets out
TCP Flags: Not specified
Duration: 52796288 ms
This is DoS Slowhttptest - Attempted traffic showing unsuccessful slow HTTP attack attempts.

DoS Hulk EXAMPLE:
Source IP: 172.16.0.1 (Port: 50750)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 348 bytes in, 11595 bytes out
Packets: 9 packets in, 9 packets out
TCP Flags: Not specified
Duration: 28504 ms
This is DoS Hulk traffic showing HTTP Unbearable Load King DoS attack patterns.

DoS Hulk - Attempted EXAMPLE:
Source IP: 172.16.0.1 (Port: 43664)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 3489 bytes in, 5355 bytes out
Packets: 17 packets in, 12 packets out
TCP Flags: Not specified
Duration: 7078331 ms
This is DoS Hulk - Attempted traffic showing unsuccessful HTTP Unbearable Load King DoS attack attempts.

DoS GoldenEye EXAMPLE:
Source IP: 172.16.0.1 (Port: 33056)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 1776 bytes in, 3525 bytes out
Packets: 9 packets in, 4 packets out
TCP Flags: Not specified
Duration: 11454901 ms
This is DoS GoldenEye traffic targeting HTTP servers.

DoS GoldenEye - Attempted EXAMPLE:
Source IP: 172.16.0.1 (Port: 58480)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 0 bytes in, 0 bytes out
Packets: 1 packets in, 1 packets out
TCP Flags: Not specified
Duration: 2951 ms
This is DoS GoldenEye - Attempted traffic showing unsuccessful GoldenEye DoS attack attempts.

Heartbleed EXAMPLE:
Source IP: 172.16.0.1 (Port: 45022)
Destination IP: 192.168.10.51 (Port: 444)
Protocol: 6
Traffic Volume: 8299 bytes in, 7556917 bytes out
Packets: 2685 packets in, 1729 packets out
TCP Flags: Not specified
Duration: 119302728 ms
This is Heartbleed traffic showing OpenSSL vulnerability exploitation patterns.

You must classify each flow with the specific type of traffic (BENIGN or the specific attack type) based on its characteristics. Respond with only one class label from this list: BENIGN, FTP-Patator, FTP-Patator - Attempted, SSH-Patator, SSH-Patator - Attempted, DoS Slowloris, DoS Slowloris - Attempted, DoS Slowhttptest, DoS Slowhttptest - Attempted, DoS Hulk, DoS Hulk - Attempted, DoS GoldenEye, DoS GoldenEye - Attempted, Heartbleed.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Analyze this network flow:
Source IP: {row['Src IP']} (Port: {row['Src Port']})
Destination IP: {row['Dst IP']} (Port: {row['Dst Port']})
Protocol: {row['Protocol']}
Traffic Volume: {row['Total Length of Fwd Packet']} bytes in, {row['Total Length of Bwd Packet']} bytes out
Packets: {row['Total Fwd Packet']} packets in, {row['Total Bwd packets']} packets out
TCP Flags: {row.get('TCP Flags Count', row.get('FIN Flag Count', 0))}
Duration: {row['Flow Duration']} ms

What specific type of traffic is this? Answer with only the exact class name from the list."""}
    ]
    return messages

def get_attack_labels():
    """Return a list of all possible attack labels with the correct capitalization"""
    return [
        "BENIGN", 
        "FTP-Patator", 
        "FTP-Patator - Attempted", 
        "SSH-Patator", 
        "SSH-Patator - Attempted", 
        "DoS Slowloris", 
        "DoS Slowloris - Attempted", 
        "DoS Slowhttptest", 
        "DoS Slowhttptest - Attempted", 
        "DoS Hulk", 
        "DoS Hulk - Attempted", 
        "DoS GoldenEye", 
        "DoS GoldenEye - Attempted", 
        "Heartbleed"
    ]

class MulticlassClassifier:
    def __init__(self, device="cuda"):
        """Initialize the model and tokenizer"""
        self.logger = setup_logging()
        self.device = device
        model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        
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
            pad_token_id=self.tokenizer.eos_token_id  # Set pad token ID in model
        )
        
        # Store valid attack classes for prediction
        self.valid_classes = get_attack_labels()
        
    def predict_single(self, messages, get_explanation=False):
        """Generate prediction for a single prompt with optional explanation"""
        # Prepare input
        encodings = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Create attention mask
        attention_mask = torch.ones_like(encodings)
        
        # Move to device
        input_ids = encodings.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30 if not get_explanation else 100,  # More tokens for class name
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract response
        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True).strip().upper()
        
        # Return full explanation if requested
        if get_explanation:
            return response_text
        
        # Match to closest valid class
        predicted_class = self._match_response_to_class(response_text)
        return predicted_class
    
    def _match_response_to_class(self, response):
        """Match model response to the closest valid class"""
        response = response.upper()
        
        # Define mappings for class normalization
        class_mappings = {
            # Exact matches
            "BENIGN": "BENIGN",
            # FTP variants
            "FTP-PATATOR": "FTP-Patator",
            "FTP PATATOR": "FTP-Patator",
            "FTP-PATATOR - ATTEMPTED": "FTP-Patator - Attempted",
            "FTP PATATOR ATTEMPTED": "FTP-Patator - Attempted",
            # SSH variants
            "SSH-PATATOR": "SSH-Patator",
            "SSH PATATOR": "SSH-Patator",
            "SSH-PATATOR - ATTEMPTED": "SSH-Patator - Attempted",
            "SSH PATATOR ATTEMPTED": "SSH-Patator - Attempted",
            # DoS Slowloris variants
            "DOS SLOWLORIS": "DoS Slowloris",
            "SLOWLORIS": "DoS Slowloris",
            "DOS SLOWLORIS - ATTEMPTED": "DoS Slowloris - Attempted",
            "SLOWLORIS ATTEMPTED": "DoS Slowloris - Attempted",
            # DoS Slowhttptest variants
            "DOS SLOWHTTPTEST": "DoS Slowhttptest",
            "SLOWHTTPTEST": "DoS Slowhttptest",
            "DOS SLOWHTTPTEST - ATTEMPTED": "DoS Slowhttptest - Attempted",
            "SLOWHTTPTEST ATTEMPTED": "DoS Slowhttptest - Attempted",
            # DoS Hulk variants
            "DOS HULK": "DoS Hulk",
            "HULK": "DoS Hulk",
            "DOS HULK - ATTEMPTED": "DoS Hulk - Attempted",
            "HULK ATTEMPTED": "DoS Hulk - Attempted",
            # DoS GoldenEye variants
            "DOS GOLDENEYE": "DoS GoldenEye",
            "GOLDENEYE": "DoS GoldenEye",
            "DOS GOLDENEYE - ATTEMPTED": "DoS GoldenEye - Attempted",
            "GOLDENEYE ATTEMPTED": "DoS GoldenEye - Attempted",
            # Heartbleed
            "HEARTBLEED": "Heartbleed",
        }
        
        # First try direct matching with our mapping
        for pattern, class_name in class_mappings.items():
            if pattern in response:
                return class_name
        
        # Partial matching for more complex cases
        if "BENIGN" in response:
            return "BENIGN"
        if "FTP" in response and "PATATOR" in response:
            if "ATTEMPT" in response:
                return "FTP-Patator - Attempted"
            return "FTP-Patator"
        if "SSH" in response and "PATATOR" in response:
            if "ATTEMPT" in response:
                return "SSH-Patator - Attempted"
            return "SSH-Patator"
        if "SLOW" in response:
            if "HTTP" in response:
                if "ATTEMPT" in response:
                    return "DoS Slowhttptest - Attempted"
                return "DoS Slowhttptest"
            if "LORIS" in response:
                if "ATTEMPT" in response:
                    return "DoS Slowloris - Attempted"
                return "DoS Slowloris"
        if "HULK" in response:
            if "ATTEMPT" in response:
                return "DoS Hulk - Attempted"
            return "DoS Hulk"
        if "GOLD" in response or "EYE" in response:
            if "ATTEMPT" in response:
                return "DoS GoldenEye - Attempted"
            return "DoS GoldenEye"
        if "HEART" in response or "BLEED" in response:
            return "Heartbleed"
        if "DOS" in response or "DDOS" in response:
            # Generic DoS detection - try to determine the type
            if "ATTEMPT" in response:
                return "DoS Hulk - Attempted"  # Default to most common
            return "DoS Hulk"  # Default to most common
        
        # If nothing matches, return BENIGN as default
        return "BENIGN"
    
    def evaluate_files(self, data_dir, output_dir, target_file="test_thurs_friday.csv", samples_per_file=100):
        """Evaluate the model on a random sample from each CSV file in the directory"""
        logger = setup_logging()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all CSV files but filter for the target file
        csv_files = [f for f in os.listdir(data_dir) if f == target_file]
        
        if not csv_files:
            logger.warning(f"Target file {target_file} not found in {data_dir}")
            return {}
        
        all_results = {}
        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            logger.info(f"Processing {file_path}")
            
            try:
                # Read the full CSV to get a random sample
                df = pd.read_csv(file_path)
                
                # Take random sample
                if len(df) > samples_per_file:
                    sample_df = df.sample(n=samples_per_file, random_state=42)
                else:
                    sample_df = df
                    logger.info(f"File {file_name} has fewer than {samples_per_file} samples. Using all {len(df)} samples.")
                
                # Get explanations for a few samples
                explanations = []
                # Get 3 samples of different types if available
                class_samples = sample_df.groupby('Label').apply(lambda x: x.sample(min(3, len(x)), random_state=42))
                if isinstance(class_samples, pd.DataFrame):
                    sample_rows = class_samples
                else:
                    sample_rows = class_samples.reset_index(level=0, drop=True)
                
                for _, row in sample_rows.iterrows():
                    try:
                        prompt = format_flow_prompt(row)
                        explanation = self.predict_single(prompt, get_explanation=True)
                        true_label = row['Label']
                        explanations.append({
                            "true_label": true_label,
                            "predicted": explanation,
                            "data": {k: str(row[k]) for k in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 
                                                        'Protocol', 'Total Fwd Packet', 
                                                        'Total Length of Fwd Packet', 'Label']}
                        })
                    except Exception as e:
                        logger.error(f"Error getting explanation: {e}")
                
                # Save explanations to file
                with open(output_path / f"{file_name.replace('.csv', '')}_explanations_10000_multiclass.json", 'w') as f:
                    json.dump(explanations, f, indent=4)
                
                # Log a few explanations
                logger.info(f"Sample explanations for {file_name}:")
                for i, exp in enumerate(explanations):
                    if i < 4:  # Just log a few for console output
                        logger.info(f"True label: {exp['true_label']}")
                        logger.info(f"Model output: {exp['predicted']}")
                        logger.info("---")
                
                # Continue with original predictions for metrics calculation
                predictions = []
                true_labels = []
                
                for _, row in tqdm(sample_df.iterrows(), desc=f"Evaluating {file_name}", total=len(sample_df)):
                    try:
                        prompt = format_flow_prompt(row)
                        pred = self.predict_single(prompt)
                        predictions.append(pred)
                        
                        # Normalize true label to match our prediction categories
                        true_label = row['Label']
                        if "Web Attack" in true_label:
                            if "Brute Force" in true_label:
                                true_label = "WEB ATTACK - BRUTE FORCE"
                            elif "XSS" in true_label:
                                true_label = "WEB ATTACK - XSS"
                            elif "SQL Injection" in true_label:
                                true_label = "WEB ATTACK - SQL INJECTION"
                        
                        true_labels.append(true_label)
                        
                    except Exception as e:
                        logger.error(f"Error processing sample: {e}")
                        predictions.append("BENIGN")  # Default to benign on error
                        true_labels.append(row['Label'])
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels, predictions)
                report = classification_report(true_labels, predictions)
                
                # Generate confusion matrix
                classes = list(set(true_labels + predictions))
                cm = confusion_matrix(true_labels, predictions, labels=classes)
                cm_dict = {
                    'matrix': cm.tolist(),
                    'classes': classes
                }
                
                # Calculate attack-specific metrics
                attack_metrics = {}
                attack_types = list(set(true_labels))
                
                # Calculate metrics for each attack type
                for attack in attack_types:
                    # Get indices for this attack type
                    indices = [i for i, label in enumerate(true_labels) if label == attack]
                    if not indices:
                        continue
                    
                    # Get predictions for this attack type
                    attack_preds = [predictions[i] for i in indices]
                    attack_true = [attack] * len(indices)
                    
                    # For multiclass, we use accuracy and F1 score
                    accuracy = sum(1 for p, t in zip(attack_preds, attack_true) if p == t) / len(indices)
                    
                    # Calculate precision, recall, F1 using one-vs-rest approach
                    tp = sum(1 for p, t in zip(attack_preds, attack_true) if p == t and p == attack)
                    fp = sum(1 for p, t in zip(attack_preds, attack_true) if p == attack and t != attack)
                    fn = sum(1 for p, t in zip(attack_preds, attack_true) if p != attack and t == attack)
                    
                    # Calculate precision, recall, F1
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    attack_metrics[attack] = {
                        'count': len(indices),
                        'true_positive': tp,
                        'false_positive': fp,
                        'false_negative': fn,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'accuracy': float(accuracy),
                        'detection_rate': float(recall)  # Detection rate is the same as recall
                    }
                
                # Calculate weighted F1 score for all attack types
                attack_weights = {}
                attack_f1_scores = {}
                total_attack_samples = 0
                
                # Get sample counts for each attack type
                for attack, metrics in attack_metrics.items():
                    attack_weights[attack] = metrics['count']
                    attack_f1_scores[attack] = metrics.get('f1_score', 0)
                    total_attack_samples += metrics['count']
                
                # Calculate weighted F1 score
                weighted_f1 = 0
                if total_attack_samples > 0:
                    for attack, count in attack_weights.items():
                        weight = count / total_attack_samples
                        weighted_f1 += weight * attack_f1_scores[attack]
                
                # Calculate macro F1 (unweighted average)
                macro_f1 = sum(attack_f1_scores.values()) / len(attack_f1_scores) if attack_f1_scores else 0
                
                # Log attack-specific metrics
                logger.info("\nAttack-specific metrics:")
                for attack, metrics in attack_metrics.items():
                    logger.info(f"{attack} (n={metrics['count']}): F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}, Detection Rate={metrics['detection_rate']:.4f}")
                
                logger.info(f"Weighted F1 Score: {weighted_f1:.4f}")
                logger.info(f"Macro F1 Score: {macro_f1:.4f}")
                
                logger.info(f"Results for {file_name}:")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Classification Report:\n{report}")
                
                # Store results
                file_results = {
                    'accuracy': float(accuracy),
                    'classification_report': report,
                    'total_samples': len(sample_df),
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'confusion_matrix': cm_dict,
                    'class_distribution': {
                        'true': dict(pd.Series(true_labels).value_counts().to_dict()),
                        'predicted': dict(pd.Series(predictions).value_counts().to_dict())
                    },
                    'attack_specific_metrics': attack_metrics,
                    'weighted_f1_score': float(weighted_f1),
                    'macro_f1_score': float(macro_f1)
                }
                
                all_results[file_name] = file_results
                
                # Save individual file results
                with open(output_path / f"{file_name.replace('.csv', '')}_results_10000_multiclass.json", 'w') as f:
                    json.dump(file_results, f, indent=4)
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Save combined results
        with open(output_path / "all_results_10000_multiclass.json", 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Multiclass testing of LLaMA 3.1 on CIC-IDS-2017 dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--target_file", default="test_thurs_friday.csv", 
                        help="Target file to evaluate")
    parser.add_argument("--samples_per_file", type=int, default=100, 
                       help="Number of random samples to test from each file")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Initialize classifier
    classifier = MulticlassClassifier()
    
    try:
        # Evaluate files
        results = classifier.evaluate_files(
            args.data_dir,
            args.output_dir,
            target_file=args.target_file,
            samples_per_file=args.samples_per_file
        )
        
        # Aggregate metrics across all files
        if results:
            total_samples = sum(r['total_samples'] for r in results.values())
            total_accuracy = sum(r['accuracy'] * r['total_samples'] for r in results.values()) / total_samples
            
            # Create aggregated class distribution
            all_true_labels = []
            all_predictions = []
            for r in results.values():
                all_true_labels.extend(r['true_labels'])
                all_predictions.extend(r['predictions'])
            
            aggregated_report = classification_report(all_true_labels, all_predictions)
            
            # Calculate aggregated attack-specific metrics
            all_attack_metrics = {}
            
            # Collect all attack types across all files
            all_attack_types = set()
            for file_results in results.values():
                if 'attack_specific_metrics' in file_results:
                    all_attack_types.update(file_results['attack_specific_metrics'].keys())
            
            # Initialize metrics for each attack type
            for attack in all_attack_types:
                all_attack_metrics[attack] = {
                    'total_count': 0,
                    'total_tp': 0,
                    'total_fp': 0,
                    'total_fn': 0,
                }
            
            # Aggregate metrics
            for file_results in results.values():
                if 'attack_specific_metrics' in file_results:
                    for attack, metrics in file_results['attack_specific_metrics'].items():
                        all_attack_metrics[attack]['total_count'] += metrics['count']
                        all_attack_metrics[attack]['total_tp'] += metrics['true_positive']
                        all_attack_metrics[attack]['total_fp'] += metrics['false_positive']
                        all_attack_metrics[attack]['total_fn'] += metrics['false_negative']
            
            # Calculate aggregate metrics for each attack
            for attack, metrics in all_attack_metrics.items():
                # Calculate precision, recall, F1
                precision = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fp']) if (metrics['total_tp'] + metrics['total_fp']) > 0 else 0
                recall = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fn']) if (metrics['total_tp'] + metrics['total_fn']) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                all_attack_metrics[attack]['precision'] = float(precision)
                all_attack_metrics[attack]['recall'] = float(recall)
                all_attack_metrics[attack]['f1_score'] = float(f1)
                all_attack_metrics[attack]['detection_rate'] = float(recall)
            
            # Calculate overall weighted F1 score across all attack types
            attack_weights = {}
            attack_f1_scores = {}
            total_attack_samples = 0
            
            # Get sample counts for each attack type
            for attack, metrics in all_attack_metrics.items():
                attack_weights[attack] = metrics['total_count']
                attack_f1_scores[attack] = metrics.get('f1_score', 0)
                total_attack_samples += metrics['total_count']
            
            # Calculate weighted F1 score
            weighted_f1 = 0
            if total_attack_samples > 0:
                for attack, count in attack_weights.items():
                    weight = count / total_attack_samples
                    weighted_f1 += weight * attack_f1_scores[attack]
            
            # Calculate macro F1 score (simple average)
            macro_f1 = sum(attack_f1_scores.values()) / len(attack_f1_scores) if attack_f1_scores else 0
            
            # Print overall results first
            logger.info("\n" + "="*50)
            logger.info("OVERALL RESULTS:")
            logger.info("="*50)
            logger.info(f"Average Accuracy: {total_accuracy:.4f}")
            logger.info(f"Number of files processed: {len(results)}")
            logger.info(f"Total samples analyzed: {total_samples}")
            logger.info(f"Overall Weighted F1 Score: {weighted_f1:.4f}")
            logger.info(f"Overall Macro F1 Score: {macro_f1:.4f}")
            logger.info(f"Aggregated Classification Report:\n{aggregated_report}")
            
            # Print attack-specific metrics
            logger.info("\n" + "="*50)
            logger.info("ATTACK-SPECIFIC METRICS:")
            logger.info("="*50)
            
            # Sort attacks by count for better readability (higher counts first)
            sorted_attacks = sorted(
                [(attack, metrics) for attack, metrics in all_attack_metrics.items()],
                key=lambda x: x[1]['total_count'],
                reverse=True
            )
            
            # Print a header for the attack metrics table
            logger.info(f"{'Attack Type':<30} {'Count':<8} {'Precision':<12} {'Recall':<10} {'F1 Score':<10}")
            logger.info("-" * 80)
            
            # Print each attack's metrics
            for attack, metrics in sorted_attacks:
                logger.info(f"{attack:<30} {metrics['total_count']:<8} {metrics.get('precision', 0):.4f}{'':<5} {metrics.get('recall', 0):.4f}{'':<3} {metrics.get('f1_score', 0):.4f}")
            
            # Save overall results
            overall_results = {
                'average_accuracy': float(total_accuracy),
                'total_samples': total_samples,
                'aggregated_classification_report': aggregated_report,
                'aggregated_class_distribution': {
                    'true': dict(pd.Series(all_true_labels).value_counts().to_dict()),
                    'predicted': dict(pd.Series(all_predictions).value_counts().to_dict())
                },
                'files_processed': len(results),
                'attack_specific_metrics': all_attack_metrics,
                'weighted_f1_score': float(weighted_f1),
                'macro_f1_score': float(macro_f1)
            }
            
            with open(Path(args.output_dir) / "overall_results_10000_multiclass.json", 'w') as f:
                json.dump(overall_results, f, indent=4)
        else:
            logger.warning("No results to aggregate.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
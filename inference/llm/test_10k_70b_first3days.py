import os

# Set specific GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import json
import random

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def format_flow_prompt(row):
    """Format a single network flow into a prompt for the model"""
    
    # Enhanced system prompt with examples of only relevant attack types
    system_prompt = """You are a cybersecurity expert analyzing network flows from the CIC-IDS-2017 dataset to identify potential security threats.

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

FTP-PATATOR ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 52146) 
Destination IP: 192.168.10.50 (Port: 21)
Protocol: 6
Traffic Volume: 30 bytes in, 76 bytes out
Packets: 6 packets in, 6 packets out
TCP Flags: Not specified
Duration: 4008190 ms
This is MALICIOUS traffic showing FTP brute force password attack patterns.

SSH-PATATOR ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 52032)
Destination IP: 192.168.10.50 (Port: 22)
Protocol: 6
Traffic Volume: 1304 bytes in, 2153 bytes out
Packets: 14 packets in, 15 packets out
TCP Flags: Not specified
Duration: 4755497 ms
This is MALICIOUS traffic showing SSH brute force password attack patterns.

DoS SLOWLORIS ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 53816)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 239 bytes in, 0 bytes out
Packets: 4 packets in, 3 packets out
TCP Flags: Not specified
Duration: 17072865 ms
This is MALICIOUS traffic showing DoS attack targeting web servers by slowly opening connections.

DoS SLOWHTTPTEST ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 33664)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 5200 bytes in, 0 bytes out
Packets: 18 packets in, 2 packets out
TCP Flags: Not specified
Duration: 83400021 ms
This is MALICIOUS traffic showing slow HTTP POST DoS attack patterns.

DoS HULK ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 50750)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 348 bytes in, 11595 bytes out
Packets: 9 packets in, 9 packets out
TCP Flags: Not specified
Duration: 28504 ms
This is MALICIOUS traffic showing HTTP Unbearable Load King DoS attack patterns.

DoS GOLDENEYE ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 33056)
Destination IP: 192.168.10.50 (Port: 80)
Protocol: 6
Traffic Volume: 1776 bytes in, 3525 bytes out
Packets: 9 packets in, 4 packets out
TCP Flags: Not specified
Duration: 11454901 ms
This is MALICIOUS traffic showing DoS attack targeting HTTP servers.

HEARTBLEED ATTACK EXAMPLE:
Source IP: 172.16.0.1 (Port: 45022)
Destination IP: 192.168.10.51 (Port: 444)
Protocol: 6
Traffic Volume: 8299 bytes in, 7556917 bytes out
Packets: 2685 packets in, 1729 packets out
TCP Flags: Not specified
Duration: 119302728 ms
This is MALICIOUS traffic showing OpenSSL vulnerability exploitation patterns.

You must classify each flow as either malicious or benign based on its characteristics. Respond with only the word 'malicious' or 'benign'.
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

Is this network flow malicious or benign? Answer with only one word."""}
    ]
    return messages

class ZeroShotTester:
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
                max_new_tokens=5 if not get_explanation else 100,  # More tokens for explanation
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract response
        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True).strip().lower()
        
        # Return full explanation if requested
        if get_explanation:
            return response_text
        
        # Classify output
        return 1 if 'malicious' in response_text else 0
    
    def evaluate_files(self, data_dir, output_dir, target_file="balanced_10k_test.csv", samples_per_file=100):
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
                
                # Convert labels to binary (1 for any attack, 0 for benign)
                sample_df['binary_label'] = sample_df['Label'].str.upper().apply(
                    lambda x: 0 if x == 'BENIGN' else 1
                )
                
                # Get explanations for a few samples
                explanations = []
                # Get 3 benign and 3 malicious samples if available
                benign_samples = sample_df[sample_df['binary_label'] == 0].head(3)
                malicious_samples = sample_df[sample_df['binary_label'] == 1].head(3)
                
                for _, row in pd.concat([benign_samples, malicious_samples]).iterrows():
                    try:
                        prompt = format_flow_prompt(row)
                        explanation = self.predict_single(prompt, get_explanation=True)
                        true_label = "malicious" if row['binary_label'] == 1 else "benign"
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
                with open(output_path / f"{file_name.replace('.csv', '')}_explanations_10000_70b.json", 'w') as f:
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
                for _, row in tqdm(sample_df.iterrows(), desc=f"Evaluating {file_name}", total=len(sample_df)):
                    try:
                        prompt = format_flow_prompt(row)
                        pred = self.predict_single(prompt)
                        predictions.append(pred)
                    except Exception as e:
                        logger.error(f"Error processing sample: {e}")
                        predictions.append(0)  # Default to benign on error
                
                # Calculate metrics
                true_labels = sample_df['binary_label'].tolist()
                
                # Get attack-specific statistics
                attack_types = sample_df['Label'].unique()
                attack_types = [attack for attack in attack_types if attack.upper() != 'BENIGN']
                
                # Calculate metrics for each attack type
                attack_metrics = {}
                for attack in attack_types:
                    # Get indices for this attack type
                    attack_indices = sample_df[sample_df['Label'] == attack].index
                    
                    # Get predictions and true labels for this attack
                    attack_preds = [predictions[sample_df.index.get_loc(idx)] for idx in attack_indices]
                    
                    # Calculate F1 score and other metrics for this attack type
                    # For attacks, true positive = predicted as malicious (1)
                    tp = sum(1 for p in attack_preds if p == 1)
                    fn = sum(1 for p in attack_preds if p == 0)
                    
                    # Calculate precision, recall, F1
                    precision = tp / len(attack_preds) if len(attack_preds) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    attack_metrics[attack] = {
                        'count': len(attack_indices),
                        'true_positive': tp,
                        'false_negative': fn,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'detection_rate': float(tp / len(attack_indices)) if len(attack_indices) > 0 else 0
                    }
                
                # Also calculate metrics for BENIGN traffic
                benign_indices = sample_df[sample_df['Label'].str.upper() == 'BENIGN'].index
                if len(benign_indices) > 0:
                    benign_preds = [predictions[sample_df.index.get_loc(idx)] for idx in benign_indices]
                    
                    # For benign, true negative = predicted as benign (0)
                    tn = sum(1 for p in benign_preds if p == 0)
                    fp = sum(1 for p in benign_preds if p == 1)
                    
                    # Calculate metrics
                    specificity = tn / len(benign_indices) if len(benign_indices) > 0 else 0
                    false_alarm_rate = fp / len(benign_indices) if len(benign_indices) > 0 else 0
                    
                    attack_metrics['BENIGN'] = {
                        'count': len(benign_indices),
                        'true_negative': tn,
                        'false_positive': fp,
                        'specificity': float(specificity),
                        'false_alarm_rate': float(false_alarm_rate)
                    }
                
                # Calculate weighted F1 score for this file
                attack_weights = {}
                attack_f1_scores = {}
                total_attack_samples = 0
                
                # Get sample counts for each attack type
                for attack, metrics in attack_metrics.items():
                    if attack.upper() != 'BENIGN':
                        attack_weights[attack] = metrics['count']
                        attack_f1_scores[attack] = metrics.get('f1_score', 0)
                        total_attack_samples += metrics['count']
                
                # Calculate weighted F1 score for this file
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
                    if attack.upper() != 'BENIGN':
                        logger.info(f"{attack} (n={metrics['count']}): F1={metrics['f1_score']:.4f}, Detection Rate={metrics['detection_rate']:.4f}")
                    else:
                        logger.info(f"{attack} (n={metrics['count']}): Specificity={metrics['specificity']:.4f}, False Alarm Rate={metrics['false_alarm_rate']:.4f}")
                
                logger.info(f"Weighted F1 Score: {weighted_f1:.4f}")
                logger.info(f"Macro F1 Score: {macro_f1:.4f}")
                
                # Calculate standard metrics
                accuracy = accuracy_score(true_labels, predictions)
                try:
                    auc = roc_auc_score(true_labels, predictions)
                    auc_str = f"{auc:.4f}"
                except Exception as e:
                    logger.warning(f"Could not calculate AUC for {file_name}: {e}")
                    auc = None
                    auc_str = "N/A"
                
                # Calculate confusion matrix metrics
                tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
                tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
                fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
                fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
                
                # Calculate rates
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity, Recall)
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
                
                report = classification_report(true_labels, predictions)
                
                logger.info(f"Results for {file_name}:")
                logger.info(f"Accuracy: {accuracy:.4f}, AUC: {auc_str}")
                logger.info(f"True Positives: {tp}, True Negatives: {tn}")
                logger.info(f"False Positives: {fp}, False Negatives: {fn}")
                logger.info(f"False Positive Rate: {fpr:.4f}, False Negative Rate: {fnr:.4f}")
                logger.info(f"Classification Report:\n{report}")
                
                # Store results
                file_results = {
                    'accuracy': float(accuracy),
                    'auc': float(auc) if auc is not None else None,
                    'classification_report': report,
                    'total_samples': len(sample_df),
                    'predictions': [int(p) for p in predictions],
                    'true_labels': [int(l) for l in true_labels],
                    'confusion_matrix': {
                        'true_positives': tp,
                        'true_negatives': tn,
                        'false_positives': fp,
                        'false_negatives': fn
                    },
                    'rates': {
                        'true_positive_rate': float(tpr),
                        'true_negative_rate': float(tnr),
                        'false_positive_rate': float(fpr),
                        'false_negative_rate': float(fnr)
                    },
                    'attack_specific_metrics': attack_metrics,
                    'weighted_f1_score': float(weighted_f1),
                    'macro_f1_score': float(macro_f1)
                }
                
                all_results[file_name] = file_results
                
                # Save individual file results
                with open(output_path / f"{file_name.replace('.csv', '')}_results_10000_70b.json", 'w') as f:
                    json.dump(file_results, f, indent=4)
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Save combined results
        with open(output_path / "all_results_10000_70b.json", 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Zero-shot testing of LLaMA 3.1 on CIC-IDS-2017 dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--samples_per_file", type=int, default=100, 
                       help="Number of random samples to test from each file")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Initialize tester
    tester = ZeroShotTester()
    
    try:
        # Evaluate files
        results = tester.evaluate_files(
            args.data_dir,
            args.output_dir,
            samples_per_file=args.samples_per_file
        )
        
        # Aggregate metrics across all files
        if results:
            total_samples = sum(r['total_samples'] for r in results.values())
            total_accuracy = sum(r['accuracy'] * r['total_samples'] for r in results.values()) / total_samples
            
            # Aggregate confusion matrix metrics
            total_tp = sum(r['confusion_matrix']['true_positives'] for r in results.values())
            total_tn = sum(r['confusion_matrix']['true_negatives'] for r in results.values())
            total_fp = sum(r['confusion_matrix']['false_positives'] for r in results.values())
            total_fn = sum(r['confusion_matrix']['false_negatives'] for r in results.values())
            
            # Calculate overall rates
            overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
            overall_fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0
            
            # Calculate aggregated attack-specific metrics
            all_attack_metrics = {}
            
            # Collect all attack types across all files
            all_attack_types = set()
            for file_results in results.values():
                if 'attack_specific_metrics' in file_results:
                    all_attack_types.update(file_results['attack_specific_metrics'].keys())
            
            # Initialize metrics for each attack type
            for attack in all_attack_types:
                is_benign = attack.upper() == 'BENIGN'
                all_attack_metrics[attack] = {
                    'total_count': 0,
                    'total_tp': 0 if not is_benign else 0,
                    'total_fn': 0 if not is_benign else 0,
                    'total_tn': 0 if is_benign else 0,
                    'total_fp': 0 if is_benign else 0,
                }
            
            # Aggregate metrics
            for file_results in results.values():
                if 'attack_specific_metrics' in file_results:
                    for attack, metrics in file_results['attack_specific_metrics'].items():
                        all_attack_metrics[attack]['total_count'] += metrics['count']
                        if attack.upper() != 'BENIGN':
                            all_attack_metrics[attack]['total_tp'] += metrics['true_positive']
                            all_attack_metrics[attack]['total_fn'] += metrics['false_negative']
                        else:
                            all_attack_metrics[attack]['total_tn'] += metrics['true_negative']
                            all_attack_metrics[attack]['total_fp'] += metrics['false_positive']
            
            # Calculate aggregate metrics for each attack
            for attack, metrics in all_attack_metrics.items():
                if attack.upper() != 'BENIGN':
                    # Calculate precision, recall, F1
                    precision = metrics['total_tp'] / metrics['total_count'] if metrics['total_count'] > 0 else 0
                    recall = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fn']) if (metrics['total_tp'] + metrics['total_fn']) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    all_attack_metrics[attack]['precision'] = float(precision)
                    all_attack_metrics[attack]['recall'] = float(recall)
                    all_attack_metrics[attack]['f1_score'] = float(f1)
                    all_attack_metrics[attack]['detection_rate'] = float(metrics['total_tp'] / metrics['total_count']) if metrics['total_count'] > 0 else 0
                else:
                    # Calculate specificity, false alarm rate
                    specificity = metrics['total_tn'] / metrics['total_count'] if metrics['total_count'] > 0 else 0
                    false_alarm_rate = metrics['total_fp'] / metrics['total_count'] if metrics['total_count'] > 0 else 0
                    
                    all_attack_metrics[attack]['specificity'] = float(specificity)
                    all_attack_metrics[attack]['false_alarm_rate'] = float(false_alarm_rate)
            
            # Calculate overall weighted F1 score across all attack types (excluding BENIGN)
            attack_weights = {}
            attack_f1_scores = {}
            total_attack_samples = 0
            
            # First get sample counts for each attack type
            for attack, metrics in all_attack_metrics.items():
                if attack.upper() != 'BENIGN':
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
            logger.info(f"Total True Positives: {total_tp}, Total True Negatives: {total_tn}")
            logger.info(f"Total False Positives: {total_fp}, Total False Negatives: {total_fn}")
            logger.info(f"Overall False Positive Rate: {overall_fpr:.4f}")
            logger.info(f"Overall False Negative Rate: {overall_fnr:.4f}")
            logger.info(f"Number of files processed: {len(results)}")
            logger.info(f"Total samples analyzed: {total_samples}")
            logger.info(f"Overall Weighted F1 Score: {weighted_f1:.4f}")
            logger.info(f"Overall Macro F1 Score: {macro_f1:.4f}")
            
            # Print attack-specific metrics
            logger.info("\n" + "="*50)
            logger.info("ATTACK-SPECIFIC METRICS:")
            logger.info("="*50)
            
            # Sort attacks by count for better readability (higher counts first)
            sorted_attacks = sorted(
                [(attack, metrics) for attack, metrics in all_attack_metrics.items() if attack.upper() != 'BENIGN'],
                key=lambda x: x[1]['total_count'],
                reverse=True
            )
            
            # Print a header for the attack metrics table
            logger.info(f"{'Attack Type':<25} {'Count':<8} {'Detection Rate':<15} {'Precision':<12} {'Recall':<10} {'F1 Score':<10}")
            logger.info("-" * 80)
            
            # Print each attack's metrics
            for attack, metrics in sorted_attacks:
                logger.info(f"{attack:<25} {metrics['total_count']:<8} {metrics.get('detection_rate', 0):.4f}{'':<8} {metrics.get('precision', 0):.4f}{'':<5} {metrics.get('recall', 0):.4f}{'':<3} {metrics.get('f1_score', 0):.4f}")
            
            # Print benign metrics separately
            if 'BENIGN' in all_attack_metrics:
                benign = all_attack_metrics['BENIGN']
                logger.info("\n" + "-"*50)
                logger.info(f"BENIGN Traffic (n={benign['total_count']}):")
                logger.info(f"Specificity (True Negative Rate): {benign.get('specificity', 0):.4f}")
                logger.info(f"False Alarm Rate: {benign.get('false_alarm_rate', 0):.4f}")
            
            # Save overall results
            overall_results = {
                'average_accuracy': float(total_accuracy),
                'total_samples': total_samples,
                'confusion_matrix': {
                    'total_true_positives': total_tp,
                    'total_true_negatives': total_tn,
                    'total_false_positives': total_fp,
                    'total_false_negatives': total_fn
                },
                'rates': {
                    'overall_false_positive_rate': float(overall_fpr),
                    'overall_false_negative_rate': float(overall_fnr)
                },
                'files_processed': len(results),
                'attack_specific_metrics': all_attack_metrics,
                'weighted_f1_score': float(weighted_f1),
                'macro_f1_score': float(macro_f1)
            }
            
            with open(Path(args.output_dir) / "overall_results_10000_70b.json", 'w') as f:
                json.dump(overall_results, f, indent=4)
        else:
            logger.warning("No results to aggregate.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
import os
# Set specific GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import os
import re
from unsloth import FastLanguageModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def format_flow_prompt(row):
    """Format a single network flow into a prompt for the model"""
    
    prompt = f"""Analyze this network flow for potential security threats:

Network Flow Description:
Source: {row['Src IP']} (Port: {row['Src Port']})
Destination: {row['Dst IP']} (Port: {row['Dst Port']})
Protocol Information:
- Protocol ID: {row['Protocol']}
- TCP Flags: {row.get('TCP Flags Count', row.get('FIN Flag Count', 0))}
Traffic Metrics:
- Bytes: {row['Total Length of Fwd Packet']} inbound, {row['Total Length of Bwd Packet']} outbound
- Packets: {row['Total Fwd Packet']} packets in, {row['Total Bwd packets']} packets out
- Duration: {row['Flow Duration']} milliseconds"""

    messages = [{"role": "user", "content": prompt}]
    return messages

def get_attack_labels():
    """Return a list of all possible attack labels with the correct capitalization"""
    return [
        "BENIGN", 
        "FTP-Patator", 
        "SSH-Patator", 
        "DoS slowloris", 
        "DoS Slowhttptest", 
        "DoS Hulk", 
        "DoS GoldenEye", 
        "Heartbleed", 
        "Web Attack - Brute Force",
        "Web Attack - XSS", 
        "Web Attack - SQL Injection", 
        "Infiltration", 
        "Botnet", 
        "PortScan", 
        "DDoS"
    ]

class MulticlassModelEvaluator:
    def __init__(self, model_path, device="cuda"):
        """Initialize the model and tokenizer using Unsloth FastLanguageModel"""
        self.logger = setup_logging()
        self.device = device
        
        self.logger.info(f"Loading fine-tuned model from {model_path}...")
        
        # Load model and tokenizer using Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            load_in_4bit=True,
            device_map="auto"
        )
        
        # Enable faster inference
        FastLanguageModel.for_inference(self.model)
        
        # Set up tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Store valid attack classes for prediction
        self.valid_classes = get_attack_labels()
        
        # Create mapping dictionaries for normalization of labels
        self._setup_attack_mappings()
        
    def _setup_attack_mappings(self):
        """Setup mappings for attack label normalization"""
        # Direct class mappings (lowercase input -> standard output)
        self.class_map = {
            "benign": "BENIGN",
            "ftp-patator": "FTP-Patator",
            "ftp patator": "FTP-Patator",
            "ssh-patator": "SSH-Patator",
            "ssh patator": "SSH-Patator",
            "dos slowloris": "DoS slowloris",
            "slowloris": "DoS slowloris",
            "dos slowhttptest": "DoS Slowhttptest",
            "slowhttptest": "DoS Slowhttptest",
            "dos hulk": "DoS Hulk",
            "hulk": "DoS Hulk",
            "dos goldeneye": "DoS GoldenEye",
            "goldeneye": "DoS GoldenEye",
            "heartbleed": "Heartbleed",
            "web attack - brute force": "Web Attack - Brute Force",
            "web attack brute force": "Web Attack - Brute Force",
            "web attack - xss": "Web Attack - XSS",
            "xss": "Web Attack - XSS",
            "web attack - sql injection": "Web Attack - SQL Injection",
            "sql injection": "Web Attack - SQL Injection",
            "infiltration": "Infiltration",
            "infiltration - portscan": "Infiltration",
            "botnet": "Botnet",
            "botnet - attempted": "Botnet",
            "portscan": "PortScan",
            "port scan": "PortScan",
            "ddos": "DDoS"
        }
        
        # Keywords to check for each attack type
        self.attack_keywords = {
            "FTP-Patator": ["ftp", "patator"],
            "SSH-Patator": ["ssh", "patator"],
            "DoS slowloris": ["slowloris"],
            "DoS Slowhttptest": ["slowhttp", "slow http"],
            "DoS Hulk": ["hulk"],
            "DoS GoldenEye": ["golden", "goldeneye", "eye"],
            "Heartbleed": ["heart", "bleed"],
            "Web Attack - Brute Force": ["web attack", "brute force"],
            "Web Attack - XSS": ["xss", "cross site"],
            "Web Attack - SQL Injection": ["sql", "injection"],
            "Infiltration": ["infiltration"],
            "Botnet": ["botnet", "bot net"],
            "PortScan": ["port", "scan"],
            "DDoS": ["ddos"]
        }
        
    def predict_single(self, messages, get_explanation=False):
        """Generate prediction for a single prompt with optional explanation"""
        # Format prompt in Llama 3 Chat format
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{messages[0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            if get_explanation:
                # Generate longer output for explanation
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                # Extract response
                response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                return response_text
            else:
                # Generate shorter output just to get the classification
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Enough to capture classification but not full explanation
                    do_sample=False,    # Deterministic for faster generation
                    pad_token_id=self.tokenizer.pad_token_id
                )
                # Extract response
                response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                # Extract attack class
                predicted_class = self._extract_attack_class(response_text)
                return predicted_class
    
    def _extract_attack_class(self, response):
        """Extract attack class from model response"""
        response_lower = response.lower()
        
        # Check for "classified as" pattern first
        classified_pattern = r"classified as ([A-Za-z0-9\s\-]+)"
        match = re.search(classified_pattern, response, re.IGNORECASE)
        
        if match:
            extracted_class = match.group(1).strip()
            return self._normalize_attack_class(extracted_class)
        
        # Try direct normalized matching
        for valid_class in self.valid_classes:
            if valid_class.lower() in response_lower:
                return valid_class
        
        # Check for benign explicitly
        if "benign" in response_lower:
            return "BENIGN"
        
        # Try keyword matching for each attack type
        for attack_class, keywords in self.attack_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    # Make sure it's not in a negation context
                    negation_patterns = [
                        f"no {keyword}", f"not {keyword}", 
                        f"isn't {keyword}", f"is not {keyword}"
                    ]
                    
                    # Check if not negated
                    if not any(neg in response_lower for neg in negation_patterns):
                        return attack_class
        
        # Default to benign if nothing matches
        return "BENIGN"
    
    def _normalize_attack_class(self, raw_class):
        """Normalize attack class to a standard format"""
        raw_lower = raw_class.lower()
        
        # Try direct mapping
        if raw_lower in self.class_map:
            return self.class_map[raw_lower]
        
        # Try partial matching
        for pattern, class_name in self.class_map.items():
            if pattern in raw_lower:
                return class_name
        
        # Try keywords for each attack type
        for attack_class, keywords in self.attack_keywords.items():
            for keyword in keywords:
                if keyword in raw_lower:
                    return attack_class
        
        # Return original if no match
        # Check if any of the valid classes match (case-insensitive)
        for valid_class in self.valid_classes:
            if valid_class.lower() == raw_lower:
                return valid_class
            
        # Return original with first letter capitalized as fallback
        return raw_class.capitalize()
    
    def _normalize_true_label(self, label):
        """Normalize true labels from the dataset to match prediction format"""
        # Handle common variations in dataset labels
        label_lower = label.lower()
        
        # Direct mapping
        if label_lower in self.class_map:
            return self.class_map[label_lower]
            
        # Try partial matching
        for pattern, class_name in self.class_map.items():
            if pattern in label_lower:
                return class_name
                
        # Special cases for dataset-specific labels
        if "web attack" in label_lower:
            if "brute" in label_lower:
                return "Web Attack - Brute Force"
            elif "xss" in label_lower:
                return "Web Attack - XSS"
            elif "sql" in label_lower or "injection" in label_lower:
                return "Web Attack - SQL Injection"
        
        # Return the original label if no mapping found
        # Check if matches any valid class case-insensitively
        for valid_class in self.valid_classes:
            if valid_class.lower() == label_lower:
                return valid_class
                
        return label  # Return original as last resort
        
    def evaluate_files(self, data_dir, output_dir, target_file="finetune_test_A.csv", samples_per_file=None):
        """Evaluate the model on test files"""
        logger = setup_logging()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get target file
        csv_files = [f for f in os.listdir(data_dir) if f == target_file]
        
        if not csv_files:
            logger.warning(f"Target file {target_file} not found in {data_dir}")
            return {}
        
        all_results = {}
        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            logger.info(f"Processing {file_path}")
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Sample if requested
                if samples_per_file and len(df) > samples_per_file:
                    sample_df = df.sample(n=samples_per_file, random_state=42)
                else:
                    sample_df = df
                    if samples_per_file:
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
                        predicted_class = self._extract_attack_class(explanation)
                        explanations.append({
                            "true_label": true_label,
                            "normalized_true_label": self._normalize_true_label(true_label),
                            "predicted": explanation,
                            "predicted_class": predicted_class,
                            "data": {k: str(row[k]) for k in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 
                                                        'Protocol', 'Total Fwd Packet', 
                                                        'Total Length of Fwd Packet', 'Label']}
                        })
                    except Exception as e:
                        logger.error(f"Error getting explanation: {e}")
                
                # Save explanations to file
                with open(output_path / f"{file_name.replace('.csv', '')}_explanations_ft_multiclass.json", 'w') as f:
                    json.dump(explanations, f, indent=4)
                
                # Log a few explanations
                logger.info(f"Sample explanations for {file_name}:")
                for i, exp in enumerate(explanations):
                    if i < 4:  # Just log a few for console output
                        logger.info(f"True label: {exp['true_label']}")
                        logger.info(f"Normalized true label: {exp['normalized_true_label']}")
                        logger.info(f"Predicted class: {exp['predicted_class']}")
                        logger.info(f"Model output: {exp['predicted']}")
                        logger.info("---")
                
                # Process in smaller batches for better memory management
                batch_size = 100
                predictions = []
                true_labels = []
                normalized_true_labels = []
                
                for i in tqdm(range(0, len(sample_df), batch_size), desc=f"Evaluating {file_name} in batches"):
                    batch_df = sample_df.iloc[i:i+batch_size]
                    batch_preds = []
                    batch_true = []
                    batch_true_norm = []
                    
                    for _, row in batch_df.iterrows():
                        try:
                            prompt = format_flow_prompt(row)
                            pred = self.predict_single(prompt)
                            batch_preds.append(pred)
                            
                            # Get true label and its normalized version
                            true_label = row['Label']
                            batch_true.append(true_label)
                            
                            # Normalize the true label for fair comparison
                            norm_true = self._normalize_true_label(true_label)
                            batch_true_norm.append(norm_true)
                            
                        except Exception as e:
                            logger.error(f"Error processing sample: {e}")
                            batch_preds.append("BENIGN")  # Default to benign on error
                            batch_true.append(row['Label'])
                            batch_true_norm.append(self._normalize_true_label(row['Label']))
                    
                    predictions.extend(batch_preds)
                    true_labels.extend(batch_true)
                    normalized_true_labels.extend(batch_true_norm)
                    
                    # Clear CUDA cache periodically to avoid memory issues
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate metrics using normalized labels
                accuracy = accuracy_score(normalized_true_labels, predictions)
                report = classification_report(normalized_true_labels, predictions)
                
                # Generate confusion matrix
                classes = sorted(list(set(normalized_true_labels + predictions)))
                cm = confusion_matrix(normalized_true_labels, predictions, labels=classes)
                cm_dict = {
                    'matrix': cm.tolist(),
                    'classes': classes
                }
                
                # Calculate attack-specific metrics
                attack_metrics = {}
                attack_types = list(set(normalized_true_labels))
                
                # Calculate metrics for each attack type
                for attack in attack_types:
                    # Get indices for this attack type
                    indices = [i for i, label in enumerate(normalized_true_labels) if label == attack]
                    if not indices:
                        continue
                    
                    # Get predictions for this attack type
                    attack_preds = [predictions[i] for i in indices]
                    attack_true = [attack] * len(indices)
                    
                    # For multiclass, we use accuracy and F1 score
                    accuracy = sum(1 for p, t in zip(attack_preds, attack_true) if p == t) / len(indices)
                    
                    # Calculate precision, recall, F1 using one-vs-rest approach
                    tp = sum(1 for p, t in zip(attack_preds, attack_true) if p == t and p == attack)
                    fp = sum(1 for p, t in zip(predictions, normalized_true_labels) if p == attack and t != attack)
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
                    'normalized_true_labels': normalized_true_labels,
                    'confusion_matrix': cm_dict,
                    'class_distribution': {
                        'true': dict(pd.Series(normalized_true_labels).value_counts().to_dict()),
                        'predicted': dict(pd.Series(predictions).value_counts().to_dict())
                    },
                    'attack_specific_metrics': attack_metrics,
                    'weighted_f1_score': float(weighted_f1),
                    'macro_f1_score': float(macro_f1)
                }
                
                all_results[file_name] = file_results
                
                # Save individual file results
                with open(output_path / f"{file_name.replace('.csv', '')}_results_ft_multiclass.json", 'w') as f:
                    json.dump(file_results, f, indent=4)
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Save combined results
        with open(output_path / "all_results_ft_multiclass.json", 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Multi-class evaluation of fine-tuned model on CIC-IDS-2017 dataset")
    parser.add_argument("--model_path", default="cicids_finetuned_multiclass_B/", help="Path to fine-tuned model")
    parser.add_argument("--data_dir", default="./", help="Directory containing test CSV files")
    parser.add_argument("--output_dir", default="multi_finetunedB_results_A", help="Directory to save results")
    parser.add_argument("--target_file", default="dataset_A_10k.csv", help="Target file to evaluate")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to test (None for all)")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Initialize evaluator with fine-tuned model
    evaluator = MulticlassModelEvaluator(args.model_path)
    
    try:
        # Evaluate files
        results = evaluator.evaluate_files(
            args.data_dir,
            args.output_dir,
            target_file=args.target_file,
            samples_per_file=args.samples
        )
        
        # Aggregate metrics across all files
        if results:
            total_samples = sum(r['total_samples'] for r in results.values())
            total_accuracy = sum(r['accuracy'] * r['total_samples'] for r in results.values()) / total_samples
            
            logger.info("\nOverall Results:")
            logger.info(f"Average Accuracy: {total_accuracy:.4f}")
            logger.info(f"Number of files processed: {len(results)}")
            
            # Create aggregated class distribution
            all_true_labels = []
            all_predictions = []
            for r in results.values():
                all_true_labels.extend(r['normalized_true_labels'])
                all_predictions.extend(r['predictions'])
            
            aggregated_report = classification_report(all_true_labels, all_predictions)
            logger.info(f"Aggregated Classification Report:\n{aggregated_report}")
            
            # Save overall results
            overall_results = {
                'average_accuracy': float(total_accuracy),
                'total_samples': total_samples,
                'aggregated_classification_report': aggregated_report,
                'aggregated_class_distribution': {
                    'true': dict(pd.Series(all_true_labels).value_counts().to_dict()),
                    'predicted': dict(pd.Series(all_predictions).value_counts().to_dict())
                },
                'files_processed': len(results)
            }
            
            with open(Path(args.output_dir) / "overall_results_ft_multiclass.json", 'w') as f:
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
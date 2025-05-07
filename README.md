# GTL-HIDS: Generative Tabular Learning-Enhanced Intrusion Detection System

## Overview

GTL-HIDS is a novel intrusion detection system that leverages embeddings from finetuned Large Language Models (LLMs) to enhance network security. This project bridges the gap between structured tabular data and natural language processing to enable more effective detection of both known attack patterns and zero-day threats.

Our approach enhances network intrusion detection by converting tabular network data into language model embeddings, enabling more effective pattern recognition and anomaly detection.

## Project Goals

- Develop novel techniques for applying LLMs to tabular data
- Establish effective data representation strategies for tabular-to-text conversion
- Demonstrate improved zero-shot and few-shot learning capabilities on tabular data

## Architecture

GTL-HIDS takes a novel approach to intrusion detection by converting network flow data into language-friendly representations and leveraging the pattern recognition abilities of finetuned LLM embeddings.

Our system uses a streamlined architecture for effective threat detection:

1. **Data Transformation Module**: Converts network flows into semantic text representations.

2. **Embedding Generation**: A finetuned LLaMA-3 8B model processes text representations to generate rich contextual embeddings that capture network behavior patterns.

3. **Neural Classification Network**: A specialized neural network trained on these LLaMA embeddings to classify traffic as benign or malicious, enabling detection of both known patterns and zero-day attacks.

## Dataset

We use the CIC-IDS2017 dataset, which is a temporal-based network traffic dataset containing 5 days of activity (Monday-Friday) with diverse attacks in a realistic network topology.

The CICIDS2017 dataset contains benign and the most up-to-date common attacks, which resembles true real-world data (PCAPs). It includes the results of network traffic analysis with labeled flows based on time stamp, source and destination IPs, ports, protocols and attack types.

### Attack Types Include:

- DDoS
- DoS
- Web Attacks (Brute Force, XSS, SQL Injection)
- Infiltration
- Botnet
- Port Scanning

### Finetuned Models:

8B_Finetuned_DatasetA : [Hmehdi515/gtl-hids-binary-A](https://huggingface.co/Hmehdi515/gtl-hids-binary-A)

8B_Finetuned_DatasetB : [Hmehdi515/gtl-hids-binary-B](https://huggingface.co/Hmehdi515/gtl-hids-binary-B)

### Embeddings:

https://huggingface.co/datasets/Hmehdi515/gtl-hids-embeddings

## Advantages of Our Approach

Our neural network approach leverages the rich understanding of language models while providing the speed and efficiency needed for real-time security applications. By using embeddings from a finetuned LLaMA model, our system can:

1. Understand the semantic meaning of network communications
2. Recognize novel attack patterns without prior exposure
3. Adapt to changing network environments with minimal retraining

Traditional IDS systems rely on fixed rules or statistical patterns that can be easily bypassed by new attack techniques. Our approach brings natural language understanding to network security, enabling more robust threat detection.

## Future Work

- Enhance the model with larger language models and more diverse training data
- Implement attention mechanisms to focus on the most security-relevant aspects of network flows
- Develop explainable AI components to help security analysts understand detection decisions
- Expand to additional network protocols and environments

## Acknowledgments

- Rochester Institute of Technology
- The contributors of the CIC-IDS2017 dataset
- The authors of the research papers that inspired this work

## Team

- Vaishak Nair
- Basanth Varaganti
- Hasan Mehdi
- Tanvirul Alam
- Nidhi Rastogi

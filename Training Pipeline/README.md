# Lab 2: Parameter Efficient Fine-Tuning of LLMs

This repository contains the implementation for Lab 2 of the ID2223/HT2025 course, focusing on parameter-efficient fine-tuning of Large Language Models using PEFT/LoRA techniques.

## Project Overview

This project demonstrates fine-tuning Llama-3.2 models (1B and 3B variants) on the FineTome-100k instruction dataset using QLoRA, a memory-efficient approach that combines 4-bit quantization with Low-Rank Adaptation. The fine-tuned models are exported in multiple formats for different deployment scenarios and evaluated against their base counterparts.

## Task Description

The lab consists of two main tasks:

**Task 1: Fine-Tune and Deploy**
- Fine-tune a Llama-3 model (1B or 3B) on the FineTome dataset using parameter-efficient methods
- Build a user interface with Gradio
- Deploy the UI to HuggingFace Spaces or Streamlit Cloud
- Convert the model to GGUF format for CPU inference compatibility

**Task 2: Document Improvement Strategies**
- Describe model-centric approaches (architecture, hyperparameters, training techniques)
- Describe data-centric approaches (dataset selection, preprocessing, augmentation)
- Document observed performance improvements

## Dataset Description

**FineTome-100k** (mlabonne/FineTome-100k)

A high-quality instruction-following dataset containing 100,000 conversational examples. For computational efficiency, a 1,000-sample subset was used for fine-tuning.

- Structure: Each example contains conversations between human and assistant, source attribution, and quality scores
- Sources: infini-instruct-top-500k, WebInstructSub_axolotl
- Split: 80% training (800 samples), 20% test (200 samples)
- Purpose: General instruction fine-tuning to improve task completion, coherence, and instruction following

**Bitext Customer Support Dataset** (bitext/Bitext-customer-support-llm-chatbot-training-dataset)

A specialized dataset designed for training customer service chatbots. Contains 26,872 examples of customer support interactions covering various categories including orders, billing, technical support, and account management.

- Structure: Each example contains instruction (customer query), category, intent, and response (agent reply)
- Subset used: 5,000 samples for computational efficiency
- Split: 80% training (4,000 samples), 20% test (1,000 samples)
- Purpose: Domain-specific fine-tuning to create a specialized customer support assistant
- Categories: ORDER, PAYMENT, ACCOUNT, SHIPPING, TECHNICAL_SUPPORT, and more

## Model Architecture and Training

### Base Models

Two Llama-3.2 model sizes were fine-tuned:

- **Llama-3.2-1B**: 1.24 billion parameters, suitable for resource-constrained deployment
- **Llama-3.2-3B**: 3.24 billion parameters, higher capacity for richer representations

Both models feature:
- Grouped Query Attention (GQA) for efficient inference
- SwiGLU activation functions
- RoPE positional embeddings
- 2048 hidden dimensions

### Fine-Tuning Methodology

**QLoRA Configuration**

4-bit quantization settings:
- Quantization type: NF4
- Double quantization enabled
- Compute dtype: bfloat16

**LoRA Parameters**

For Llama-3.2-1B:
- Rank (r): 8
- Alpha: 32
- Target modules: all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Dropout: 0.05
- Trainable parameters: 5.6M / 1.24B (0.454%)

For Llama-3.2-3B:
- Rank (r): 16
- Alpha: 32
- Target modules: all linear layers
- Dropout: 0.05
- Trainable parameters: 24.3M / 3.24B (0.751%)

**Training Hyperparameters**

For Llama-3.2-1B (FineTome):
- Training steps: 100
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: 2e-4
- Checkpointing: every 25 steps

For Llama-3.2-1B (Customer Support):
- Training steps: 200
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: 2e-4
- Checkpointing: every 50 steps
- Special training: Uses CompletionOnlyCollator to train only on assistant responses

For Llama-3.2-3B:
- Training steps: 150
- Batch size: 1
- Gradient accumulation: 8
- Learning rate: 2e-4
- Checkpointing: every 50 steps

### Model Variants

Three model variants were created:

1. **Fine-tuned 1B (General)**: Trained on FineTome-100k for general instruction following
2. **Fine-tuned 3B (General)**: Larger model trained on the same dataset for comparison
3. **Fine-tuned 1B (Customer Support)**: Specialized variant trained on Bitext Customer Support dataset

### Export Formats

Each fine-tuned model is available in three formats:

1. **LoRA Adapters**: Lightweight adapter weights that can be loaded on top of base models
2. **Merged Float16**: Full model with adapters merged, optimized for VLLM deployment
3. **GGUF**: Quantized formats (f16, Q4_K_M) for llama.cpp CPU inference

## Notebook Descriptions

### FineTuning-2.ipynb

Fine-tunes Llama-3.2-1B using the QLoRA approach.

Key steps:
1. Environment setup and dependency installation
2. Model initialization with 4-bit quantization
3. Dataset loading and preprocessing (FineTome-100k subset)
4. LoRA configuration with rank 8 targeting all linear layers
5. Training with SFTTrainer for 100 steps
6. Export to three formats: LoRA adapters, merged float16, and GGUF

Outputs:
- LoRA adapters: `Zedel17/fine_tuned_llama_1b`
- Merged float16: `Zedel17/llama_1b_merged_float16`
- GGUF files: `Zedel17/llama_1b_gguf`

### Evaluation_3_MODELS.ipynb

Compares three model variants through side-by-side response generation across diverse prompts.

Models evaluated:
1. Base Llama-3.2-1B-Instruct (unmodified)
2. FineTome fine-tuned model (general instruction following)
3. Customer Support fine-tuned model (domain-specific)

Evaluation methodology:
1. Load all three models with 4-bit quantization
2. Generate responses to identical prompts across all models
3. Display outputs side-by-side for direct comparison
4. Analyze differences in response quality, structure, and domain adaptation

Test categories:
- General conversation (greetings and casual interaction)
- Instruction following (structured outputs like poems)
- Knowledge questions (explaining technical concepts)
- Problem-solving (providing actionable advice)
- Code-related queries (Python explanations)
- Creative tasks (naming and brainstorming)
- Reasoning tasks (logic puzzles)
- Customer support scenarios (returns, payments, account issues, technical support)

Key findings:
- **Base model**: Provides generic, coherent responses but lacks specialization
- **FineTome fine-tuned model**: Shows better structure, improved instruction adherence, and more complete responses
- **Customer Support model**: Demonstrates domain-specific behavior with professional tone, but occasionally shows redundancy and includes policies not directly relevant (reflecting dataset characteristics)
- **Common limitation**: All three models sometimes stop generation early on long enumerations, a behavior typical of 1B parameter models with limited training

The evaluation demonstrates that fine-tuning successfully adapts the base model, with domain-specific training (Bitext) creating specialized behavior for customer service scenarios.

### LAST_VERSION_CustomerSupport.ipynb

Fine-tunes Llama-3.2-1B on the Bitext Customer Support dataset to create a domain-specific assistant.

Key features:
1. Dataset: 5,000 samples from bitext/Bitext-customer-support-llm-chatbot-training-dataset
2. Training approach: 200 steps (double the FineTome training)
3. Special data collator: CompletionOnlyCollator trains only on assistant responses, ignoring user prompts in loss calculation
4. Formatted as User/Assistant conversations for natural dialogue flow
5. Covers diverse customer service intents: order management, billing, technical support, returns, account issues

Training configuration:
- LoRA rank: 8
- Training steps: 200
- Gradient accumulation: 4
- Checkpointing: every 50 steps
- Dataset structure: instruction (customer query) → response (agent reply)

Outputs:
- LoRA adapters: `Zedel17/fine_tuned_llama_1b_Customer`
- Merged float16: `Zedel17/llama_1b_merged_float16_Customer`
- GGUF files: `Zedel17/llama_1b_gguf_Customer`

This specialized model is designed for integration into customer support chatbots and helpdesk systems, providing professional, context-aware responses to common customer service scenarios.

### FineTuning_3B.ipynb

Fine-tunes Llama-3.2-3B using an adapted QLoRA configuration.

Differences from 1B approach:
- Higher LoRA rank (16 vs 8) to accommodate larger model capacity
- Increased gradient accumulation (8 vs 4) due to memory constraints
- Extended training: 150 steps

Outputs:
- LoRA adapters: `Zedel17/fine_tuned_llama_3b`
- Merged float16: `Zedel17/llama_3b_merged_float16`
- GGUF files: `Zedel17/llama_3b_gguf`

## Gradio UI Deployment

A customer support chatbot interface was developed and deployed to HuggingFace Spaces, demonstrating real-world application of the fine-tuned model.

### Implementation Details

The UI consists of two main components:

**app.py**: Main application file that implements the Gradio interface and model inference.

- Loads the merged float16 model (`Zedel17/llama_1b_merged_float16`) from HuggingFace Hub
- Uses 8-bit quantization for CPU-efficient inference on HuggingFace Spaces
- Implements a chat function with the following generation parameters:
  - Max new tokens: 350
  - Temperature: 0.7
  - Top-p sampling: 0.9
  - Repetition penalty: 1.1
- Includes response post-processing to remove hallucinated follow-up prompts
- Features a clean, professional interface with welcome message and quick action examples

**personas.py**: Defines four customer service persona system prompts to guide model behavior.

The personas enable specialized responses for different customer service scenarios:

1. **Order Assistant**: Handles inquiries about orders, tracking numbers, shipping, delivery, returns, and exchanges. Provides concise, action-oriented responses with clear next steps.

2. **Billing & Pricing Assistant**: Focuses on costs, invoices, refunds, payment plans, and pricing questions. Explains financial details clearly with step-by-step breakdowns using simple language.

3. **Sales & Opportunities Advisor**: Recommends suitable plans, products, and upgrade options based on customer needs. Takes a consultative approach, presenting options with pros and cons while highlighting value.

4. **Technical Support**: Provides troubleshooting guidance, system configuration help, and technical explanations. Offers methodical step-by-step instructions and adjusts technical depth based on user expertise.

### User Interface Features

- Persona selector radio buttons for choosing assistant type
- Scrollable chat history (500px height) displaying conversation flow
- Text input box with send button for message submission
- Clear conversation button to reset chat history
- Quick action examples demonstrating common use cases:
  - "Track my order #12345"
  - "What's my account balance?"
  - "I need help with technical issues"
  - "Show me upgrade options"
- Technical details footer showing model specifications and training information

### Design Principles

The UI implements several best practices for customer service chatbots:

- **Context-aware responses**: Each persona uses tailored system prompts to maintain appropriate tone and focus
- **Single-turn optimization**: Personas are instructed to answer only the user's last message without unnecessary follow-up questions
- **Clean output**: Post-processing removes hallucinated "User:" and "Assistant:" tokens that may appear in generated text
- **Accessibility**: Professional interface without distracting elements, focused on practical customer service

The deployed application demonstrates the practical utility of the fine-tuned model in a customer support context, with the persona system enabling flexible adaptation to different service scenarios without requiring separate model variants.

## Evaluation Summary

The evaluation compared base and fine-tuned models across diverse prompts to assess improvements in instruction following, response quality, coherence, and task completion.

**Findings**:
- Fine-tuned models consistently provide more structured responses
- Instruction adherence improves notably on specific tasks (e.g., poem writing, formatting)
- Domain-specific fine-tuning (customer support) enables specialized behavior
- Both 1B and 3B fine-tuned models maintain coherence while following instructions more precisely

The qualitative evaluation demonstrates that parameter-efficient fine-tuning with LoRA successfully adapts base models to instruction-following tasks with minimal trainable parameters (less than 1% of total model weights).

## Repository Structure

```
lab2_scalable/
├── Training Pipeline/
│   ├── FineTuning-2.ipynb                      # Fine-tune Llama-3.2-1B on FineTome
│   ├── FineTuning_3B.ipynb                     # Fine-tune Llama-3.2-3B on FineTome
│   ├── LAST_VERSION_CustomerSupport.ipynb      # Fine-tune Llama-3.2-1B on Bitext Customer Support
│   ├── Evaluation_3_MODELS.ipynb               # Compare base vs FineTome vs Customer Support
│   ├── id2223_kth_lab2_2025.pdf               # Lab assignment document
│   └── README.md                               # This file
└── iris/
    ├── app.py                                  # Gradio UI application
    └── personas.py                             # Customer service persona definitions
```

**HuggingFace Links**:

Model Repositories:

Llama-3.2-1B outputs:
- https://huggingface.co/Zedel17/fine_tuned_llama_1b
- https://huggingface.co/Zedel17/llama_1b_merged_float16
- https://huggingface.co/Zedel17/llama_1b_gguf

Llama-3.2-3B outputs:
- https://huggingface.co/Zedel17/fine_tuned_llama_3b
- https://huggingface.co/Zedel17/llama_3b_merged_float16
- https://huggingface.co/Zedel17/llama_3b_gguf

Customer Support variant:
- https://huggingface.co/Zedel17/fine_tuned_llama_1b_Customer
- https://huggingface.co/Zedel17/llama_1b_merged_float16_Customer
- https://huggingface.co/Zedel17/llama_1b_gguf_Customer

Deployed Application:
- HuggingFace Space: (Add your Space URL here)

## Limitations and Future Work

### Current Limitations

**Dataset Size**: Only 1,000 samples used from FineTome-100k due to computational constraints. Full dataset fine-tuning could yield further improvements.

**Evaluation Scope**: Qualitative assessment only. Quantitative metrics (perplexity, BLEU, ROUGE) would provide objective performance measurements.

**Hardware Constraints**: Google Colab T4 GPU limits batch size and requires 4-bit quantization. Larger GPUs would enable higher precision training.

### Potential Improvements

**Model-Centric Approaches**:
- Experiment with different LoRA ranks and alpha values
- Test alternative quantization methods (8-bit, full precision)
- Implement learning rate scheduling and warmup strategies
- Evaluate larger model sizes (7B, 13B variants)

**Data-Centric Approaches**:
- Fine-tune on full FineTome-100k dataset
- Experiment with domain-specific datasets beyond customer support
- Implement data augmentation techniques
- Filter low-quality examples based on score thresholds

**Evaluation Enhancements**:
- Add quantitative metrics for objective comparison
- Expand test prompt diversity and count
- Conduct user studies for real-world performance assessment
- Benchmark inference speed across different formats

## Resource Constraints and Training Observations

The models developed in this project were trained under significant computational constraints that affected the final performance. Understanding these limitations provides context for the results and highlights opportunities for improvement.

### Computational Limitations

**Hardware Constraints**:
- Training environment: Google Colab with free T4 GPU allocation
- Memory limitations: Necessitated 4-bit quantization and small batch sizes
- Training interruptions: Free Colab sessions have time limits requiring checkpoint resumption
- No access to high-memory GPUs (A100, H100) for full-precision training

**Dataset Limitations**:
- FineTome: Only 1,000 samples used (1% of full dataset)
- Bitext Customer Support: Only 5,000 samples used (18.6% of full dataset)
- Subset selection was necessary to fit within time and memory constraints

**Training Duration Constraints**:
- Llama-3.2-1B (FineTome): 100 steps
- Llama-3.2-1B (Customer Support): 200 steps
- Llama-3.2-3B (FineTome): 150 steps
- Industry standard fine-tuning typically runs for 1,000-10,000 steps

### Impact on Model Performance

The limited training has observable effects on model behavior:

1. **Incomplete Generation**: Models sometimes stop mid-sentence or mid-enumeration, particularly when generating lists or multi-step instructions. This is characteristic of undertrained small models.

2. **Inconsistent Quality**: Responses can vary in quality, with some showing excellent instruction adherence while others revert to base model behavior.

3. **Limited Domain Adaptation**: While the Customer Support model shows domain-specific behavior, it occasionally includes irrelevant information or redundant explanations, indicating incomplete learning of the task distribution.

4. **Generic Fallback**: On complex or edge-case queries, models may fall back to generic responses rather than demonstrating specialized knowledge.

### Expected Improvements with Additional Training

With access to greater computational resources, the following improvements would be achievable:

**Extended Training (1,000-3,000 steps)**:
- More consistent response quality across diverse prompts
- Better instruction adherence and task completion
- Reduced tendency to generate incomplete responses
- Stronger domain specialization for customer support scenarios

**Full Dataset Fine-Tuning**:
- FineTome-100k (full 100,000 samples): Broader instruction-following capabilities
- Bitext (full 26,872 samples): Comprehensive coverage of customer service intents
- Improved generalization to unseen queries within the domain

**Higher Precision Training**:
- 8-bit or full precision fine-tuning (vs 4-bit QLoRA)
- Better preservation of base model capabilities
- Reduced quantization artifacts

**Larger Model Sizes**:
- Llama-3.2-7B or Llama-3.2-13B variants
- Substantially better reasoning and generation quality
- More coherent long-form responses

**Optimized Hyperparameters**:
- Learning rate scheduling with warmup and decay
- Larger batch sizes through gradient accumulation
- Multiple training epochs over the full dataset

### Conclusion on Resource Constraints

Despite these limitations, the fine-tuned models demonstrate measurable improvements over the base model in instruction following, response structure, and domain-specific behavior. The results validate the PEFT/LoRA approach as effective for adapting LLMs with minimal computational overhead.

However, these models represent a proof of concept rather than production-ready systems. Given the free-tier computational constraints of Google Colab and the limited time available for this lab project, the achieved results represent the best possible outcome within these boundaries.

For real-world deployment, we recommend:
- Training on the full datasets for at least 1,000 steps
- Using larger model sizes (7B+ parameters) for improved reasoning
- Employing higher precision training when memory permits
- Conducting extensive evaluation with quantitative metrics and user studies

The gap between these constrained experiments and production-quality models highlights the critical importance of computational resources in modern LLM development.

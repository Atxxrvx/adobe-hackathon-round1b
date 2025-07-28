# Document Analysis and Section Extraction Methodology

## Overview
This solution implements an intelligent document analysis system that extracts and ranks the most relevant sections from PDF documents based on user persona and job requirements. The approach combines semantic similarity analysis with multilingual support and advanced ranking algorithms.

## Core Methodology

### 1. Document Selection and Preprocessing
The system first performs intelligent document filtering using semantic similarity between the user's job requirements and document content descriptions. This reduces computational overhead by focusing only on the most relevant documents (top 70% by default) rather than processing all available PDFs.

### 2. Multilingual Language Detection
Advanced language detection is implemented using the `langdetect` library with robust fallback mechanisms. The system automatically identifies the language of both the job description and document content, enabling proper stopword filtering and text processing for multiple languages including English, Spanish, French, German, and many others.

### 3. Semantic Section Extraction
The core extraction algorithm uses PyMuPDF (fitz) to analyze PDF structure by identifying headings based on font characteristics. Sections are extracted by:
- Analyzing font sizes and styles to identify potential headings
- Using median font size as a baseline to detect headings (110% larger than median)
- Extracting content between consecutive headings with precise coordinate-based clipping

### 4. Multi-Perspective Query Generation
Instead of relying on a single query, the system generates multiple perspective queries to capture different aspects of the user's requirements:
- Direct persona-task combinations
- Requirement-focused queries
- Solution-oriented queries
- How-to guides specific to the persona

### 5. Advanced Similarity Scoring
The ranking algorithm employs a sophisticated ensemble approach:
- **Primary Similarity**: Uses multilingual sentence transformers (paraphrase-multilingual-MiniLM-L12-v2) to compute semantic similarity
- **Ensemble Scoring**: Combines maximum similarity (70%) with average similarity (30%) across all perspective queries
- **Concept Boosting**: Extracts key concepts from job descriptions and provides additional scoring weight (25%) for sections containing related concepts

### 6. Intelligent Content Refinement
For the top-ranked sections, the system performs sentence-level analysis:
- Splits content into individual sentences
- Ranks sentences by relevance to the primary query
- Reconstructs refined content using the top 10 most relevant sentences
- Maintains logical flow by preserving sentence order

## Technical Advantages
- **Scalability**: Batch processing with configurable batch sizes for optimal memory usage
- **Multilingual Support**: Automatic language detection with appropriate stopword filtering
- **GPU Acceleration**: Automatic CUDA detection for faster embedding computation
- **Robust Error Handling**: Graceful fallbacks for edge cases in PDF processing and language detection

This approach ensures high-precision extraction of the most relevant document sections while maintaining computational efficiency and supporting diverse multilingual


# Challenge 1B - Instructions

## Docker Setup & Run

1. Build the image:

   ```bash
   docker build --platform=linux/amd64 -t challenge1b .
   ```

2. Run the container:
   ```bash
   docker run --rm \
     -v $(pwd)/sample_dataset/persona_input:/app/input:ro \
     -v $(pwd)/sample_dataset/persona_output:/app/output \
     --network none \
     challenge1b
   ```

## Local Setup & Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run:
   ```bash
   python analyze_documents.py
   ```

## Requirements

**Input:** Each `Collection_*/` folder must contain:

- `challenge1b_input.json`
- A `PDFs/` directory with the listed documents

**Output:** For each collection, the container will generate:

- `challenge1b_output.json` saved under the corresponding subfolder in `/app/output/`


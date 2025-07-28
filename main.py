import json
import fitz
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import time
import statistics
import math
from typing import List, Dict, Optional
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import stopwordsiso

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SUBSECTION_ANALYSIS_COUNT = 10
MAX_CONTENT_LENGTH = 1000
BATCH_SIZE = 8
CONCEPT_BOOST_WEIGHT = 0.25
MIN_TEXT_LENGTH_FOR_DETECTION = 50
PENALTY_FACTOR = -1.0

def detect_language(text: str, fallback: str = 'en') -> str:
    try:
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        clean_text = ' '.join(clean_text.split())
        
        if len(clean_text) < MIN_TEXT_LENGTH_FOR_DETECTION:
            return fallback
            
        detected_lang = detect(clean_text)
        
        if detected_lang in stopwordsiso.langs():
            return detected_lang
        else:
            return fallback
            
    except (LangDetectException, Exception):
        return fallback

def get_stopwords_for_language(language: str) -> set:
    try:
        return set(stopwordsiso.stopwords(language))
    except Exception:
        return set(stopwordsiso.stopwords('en'))

def select_relevant_documents(documents_info: List[Dict], persona: str, job_to_be_done: str, 
                            model: SentenceTransformer, k_ratio: float = 0.7) -> List[Dict]:
    if len(documents_info) <= 2:
        return documents_info
    
    k = max(2, min(len(documents_info), int(math.ceil(len(documents_info) * k_ratio))))
    
    doc_selection_query = f"A {persona} needs to {job_to_be_done}. Relevant document topics and content areas."
    
    doc_descriptions = [
        f"Document about: {re.sub(r'[_-]', ' ', doc_info['filename'].replace('.pdf', ''))}"
        for doc_info in documents_info
    ]
    
    query_embedding = model.encode(doc_selection_query, convert_to_tensor=True, normalize_embeddings=True)
    doc_embeddings = model.encode(doc_descriptions, convert_to_tensor=True, normalize_embeddings=True, batch_size=BATCH_SIZE)
    
    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
    
    document_scores = []
    for i, doc_info in enumerate(documents_info):
        document_scores.append({
            'doc_info': doc_info,
            'filename': doc_info["filename"],
            'score': similarities[i].item(),
            'clean_name': doc_descriptions[i].replace("Document about: ", "")
        })
    
    document_scores.sort(key=lambda x: x['score'], reverse=True)
    selected_docs = [item['doc_info'] for item in document_scores[:k]]
    
    return selected_docs

def create_multi_perspective_queries(persona: str, job_to_be_done: str) -> List[str]:
    return [
        f"Persona: {persona}. Task: {job_to_be_done}.",
        f"A {persona} needs to {job_to_be_done}",
        f"Requirements for {persona}: {job_to_be_done}",
        job_to_be_done,
        f"Solutions for: {job_to_be_done}",
        f"How to {job_to_be_done} as a {persona}",
        f"{persona} guide for {job_to_be_done}"
    ]

def extract_key_concepts(text: str, detected_language: Optional[str] = None) -> List[str]:
    if detected_language is None:
        detected_language = detect_language(text)
    
    stop_words = get_stopwords_for_language(detected_language)
    
    words = re.findall(r'\b[a-zA-ZÀ-ÿĀ-žА-я]{3,}\b', text.lower())
    
    meaningful_words = [
        word for word in words 
        if word not in stop_words and len(word) > 3
    ]
    
    unique_concepts = list(dict.fromkeys(meaningful_words))
    
    return unique_concepts

def compute_ensemble_similarity(queries: List[str], section_content: str, 
                              model: SentenceTransformer) -> float:
    truncated_content = section_content[:MAX_CONTENT_LENGTH]
    
    query_embeddings = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    section_embedding = model.encode(truncated_content, convert_to_tensor=True, normalize_embeddings=True)
    
    similarities = util.cos_sim(query_embeddings, section_embedding.unsqueeze(0))
    
    return 0.7 * torch.max(similarities).item() + 0.3 * torch.mean(similarities).item()

def apply_concept_boost(sections: List[Dict], job_to_be_done: str, job_concepts: List[str], 
                       model: SentenceTransformer, job_language: str) -> List[Dict]:
    
    V_KEYWORDS = ['Vetarian', 'Van', 'Vgie', 'plant-based']
    NV_KEYWORDS = ['chicken', 'beef', 'pork', 'fish', 'turkey', 'lamb', 'meat', 'sausage', 'bacon', 'shrimp', 'ground beef', 'ground pork']

    job_is_V = any(keyword in job_to_be_done.lower() for keyword in V_KEYWORDS)
    job_is_non_V = any(keyword in job_to_be_done.lower() for keyword in NV_KEYWORDS)

    if not job_concepts and not (job_is_V or job_is_non_V):
        return sections

    concept_embeddings = model.encode(job_concepts, convert_to_tensor=True, normalize_embeddings=True) if job_concepts else None
    
    for section in sections:
        section_text = (section.get('title', '') + ' ' + section.get('content', '')).lower()
        
        if job_is_V:
            if any(keyword in section_text for keyword in NV_KEYWORDS):
                section['score'] += PENALTY_FACTOR
                continue 

        if job_is_non_V:
            if any(keyword in section_text for keyword in V_KEYWORDS):
                section['score'] += PENALTY_FACTOR
                continue

        if concept_embeddings is not None:
            section_language = detect_language(section_text, fallback=job_language)
            content_words = extract_key_concepts(section_text, section_language)
            
            if not content_words:
                continue
                
            content_embeddings = model.encode(content_words, convert_to_tensor=True, normalize_embeddings=True)
            concept_similarities = util.cos_sim(concept_embeddings, content_embeddings)
            
            max_concept_sim = torch.max(concept_similarities).item() if concept_similarities.numel() > 0 else 0
            avg_concept_sim = torch.mean(concept_similarities).item() if concept_similarities.numel() > 0 else 0
            concept_boost_val = 0.6 * max_concept_sim + 0.4 * avg_concept_sim
            
            section['score'] += concept_boost_val * CONCEPT_BOOST_WEIGHT
    
    return sorted(sections, key=lambda x: x['score'], reverse=True)


def deduplicate_sections(sections: List[Dict]) -> List[Dict]:
    seen = set()
    return [
        section for section in sections
        if not (key := (section['doc_name'], section['title'].lower().strip())) in seen
        and not seen.add(key)
    ]

def get_document_sections(pdf_path: Path) -> List[Dict]:
    sections = []
    try:
        doc = fitz.open(pdf_path)
        potential_headings = []
        font_sizes = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if b['type'] == 0 and b.get('lines'):
                    span = b['lines'][0]['spans'][0]
                    font_sizes.append(round(span['size']))
                    if "bold" in span['font'].lower():
                        heading_text = "".join(s['text'] for l in b['lines'] for s in l['spans']).strip()
                        if (heading_text and len(heading_text) > 2 and 
                            not heading_text.isspace() and 
                            any(c.isalnum() for c in heading_text)):
                            potential_headings.append({
                                "text": heading_text,
                                "bbox": b['bbox'],
                                "size": span['size'],
                                "page_num": page_num
                            })

        if not font_sizes:
            return []

        median_size = statistics.median(font_sizes)
        headings = [h for h in potential_headings if h['size'] > median_size * 1.1]
        headings.sort(key=lambda x: (x['page_num'], x['bbox'][1]))

        for i, heading in enumerate(headings):
            start_page = heading['page_num']
            start_y = heading['bbox'][3]
            end_page = headings[i+1]['page_num'] if i+1 < len(headings) else len(doc) - 1
            end_y = headings[i+1]['bbox'][1] if i+1 < len(headings) else doc[end_page].rect.height

            content_text = ""
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                clip_start = start_y if page_num == start_page else 0
                clip_end = end_y if page_num == end_page else page.rect.height
                
                clip_rect = fitz.Rect(0, clip_start, page.rect.width, clip_end)
                if not clip_rect.is_empty and clip_rect.height > 0:
                    content_text += page.get_text("text", clip=clip_rect)

            title = heading['text'].strip()
            content = content_text.strip()
            
            if title and content and len(content.split()) > 5:
                sections.append({
                    "title": title,
                    "content": content,
                    "page_num": heading['page_num'] + 1,
                    "doc_name": pdf_path.name
                })

    except Exception:
        pass
    
    return sections

def process_collection(collection_path: Path, model: SentenceTransformer) -> None:
    input_json_path = collection_path / "challenge1b_input.json"
    pdfs_dir = collection_path / "PDFs"

    if not input_json_path.exists():
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona = input_data.get("persona", {}).get("role", "")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
    documents_info = input_data.get("documents", [])

    job_language = detect_language(job_to_be_done)

    selected_documents = select_relevant_documents(documents_info, persona, job_to_be_done, model)
    
    all_sections = []
    for doc_info in selected_documents:
        pdf_path = pdfs_dir / doc_info["filename"]
        if pdf_path.exists():
            sections = get_document_sections(pdf_path)
            all_sections.extend(sections)

    if not all_sections:
        return

    all_sections = deduplicate_sections(all_sections)

    queries = create_multi_perspective_queries(persona, job_to_be_done)
    
    job_concepts = extract_key_concepts(job_to_be_done, job_language)

    for section in all_sections:
        corpus_content = f"Document: {section['doc_name']}. Section Title: {section['title']}.\nContent: {section['content']}"
        section['score'] = compute_ensemble_similarity(queries, corpus_content, model)

    all_sections = apply_concept_boost(all_sections, job_to_be_done, job_concepts, model, job_language)
    
    all_sections.sort(key=lambda x: x['score'], reverse=True)

    output_data = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in documents_info],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "extracted_sections": [
            {
                "document": section['doc_name'],
                "section_title": section['title'],
                "importance_rank": i + 1,
                "page_number": section['page_num']
            }
            for i, section in enumerate(all_sections[:10])
        ],
        "subsection_analysis": []
    }

    best_query = queries[0]
    best_query_embedding = model.encode(best_query, convert_to_tensor=True, normalize_embeddings=True)

    for section in all_sections[:SUBSECTION_ANALYSIS_COUNT]:
        sentences = re.split(r'(?<=[.!?])\s+', section['content'])
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]
        if not sentences:
            continue

        sentence_embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True, batch_size=BATCH_SIZE)
        sent_scores = util.cos_sim(best_query_embedding, sentence_embeddings)[0]
        top_indices = sent_scores.argsort(descending=True)[:10].tolist()

        refined_text = " ".join(sentences[i] for i in sorted(top_indices) if i < len(sentences))
        cleaned_refined_text = re.sub(r'\s*\n\s*', ' ', refined_text).strip()

        output_data["subsection_analysis"].append({
            "document": section['doc_name'],
            "section_title": section['title'],
            "refined_text": cleaned_refined_text,
            "page_number": section['page_num']
        })

    output_dir = Path('/app/output') / collection_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "challenge1b_output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def main():
    print("Running...")
    
    print("Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
    print(f"Model loaded successfully on {device}.")

    base_path = Path('/app/input')
    collection_dirs = sorted([d for d in base_path.glob('Collection*') if d.is_dir()])

    if collection_dirs:
        for collection_dir in collection_dirs:
            process_collection(collection_dir, embedding_model)

    print("Executed")

if __name__ == "__main__":
    main()
# Dataklout Model Architecture Overview

This document outlines **Dataklout’s phased approach** to speech analytics model deployment and enhancement, combining industry-leading tools with proprietary model development to deliver tailored solutions for client-specific requirements.

---

## 📊 Phase 1: Baseline Implementation

### 1️⃣ Tokenizer
- **Model:** Whisper

### 2️⃣ Language Model
- **Model:** Whisper

### 3️⃣ Transcription Engine
- **Model:** Whisper Large V3

### 4️⃣ Speaker Diarization
- **Method:** Agglomerative Clustering

### 5️⃣ Translation Model
- **Model:** Seamless M4T

### 6️⃣ Reporting Modules
- **Collections:**  
  NLP and fuzzy matching models leveraging syntax cues extracted from dialogues.

- **Agent Scorecard:**  
  *Pending development; awaiting finalized agent evaluation criteria from the client.*

- **Customer & Product Intent Detection:**  
  NLP and fuzzy matching models based on syntactic and contextual cues.

- **Promise-to-Pay Detection:**  
  NLP and fuzzy models designed to identify payment commitments.

- **Call Analysis:**  
  Comprehensive NLP and fuzzy logic-based models for conversational analysis.

---

## 🛠️ Phase 2: Proprietary Model Development

### 1️⃣ Custom Tokenizer
- Tailored tokenizer trained on a proprietary corpus of millions of **Taglish (Tagalog-English)** sentences.

### 2️⃣ Custom Language Model
- Bespoke language model trained on an extensive **Taglish dataset** to enhance transcription and intent detection accuracy.

### 3️⃣ Custom Transcription Model
- In-house transcription engine trained on manually labeled **Maya Bank call recordings**.

---

## 📌 Current Status

- 📦 Data preparation is actively in progress.
- 🧠 Neural network architecture first draft has been built.
- 🚀 Model training will commence upon data integration.

---

## 📑 Summary

Dataklout aims to deliver highly specialized and scalable **speech analytics solutions** through a phased strategy — starting with industry-leading tools and evolving towards fully proprietary, domain-specific models optimized for client needs.

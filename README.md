# Mini-Project #4: Generating Synthetic Data to Fine-Tune Custom Embedding Models for More Performant RAG Systems

**Author:** Jon Chun
**Date:** 8 Apr 2025
**Due:** TBA

### **TODO:** Revise for a Classification Task to simplify Benchmarks and Assessment.

---

![GenAI The Garden of Digital Vectors by AI Hieronymus Bosch](./docs/garden-of-digital-vectors_hieronymus-bosch.png?raw=true)
*The Garden of Digital Vectors: A surreal landscape inspired by Hieronymus Bosch where embedding vectors grow like crystalline trees, tended by AI gardeners in Renaissance attire, with knowledge fragments floating as translucent fruits, and a central fountain where synthetic and organic data streams merge.*

## OVERVIEW

This project takes you through an end-to-end workflow highly relevant in modern AI development. You will tackle the challenge of improving Retrieval-Augmented Generation (RAG) systems for specialized domains. You'll start by selecting a specific field with unique terminology (like finance, law, medicine, or a specific academic niche you're familiar with). You will then gather a small set of documents from this domain and learn techniques to **synthetically generate** additional training data. Using this augmented dataset, you will **fine-tune** core embedding models (like the ModernBERT shown in the example) to better understand your chosen domain's nuances. Finally, you'll build a RAG pipeline incorporating your custom models and, crucially, **benchmark** its performance against a baseline system using the original generic embedding model. This project focuses on developing practical skills in data augmentation, model fine-tuning, RAG implementation, and quantifying performance improvements – essential for translating AI theory into real-world impact.

## MOTIVATION

Large Language Models (LLMs) are powerful, but they often struggle with factual accuracy (hallucination) and lack knowledge of events after their training cutoff or information from private/specialized document sets. Retrieval-Augmented Generation (RAG) addresses this by allowing LLMs to consult a relevant knowledge base *before* answering a query, significantly improving reliability and enabling access to up-to-date or specialized information.

However, the effectiveness of RAG heavily relies on the "Retrieval" step – finding the *truly* relevant information. Embedding models trained on general web text might perform poorly in specialized domains. **Why?** Because they may not understand the specific jargon, acronyms, concepts, or phrasing unique to that field. For example, a generic RAG system asked about "alpha generation strategies" based on financial analyst conference call transcripts might retrieve irrelevant documents mentioning the word "alpha" in a different context (like software testing or Greek letters) if its embedding model doesn't grasp the specific financial meaning.

This is where **fine-tuning** comes in. By further training a base embedding model on text from your specific domain, you can teach it the relevant vocabulary and semantic relationships. But gathering large amounts of labeled data for fine-tuning is often difficult. **Synthetic data generation**, using powerful generative AI to create relevant training examples (like question-answer pairs based on your documents), offers a practical solution to augment smaller datasets.

Finally, simply fine-tuning isn't enough. In industry and research, it's critical to **benchmark** and **quantify** any performance improvements. Does your fine-tuned model *actually* lead to better RAG results in your domain? How much better? Benchmarking provides concrete evidence of the value created by translating theoretical improvements (like domain-specific fine-tuning) into measurable real-world practice.

## METHODOLOGY

Follow these steps to complete the project:

1.  **Identify Domain & Collect Initial Data:**
    * Choose a specialized domain that interests you and where you can find some text documents. This could be finance (e.g., quarterly earnings call transcripts from SEC EDGAR), law (e.g., specific types of court opinions), medicine (e.g., research papers on a particular topic from PubMed Central), diplomacy (e.g., UN meeting transcripts), an academic field (sociology, history, etc.), or even a domain specific to a hobby (e.g., complex board game rules).
    * Gather a *small but representative* set of text documents (e.g., 10-50 documents, depending on length and complexity). Focus on quality and relevance to the domain's unique language.

2.  **Generate Synthetic Training Data:**
    * The goal is to create more training examples formatted similarly to those used in the example notebook (`anchor`, `positive`, `negative` triplets for the bi-encoder; `anchor`, `positive`, `score` pairs for the cross-encoder).
    * A common approach (though requiring potentially external tools/APIs) is to use a powerful LLM (like GPT-4, Claude 3, Gemini) to:
        * Read chunks of your collected documents (`context`).
        * Generate relevant questions (`question`) based on the context.
        * Identify the context as the `positive_retrieval`.
        * Find *other* chunks in your documents that are *irrelevant* to the generated question to serve as `negative_retrieval`.
        * You might also generate more nuanced positive/negative pairs for re-ranking (`positive_reranking`, `negative_reranking`).
    * *Note:* The provided notebook `/src/fine_tune_modernbert_rag.ipynb` starts *after* this step, assuming the data is already generated and formatted. You will need to adapt methods or use external tools/scripts to perform this generation step based on your collected documents. Aim to generate a few hundred synthetic examples if possible.

3.  **Fine-Tune Embedding Models:**
    * Study the example notebook: `/src/fine_tune_modernbert_rag.ipynb`
    * Consult the detailed tutorial: `/docs/project_tutorial.md`
    * **Adapt the notebook:** Modify the data loading sections to use *your* synthetically generated dataset.
    * Run the fine-tuning process for both the **bi-encoder** (for retrieval) and the **cross-encoder** (for re-ranking), saving your custom models.

4.  **Construct RAG Systems:**
    * The example notebook (`/src/fine_tune_modernbert_rag.ipynb`) also demonstrates building a RAG pipeline using Haystack.
    * Build two versions of the pipeline:
        * **Pipeline A (Baseline):** Uses the *original* base embedding model (e.g., `nomic-ai/modernbert-embed-base` for both retrieval and ranking, or just retrieval if not fine-tuning a ranker).
        * **Pipeline B (Fine-Tuned):** Uses *your fine-tuned* bi-encoder and cross-encoder models.
    * Ensure both pipelines use the same underlying LLM for the generation step for a fair comparison. Use the documents you collected in Step 1 as the knowledge base.

5.  **Design & Execute Benchmarks:**
    * **Create a Test Set:** Develop a set of diverse Question-Answer pairs specifically for your chosen domain. Include:
        * Questions testing understanding of core domain concepts.
        * Questions requiring knowledge of specific jargon or acronyms.
        * Potential "edge case" questions that might confuse a generic model.
        * For each question, manually identify the "ground truth" document chunk(s) that contain the correct answer.
    * **Run Evaluations:** Feed each question from your test set into both Pipeline A and Pipeline B.
    * **Measure Performance:** Compare the results. Consider these evaluation approaches:
        * **Retrieval Evaluation (Recommended):** For each question, check if the correct ground truth document chunk(s) were retrieved by the retriever component within the top-k results (e.g., top 3 or top 5). Compare the Hit Rate (percentage of questions where the correct chunk was retrieved) for Pipeline A vs. Pipeline B.
        * **Answer Quality (Qualitative):** Subjectively rate the generated answers from both pipelines based on accuracy, relevance, and completeness according to your domain knowledge. Note down specific examples of where the fine-tuned pipeline performed better (or worse).
        * *(Optional/Advanced): Research and potentially implement quantitative metrics like Mean Reciprocal Rank (MRR) for retrieval or LLM-based evaluations for answer faithfulness.*

6.  **Write Report:**
    * Use the template provided in `/docs/rag_performance_report.md`.
    * Clearly describe your chosen domain, data collection, synthetic data generation approach (even if conceptual), and fine-tuning process.
    * Present your benchmarking methodology and **results** (use tables and examples!).
    * Discuss your findings: Did fine-tuning improve performance? By how much (quantitatively or qualitatively)? Where did it help most? Were there any challenges?

7.  **EXTRA CREDIT:**
    * Search Hugging Face Models ([https://huggingface.co/models](https://huggingface.co/models)) for other embedding models *already* potentially fine-tuned on your specific domain (e.g., search for "finance embedding", "biomedical embedding").
    * If you find suitable models, add a **Pipeline C** (and D, etc.) to your benchmark using these pre-existing fine-tuned models.
    * Compare their performance against both the base model and your own fine-tuned model in your report.

## REFERENCES

* [(Blog) Fine-tune ModernBERT for RAG with Synthetic Data (20 Jan 2025) by Han-Diaz et al. ](https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data)
* [Fine-tune ModernBERT with Synthetic Data for RAG](https://github.com/argilla-io/synthetic-data-generator/blob/main/examples/fine-tune-modernbert-rag.ipynb)
* [How to Fine-tune ModernBERT for Classification (5:38) (3 Jan 2025) ](https://www.youtube.com/watch?v=7-js_--plHE)
* [Does Fine Tuning Embedding Models Improve RAG? (26:03) (12 Sep 2024)](https://www.youtube.com/watch?v=hztWQcoUbt0)
  * [Github](https://github.com/ALucek/linear-adapter-embedding)
* [Improving RAG Retrieval by 60% with Fine-Tuned Embeddings (30:11) (Mar 2025)](https://www.youtube.com/watch?v=v28Pu7hsJ0s&t=479s)
  * [Github](https://github.com/ALucek/ft-modernbert-domain)
* [Huggingface.co Models](https://huggingface.co/models)
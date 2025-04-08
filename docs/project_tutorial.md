# RAG Tutorial

Okay, let's revise the tutorial for a Computer Science undergraduate audience. We'll add more background on RAG, explain concepts more fundamentally, and connect the steps back to basic principles like embeddings and similarity.

---

## RAG Explained: Fine-Tuning ModernBERT for Smarter AI Answers (Undergrad Guide)

You've probably used Large Language Models (LLMs) like ChatGPT or Gemini. They're amazing at generating human-like text, translating languages, and answering questions. But they have limitations:

1.  **Knowledge Cutoff:** They only know information up to the point their training data ended. Ask about very recent events, and they might not know.
2.  **Hallucination:** Sometimes, LLMs confidently make up incorrect information. This is a big problem if you need reliable answers.
3.  **Lack of Specificity:** They might lack deep knowledge about niche subjects or private company data.

How can we fix this? Enter **Retrieval-Augmented Generation (RAG)**!

### What is RAG? (And Why Should You Care?)

Think of RAG as giving an LLM access to a targeted search engine *before* it answers your question. Instead of relying solely on its internal (and potentially outdated or wrong) knowledge, the RAG process looks like this:

1.  **Question In:** You ask a question (the "Query").
2.  **Retrieve:** The system searches through a specific set of documents (like internal company wikis, product manuals, or, in our case, human rights texts) to find information relevant to your query. This is the "Retrieval" step.
3.  **Augment:** The relevant information found is combined with your original query.
4.  **Generate:** This combined information (context + query) is fed to the LLM, which then generates an answer *based on the provided documents*. This is the "Generation" step.

**Why is this cool?**
* **Reduces Hallucination:** The LLM bases its answer on actual retrieved text, making it much more likely to be factual.
* **Uses Current Info:** The document collection can be updated anytime, keeping the RAG system current.
* **Domain-Specific:** It can answer questions about specialized topics covered in the documents.

### The Magic Behind Retrieval: Embeddings and Similarity

How does the "Retrieve" step work? This is where the concepts of **embeddings** and **semantic similarity** that you might have learned about come into play.

1.  **Vectorization:** We use a special type of model called an **embedding model** (like the ModernBERT we'll use) to convert chunks of text from our documents into lists of numbers called **vectors** (or embeddings). The key idea is that texts with similar meanings will have vectors that are "close" to each other in a high-dimensional "semantic space".
2.  **Indexing:** These document vectors are stored in a special database (a **vector store** or **document store**) that allows for very fast searching.
3.  **Query Embedding:** When you ask a question, your query is also converted into a vector using the *same* embedding model.
4.  **Similarity Search:** The system searches the vector store for the document vectors that are mathematically closest (often using **cosine similarity**) to your query vector. These are considered the most relevant documents.

### Making Retrieval Better: Fine-Tuning and Two Types of Models

Generic, pre-trained embedding models (like the base `nomic-ai/modernbert-embed-base`) are good starting points. But for truly excellent retrieval in a specific domain (like human rights law), we can make them even better through **fine-tuning**.

**Fine-tuning** is like taking a generally smart model and giving it extra training on specific examples from our target domain. This helps it understand the specific vocabulary and nuances of that domain better.

In advanced RAG systems, we often use *two* types of fine-tuned models for retrieval, balancing speed and accuracy:

1.  **Bi-Encoder (The Fast Sorter):**
    * **How it works:** Creates a single vector (embedding) for the query and separate vectors for each document *independently*. Then, it quickly calculates similarity scores between the query vector and all document vectors.
    * **Analogy:** Like quickly scanning index cards (embeddings) for keywords related to your query.
    * **Goal:** Efficiently narrow down a large number of documents to a smaller set of *potential* candidates.
    * **Our Model:** `modernbert-embed-base-biencoder-human-rights` (after fine-tuning).

2.  **Cross-Encoder (The Careful Reader):**
    * **How it works:** Takes the query AND a single candidate document *together* as input and outputs a single score indicating relevance. It looks at both texts simultaneously, allowing for deeper interaction analysis.
    * **Analogy:** Taking the few promising books found by the index card search and actually reading their summaries (processing query + document together) to decide which is *most* relevant.
    * **Goal:** Accurately re-rank the candidates provided by the bi-encoder. Much slower per comparison, but more accurate.
    * **Our Model:** `modernbert-embed-base-crossencoder-human-rights` (after fine-tuning).

**Synthetic Data:** Often, we don't have thousands of perfectly labeled `(query, relevant_document, irrelevant_document)` examples needed for fine-tuning. So, we *generate* them! We can use powerful LLMs to read our documents and create realistic questions, find relevant snippets (positives), and find unrelated snippets (negatives). This is **synthetic data**.

This tutorial uses pre-generated synthetic data to fine-tune ModernBERT for both bi-encoder (retrieval) and cross-encoder (re-ranking) tasks, then builds a RAG pipeline.

---

Okay, let's walk through the code, keeping these concepts in mind.

### Step 0: Setup - Getting the Tools

First, we install the Python libraries we need.
* `torch`: The core library for deep learning operations.
* `datasets`: Hugging Face library for easily loading and manipulating datasets.
* `sentence-transformers`: A fantastic library specifically designed for working with and training embedding models (both bi- and cross-encoders).
* `haystack-ai`: A framework for building NLP pipelines, including RAG. Think of it as providing building blocks (components).
* `transformers`: The main Hugging Face library for accessing models and tokenizers.

```python
# In [ ]:
!pip install torch datasets sentence-transformers haystack-ai
!pip install git+https://github.com/huggingface/transformers.git # for the latest version
```

Now, import the specific functions and classes we'll use.

```python
# In [1]:
# (Imports remain the same as the previous version)
import torch
# ... other imports ...
from haystack.utils.hf import HFGenerationAPIType
```

### Step 1: Configuration - Naming Things

We set up some names and detect our hardware.
* `MODEL`: The base embedding model we start with (`nomic-ai/modernbert-embed-base`).
* `REPO_NAME`: Your username on Hugging Face Hub (where models can be shared). **Replace `"sdiazlor"` with your username if you plan to upload.**
* `MODEL_NAME_...`: Names for our *new*, fine-tuned models.
* `device`: Automatically checks if you have a compatible GPU (NVIDIA uses 'cuda', Apple Silicon uses 'mps') or falls back to 'cpu'. Training is much faster on a GPU!

```python
# In [2]:
MODEL = "nomic-ai/modernbert-embed-base"
REPO_NAME = "sdiazlor" # Replace with your HF username if needed
MODEL_NAME_BIENCODER = "modernbert-embed-base-biencoder-human-rights"
MODEL_NAME_CROSSENCODER = "modernbert-embed-base-crossencoder-human-rights"

# Detect hardware
# ... (device detection logic remains the same) ...
print(f"Using device: {device}")
```

### Step 2: Pre-processing the Synthetic Data - Getting Training Material Ready

We load our synthetically generated data, which we assume is stored on the Hugging Face Hub. This data contains examples designed to teach our models about human rights documents.

```python
# In [3]:
# Load datasets from your Hugging Face Hub repository
dataset_rag_from_file = load_dataset(f"{REPO_NAME}/rag-human-rights-from-files", split="train")
dataset_rag_from_prompt = load_dataset(f"{REPO_NAME}/rag-human-rights-from-prompt", split="train")
combined_rag_dataset = concatenate_datasets([dataset_rag_from_file, dataset_rag_from_prompt])
print(combined_rag_dataset)
```

**Data Cleaning:** We remove any incomplete examples and shuffle the data randomly to ensure our model doesn't learn any unintended order during training.

```python
# In [ ]:
# ... (filter_empty_or_nan function definition) ...
filtered_rag_dataset = combined_rag_dataset.filter(filter_empty_or_nan).shuffle(seed=42)
print(filtered_rag_dataset)
```

**Formatting for Models:** The `sentence-transformers` library needs the data structured in specific ways depending on the training goal.
* **Bi-encoder Training (using TripletLoss):** Needs three pieces of text per example:
    * `anchor`: The reference text (e.g., the original context or a query).
    * `positive`: A text that *should* be close to the anchor (e.g., a relevant document passage or the context for a query).
    * `negative`: A text that *should* be far from the anchor (e.g., an irrelevant document passage).
    The goal of training is to pull `anchor` and `positive` vectors closer, while pushing `anchor` and `negative` vectors apart in the embedding space.
* **Cross-encoder Training:** Needs pairs of text and a score indicating their similarity.
    * We'll pair `anchor` (context) and `positive` (relevant retrieval).
    * We need a `score` for how similar these two are. Since our synthetic data doesn't have this score directly, we'll generate it next.

```python
# In [ ]:
# ... (rename_and_reorder_columns function definition) ...

# For Bi-Encoder Training (anchor, positive, negative)
clean_rag_dataset_biencoder = rename_and_reorder_columns(
    filtered_rag_dataset,
    rename_map={"context": "anchor", "positive_retrieval": "positive", "negative_retrieval": "negative"},
    selected_columns=["anchor", "positive", "negative"],
)

# For Cross-Encoder Training (anchor, positive) - score added next
clean_rag_dataset_crossencoder = rename_and_reorder_columns(
    filtered_rag_dataset,
    rename_map={"context": "anchor", "positive_retrieval": "positive"},
    selected_columns=["anchor", "positive"],
)

print("Bi-encoder data format:", clean_rag_dataset_biencoder)
print("Cross-encoder data format (before scores):", clean_rag_dataset_crossencoder)
```

**Generating Scores for Cross-Encoder:** How do we get the similarity scores needed to train our cross-encoder? We use a *proxy*: another pre-trained (cross-encoder) model to estimate the similarity between our `(anchor, positive)` pairs. These estimated scores become the target labels our *own* cross-encoder will learn to predict.

```python
# In [ ]:
print("Generating similarity scores for cross-encoder training data...")
# Using a pre-trained model to predict similarity scores
model_reranking = CrossEncoder(
    model_name="Snowflake/snowflake-arctic-embed-m-v1.5", device=device
)

def add_reranking_scores(batch):
    pairs = list(zip(batch["anchor"], batch["positive"]))
    # predict() outputs scores indicating similarity for each pair
    batch["score"] = model_reranking.predict(pairs, show_progress_bar=True)
    return batch

# Apply the function
clean_rag_dataset_crossencoder = clean_rag_dataset_crossencoder.map(
    add_reranking_scores, batched=True, batch_size=32 # Smaller batch_size might be needed depending on memory
)
print("Cross-encoder data format (with scores):", clean_rag_dataset_crossencoder)
```

**Train/Evaluation Split:** Just like in standard ML practice, we split our prepared data into a training set (used to update the model's parameters) and an evaluation set (used to check how well the model is learning on unseen data).

```python
# In [ ]:
# ... (split_dataset function definition) ...
dataset_rag_biencoder = split_dataset(clean_rag_dataset_biencoder)
dataset_rag_crossencoder = split_dataset(clean_rag_dataset_crossencoder)
print("Bi-encoder data splits:", dataset_rag_biencoder)
print("Cross-encoder data splits:", dataset_rag_crossencoder)
```

### Step 3: Fine-Tuning the Bi-Encoder (The Fast Sorter)

Now we train our first model: the bi-encoder, responsible for fast initial retrieval.

**Load Base Model:** We load the base ModernBERT model using `SentenceTransformer`. `gradient_checkpointing` is a memory-saving trick useful during training.

```python
# In [ ]:
print(f"Loading base model {MODEL} for bi-encoder fine-tuning...")
model_biencoder = SentenceTransformer(MODEL)
model_biencoder.gradient_checkpointing_enable()
print("Base model loaded.")
```

**Choose Loss Function:** We select `TripletLoss`. As explained before, this loss function is ideal for training retrieval models because it directly optimizes for making relevant pairs closer and irrelevant pairs farther apart in the embedding space.

```python
# In [ ]:
loss_biencoder = TripletLoss(model=model_biencoder)
print(f"Using TripletLoss for training.")
```

**Configure Training:** We set up the training parameters (epochs, batch size, learning rate, etc.). These control *how* the model learns. `load_best_model_at_end=True` is important – it ensures we save the version of the model that performed best on the evaluation set during training.

```python
# In [ ]:
# ... (SentenceTransformerTrainingArguments definition, same as before) ...
print("Training arguments configured.")
```

**Set Up Evaluator:** The `TripletEvaluator` will measure how accurately the model places positive examples closer to the anchor than negative examples during the evaluation phases.

```python
# In [ ]:
triplet_evaluator = TripletEvaluator(
    anchors=dataset_rag_biencoder["eval"]["anchor"],
    positives=dataset_rag_biencoder["eval"]["positive"],
    negatives=dataset_rag_biencoder["eval"]["negative"],
    name="eval" # Name for logging
)
print("Triplet evaluator configured.")
```

**Train!** This is where the actual learning happens. The `SentenceTransformerTrainer` handles the training loop, feeding batches of data to the model, calculating the loss, and updating the model's weights. This takes time!

```python
# In [ ]:
trainer = SentenceTransformerTrainer(
    model=model_biencoder,
    args=training_args,
    train_dataset=dataset_rag_biencoder["train"],
    eval_dataset=dataset_rag_biencoder["eval"],
    loss=loss_biencoder,
    evaluator=triplet_evaluator,
)
print("Starting bi-encoder training...")
trainer.train()
print("Bi-encoder training finished.")
```

**Save and Upload:** We save our newly fine-tuned bi-encoder locally and (optionally) upload it to the Hugging Face Hub so we (or others) can easily use it later.

```python
# In [ ]:
# ... (save_pretrained and push_to_hub commands, same as before) ...
print(f"Fine-tuned bi-encoder saved locally and pushed to HF Hub as {REPO_NAME}/{MODEL_NAME_BIENCODER}")
```

### Step 4: Fine-Tuning the Cross-Encoder (The Careful Reader)

Next, we train the cross-encoder, which will carefully re-rank the documents selected by the bi-encoder.

**Prepare Data:** The `CrossEncoder` expects data as `InputExample` objects, where each example contains the pair of texts and the target similarity score we generated earlier.

```python
# In [ ]:
print("Preparing data for cross-encoder training...")
train_samples = [InputExample(texts=[row["anchor"], row["positive"]], label=float(row["score"])) for row in dataset_rag_crossencoder["train"]]
eval_samples = [InputExample(texts=[row["anchor"], row["positive"]], label=float(row["score"])) for row in dataset_rag_crossencoder["eval"]]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8) # Adjust batch size based on memory
print("Data prepared.")
```

**Load Base Model:** We initialize the `CrossEncoder` *using the same base ModernBERT model*. The key difference is `num_labels=1`. This tells the `CrossEncoder` to add a "head" to the model designed to output a single number (our predicted similarity score), treating it as a regression problem.

```python
# In [ ]:
print(f"Loading base model {MODEL} for cross-encoder fine-tuning...")
# We specify num_labels=1 because we want the model to predict a single similarity score (regression)
model_crossencoder = CrossEncoder(model_name=MODEL, num_labels=1, device=device)
print("Base model loaded.")
```

**Set Up Evaluator:** `CECorrelationEvaluator` measures how well the model's predicted scores correlate with the "true" scores (the ones we generated with the proxy model) on the evaluation set. A higher correlation means the model is learning to rank pairs effectively.

```python
# In [ ]:
evaluator = CECorrelationEvaluator.from_input_examples(eval_samples, name="eval")
print("Correlation evaluator configured.")
```

**Train!** The `CrossEncoder` uses a slightly different training interface (`.fit` method). It handles the process of feeding text pairs to the model and training it to predict the target scores.

```python
# In [ ]:
# Calculate warmup steps (common practice: ~10% of total training steps)
num_training_steps = len(train_dataloader) * 3 # dataloader length * epochs
warmup_steps = int(num_training_steps * 0.1)

print("Starting cross-encoder training...")
model_crossencoder.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=3,
    warmup_steps=warmup_steps,
    output_path=f"models/{MODEL_NAME_CROSSENCODER}",
    save_best_model=True,
)
print("Cross-encoder training finished.")

```

**Save and Upload:** Save the fine-tuned cross-encoder.

```python
# In [ ]:
# ... (save_pretrained and push_to_hub commands, same as before) ...
print(f"Fine-tuned cross-encoder saved locally and pushed to HF Hub as {REPO_NAME}/{MODEL_NAME_CROSSENCODER}")

```

### Step 5: Building the RAG Pipeline with Haystack (Putting it all Together)

Now we use the Haystack library to assemble our fine-tuned models and an LLM into a complete RAG system. Think of Haystack components as Lego bricks for building NLP applications.

**Prepare Documents for Haystack:** We take some sample documents (just the context part) and load them into Haystack's `Document` format. For a real application, you'd load all your relevant documents here.

```python
# In [4]:
print("Preparing documents for Haystack Document Store...")
# ... (loading df, sampling, creating Document objects, same as before) ...
# Using a sample of 100 documents for this example
df = df.sample(n=100, random_state=42)
dataset = Dataset.from_pandas(df)
docs = [Document(content=doc["context"]) for doc in dataset]
print(f"Prepared {len(docs)} documents.")
```

**Initialize Components (The Lego Bricks):**
* `InMemoryDocumentStore`: A simple vector store that holds our documents and their embeddings in memory.
* `SentenceTransformersDocumentEmbedder`: Uses our fine-tuned *bi-encoder* to create the vector embeddings for each document.
* `SentenceTransformersTextEmbedder`: Uses our fine-tuned *bi-encoder* to create the vector embedding for the user's query.
* `InMemoryEmbeddingRetriever`: Performs the fast similarity search using the query embedding and the document embeddings stored in the `DocumentStore`. It retrieves an initial list of candidate documents.
* `SentenceTransformersDiversityRanker`: Uses our fine-tuned *cross-encoder* to re-rank the candidates provided by the retriever, aiming for the most relevant results.
* `ChatPromptBuilder`: Takes the user's query and the final ranked documents and formats them nicely into a prompt for the LLM.
* `HuggingFaceAPIChatGenerator`: Connects to an LLM (like Llama 3.1) hosted on Hugging Face's infrastructure to generate the final answer based on the prompt. **Requires a Hugging Face API token.**

```python
# In [ ]:
print("Initializing Haystack components...")
# Document Store & Embedding Documents
document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model=f"{REPO_NAME}/{MODEL_NAME_BIENCODER}")
doc_embedder.warm_up() # Load the model into memory
print("Embedding documents...")
docs_with_embeddings = doc_embedder.run(docs)["documents"]
document_store.write_documents(docs_with_embeddings)
print(f"Documents embedded and stored.")

# Query Embedder (uses the same fine-tuned bi-encoder)
text_embedder = SentenceTransformersTextEmbedder(model=f"{REPO_NAME}/{MODEL_NAME_BIENCODER}")

# Retriever (finds candidates using bi-encoder embeddings)
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=10) # Ask for top 10 candidates

# Ranker (re-ranks candidates using the fine-tuned cross-encoder)
ranker = SentenceTransformersDiversityRanker(model=f"{REPO_NAME}/{MODEL_NAME_CROSSENCODER}", top_k=3) # Keep top 3 after re-ranking

# Prompt Builder (formats input for LLM)
template = """
Given the following relevant information from human rights documents, answer the question concisely.
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = ChatPromptBuilder(template=template)

# LLM Generator (gets the final answer)
# Make sure HF_TOKEN environment variable is set!
try:
    chat_generator = HuggingFaceAPIChatGenerator(
        api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
        api_params={"model": "meta-llama/Llama-3.1-8B-Instruct"},
        token=Secret.from_env_var("HF_TOKEN"),
    )
    print("LLM Generator initialized.")
except Exception as e:
    print(f"Error initializing LLM Generator: {e}. Did you set the HF_TOKEN environment variable?")
    chat_generator = None # Set to None to avoid errors later if init fails

print("Haystack components initialized.")

```

**Build and Connect Pipeline:** We create an empty `Pipeline` object and add our components ("bricks") to it. Then, we connect the output of one component to the input of the next, defining the data flow: Query -> Embed -> Retrieve -> Rank -> Build Prompt -> Generate Answer.

```python
# In [ ]:
print("Building the RAG pipeline...")
rag_pipeline = Pipeline()

# Add components
rag_pipeline.add_component("text_embedder", text_embedder) # Query embedder
rag_pipeline.add_component("retriever", retriever)         # Bi-encoder retriever
rag_pipeline.add_component("ranker", ranker)             # Cross-encoder ranker
rag_pipeline.add_component("prompt_builder", prompt_builder) # Formats prompt
if chat_generator: # Only add LLM if it initialized correctly
    rag_pipeline.add_component("llm", chat_generator)             # LLM generator

# Connect components (Define the data flow)
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding") # Query vector -> Retriever
rag_pipeline.connect("retriever.documents", "ranker.documents")             # Retrieved docs -> Ranker
rag_pipeline.connect("ranker.documents", "prompt_builder.documents")        # Ranked docs -> Prompt Builder

# We also need the original query for the Ranker and Prompt Builder
# We will pass it in the .run() call directly to these components

if chat_generator:
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")          # Final prompt -> LLM

print("Pipeline built and connections made.")
print(rag_pipeline) # Shows the pipeline structure
```

### Step 6: Running the Pipeline - Asking Questions!

Let's test our RAG system. We call `rag_pipeline.run()` and provide the necessary inputs: the question text needs to go to the `text_embedder` (to get the query vector), the `ranker` (as the cross-encoder needs the query text), and the `prompt_builder`.

```python
# In [ ]:
if chat_generator: # Only run if the LLM generator is available
    question1 = "How many human rights there are?"
    print(f"\n--- Asking: {question1} ---")
    response1 = rag_pipeline.run(
        {
            "text_embedder": {"text": question1},
            "prompt_builder": {"question": question1},
            "ranker": {"query": question1},
        }
    )
    # Accessing the content of the ChatMessage reply
    print("Response:", response1["llm"]["replies"][0].content)

    question2 = "What's the Right of Fair Trial?"
    print(f"\n--- Asking: {question2} ---")
    response2 = rag_pipeline.run(
        {
            "text_embedder": {"text": question2},
            "prompt_builder": {"question": question2},
            "ranker": {"query": question2},
        }
    )
    print("Response:", response2["llm"]["replies"][0].content)
else:
    print("\nSkipping pipeline run because LLM generator failed to initialize (check HF_TOKEN).")

```

The quality of the answers will depend on whether relevant information was present in the 100 documents we loaded into our `InMemoryDocumentStore` and how well our fine-tuned models perform at retrieving and ranking them.

### Wrapping Up: Your First RAG Pipeline!

Congratulations! You've walked through the process of building a sophisticated RAG system.

**What you learned:**

1.  **What RAG is:** Combining retrieval (search) with LLMs to get more accurate, up-to-date, and domain-specific answers.
2.  **Why fine-tuning helps:** Making general embedding models better at understanding specific topics (like human rights law).
3.  **Bi- vs. Cross-Encoders:** The trade-off between speed (bi-encoder for initial search) and accuracy (cross-encoder for re-ranking).
4.  **Key Libraries:** How `sentence-transformers` helps train embedding models and `haystack-ai` helps build the pipeline components.
5.  **The Process:** Data Prep -> Bi-Encoder Training -> Cross-Encoder Training -> Pipeline Assembly -> Querying.

This is a foundational example. From here, you could explore using larger base models, improving synthetic data generation, using more powerful vector databases, and integrating different LLMs. RAG is a rapidly evolving field, but the core principles you learned here provide a strong starting point for building powerful AI applications.
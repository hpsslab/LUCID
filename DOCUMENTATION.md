# LUCID: Literature Understanding and Consolidation for Intelligent Discovery
# Shaun's Additions (more detail)

## Add Files/Directories to KG
- Adds each filepath (or recursively adds everything in a directory) to the KG. Does this asynchronously. Only works on PDFs for now.
- `python3 main.py add [options] <path>+`

### 1. Call asyncio.gather() to asynchronously add papers to the KG.
- The path option includes 1 or more paths to papers/articles whose contents we want to add to the KG. Do this asynchronously.

### 2. Make the Knowledge Graph.
- TODO: improve parameters/hyperparameters.

1. LLM (only have Gemini programmed in for now). 
- This will answer CQ questions and be the LLM used to do things like make KG triplet output readable and extract triplets from text samples.

2. Embedding Model (tried to use the domain specific ones like SciBERT and MatSciBERT, but my hardware wasn't gonna handle that, so that still needs to be tested). Just using API calls to Gemini's embedding model for now. 
- This model will embed text samples and queries and will eventually be used to compare KG nodes with the query to determine relevance.

3. Transformations (how do we split up the work?)
- Right now, we use LlamaIndex's SemanticSplitterNodeParser to split CQ answers into chunks to divide the labor of parsing for triplets.
- The embedding model embeds the sentences of the text sample and decides which sentences to group together.
- I'm not sure we should be doing this TBH. Yes, we can use this to divide the work of extracting triplets, but this may not be something worth parallelizing. 
- I think we may be losing context in these chunks when we divide them and that may be decreasing the quality of the triplets extracted. 
- Something to think about; should we just serialize this instead ans get rid of this entirely?

4. KG Extractor (how do we extract triplets from our CQ answer text samples?)
- TODO: write more detail here.
- TODO: it seems like the LLM treats our schema as more of a suggestion (as opposed to a requirement), look into this?

5. Check memory and see if we already make a KG (would be the path at KGSTORAGEPATH, as seen by the constant in addToKG.py#L19).
- Would be bad if we made a new KG every time as opposed to appending to an existing KG when it already exists.

### 3. Add each individual paper to the KG (asynchronously).
- Make sure we are passed a .pdf file (for now, TODO: add more types later, see addToKG.py#L20?).
- Make sure we have not already added the paper to the KG. 
- The file databases/kgContents.txt contains hashes for every paper that has been added to the KG. Check for the paper's hash before we add the paper in.
- The duplicate check might be redundant (not sure if LlamaIndex already handles this), but the functionality is there. :D
- Next, we answer the CQs for the given paper. From there, we pass CQ answers into a routine that adds its passed contents into the KG (abstracted away to LlamaIndex).

- GOAL of insert: 
1. Run transformations on CQ answers (as seen in step 2).
2. Extract information from the text according to our schema (as seen in step 2).
3. Add those triplets into a JSON file, which represents our KG.
- This is where the LlamaIndex code needed to be revised by me. This ainsert function assumes that the caller isn't asynchronously doing anything.
- I added a few lines of code to fix that.
- TODO: make sed script that we can run so that you all don't get an "coroutine not awaited" issue when running main.

### 4. Save our changes to KGSTORAGEPATH, as seen by the constant in addToKG.py#L19

## Query a LLM using a KG
- Queries a LLM using the knowledge graph that has been constructed.
- `python3 main.py query [options]`

### 1. Load Knowledge Graph from memory (assuming by default that the location is KGSTORAGEPATH, as seen by the constant in addToKG.py#L19)
- TODO: make the database representing the KG something more sophisticated? 
- We are just storing JSONs in RAM to store this stuff for now. Would be bad as the scale of the KG increases.

### 2. Make a query engine out of the KG (abstracted away to the library, but goal is to use embeddings and similarity metrics like cosine similarity to find relevant triplets). 
- TODO: consider hyperparameters. 
- How many triplets do we identify (top k, what should k be)? 
- We do a iterative deepening like search from identified nodes; what should that depth be? Too low: we have basic level context (ie: 1 would mean only 1 triplet, not enough to make connections, most likely). Too high: we have unnecessary context.
- How many triplets do we fetch in total?

### 3. Make the LLM agent (including making the query engine and passing it to the agent) 
- Use LangChain/LangGraph to abstract the process of making the KG usable for the agent away to the library.
- Consider using .asretriever() function instead of .asqueryengine(). 
- Retriever would give us the relevant triplets directly, while query engine passes the triplets into an LLM to summarize the triplets into something more readable as an intermediate step.
- Stretch Goal: a better UI?

# RAG Chatbot for NHS Medicines

This chatbot allows to you to talk to the [NHS Medicines A-Z](https://www.nhs.uk/medicines/), a site containing information for patients about common medicines. It will answer only based on this information and provide direct links to the pages or even the paragraphs used.

This is a personal exploration of Agentic Retrieval Augmented Generation (RAG) models and their potential for interacting reliably with verified information sources. It utilizes the OpenAI API, LangChain, Chroma and LangGraph to query a vector database built from the freely accessible [NHS Medicines API](https://developer.api.nhs.uk/nhs-api/documentation/5b8e85b396097ba52552d63b). While conventional RAG is effective for simple queries, this solution uses multiple agents working in concert to drastically improve the retrieval search and filter out irrelevant context.

![chatbot-narrow](https://github.com/shyamdhokia1/Medicines-RAG-ChatBot/assets/92919658/24b35203-6461-4328-944c-a253f9e7a7b8)

**Important Disclaimer:**

* This is a personal project for educational and testing purposes only. 
* It is **not** developed in collaboration with any NHS organization.
* Regulation as a medical device has not been explored yet and therefore the chatbot should not be hosted and made accessible to the public.

## Technologies

* **Backend**
    * Python
    * Flask
    * OpenAI API
    * LangChain
    * LangGraph
    * ChromaDB

* **Frontend**
    * React.js
    * TailwindCSS
    * DaisyUI
    * Vite

## Optimising the RAG
![Task management](https://github.com/shyamdhokia1/Medicines-RAG-ChatBot/assets/92919658/2f5d60bb-6717-4e81-b9d8-f15dee001dcc)

### Pre-Retrieval
The text from the NHS Medicines API is converted into **Markdown** and stored alongside the JSON metadata in **LangChain Documents**. LLMs are fine-tuned on Markdown text so it is the most effective format for 'consumption' by LLMs. The metadata will allow us to augment our search using keywords later, as well as display URLs to users.

`chunk_size=512, chunk_overlap=64` ensures that the majority of paragraphs are preserved as entire units of information to preserve context and improve semantic retrieval. Larger paragraphs retain leading information due to the overlap.

### Retrieval
#### Guardrails (verify)
The `verify` edge act as a straightforward guardrail, rejecting questions that are inappropriate given the chatbot's purpose and preventing unnecessary retrieval. The `reject` node will then give a polite response and redirect to appropriate resources.

#### Query Translation (rewrite)
The `rewrite` node translates the query into a clearer version, correcting spelling errors and adding
generic names for medications, increasing efficiency of retrieval.

#### Query Expansion (retrieve)
Query expansion creates N versions of the query from alternate perspectives, then performs a seperate search for each to maximise the embedding area. This results in more relevant chunks being retrieved.

#### Self Query/Hybrid Retrieval (retrieve)
Self query extracts metadata search queries from the original query. A search is then done using a hybrid of semantic similarity and keyword similarity (using the metadata query). With organised data like this, we can effectively find relevant chunks that would otherwise be missed.

### Post Retrieval
The `rerank` node uses `FlashRank` to rerank our retrieved chunks, returning only the top N most relevant ones. This reduces noises from irrelevant context, reduces prompt size, and helps prevent the "lost in the middle" problem.

## API Scraping

NHS-medicines-scraper.py calls the NHS Medicines API to create Markdown files and Documents on all available medicines. A new version of the API has been released so the current scraper is outdated.

Here's an overview of the API structure:

**Example Responses:**

* **Base URL:** [https://api.nhs.uk/medicines](https://api.nhs.uk/medicines)

```
{
  "significantLink": [
    {
      "name": "drugname",
      "url": "API url",
      "mainEntityOfPage": {
        "dateModified": "DATE"
      }
    }
  ]
}
```

* **Individual Medication URL:** https://api.nhs.uk/medicines/medicationname

```
{
  "name": "drugname",
  "about": {
    "alernateName": "alternatename"
  },
  "description": "page description",
  "url": "page url",
  "hasPart": [
    {
      "@type": "HealthTopicContent",
      "url": "segment url",
      "description": "segment description",
      "hasPart": [
        {
          "text": "text in html",
          "headline": "headline for paragraph (often empty)"
        },
        {
          "text": "text in html",
          "headline": "headline for paragraph (often empty)"
        }
      ],
      "headline": "headline for whole segment"
    }
  ],
  "mainEntityOfPage": [
    {
      "name": "keylinks",
      "mainEntityOfPage": [
        {
          "headline": "headline for segment",
          "url": "url"
        }
      ]
    },
    {
      "name": "RelatedLinks",
      "headline": "Related conditions/resources",
      "mainEntityOfPage": [
        {
          "headline": "name of condition",
          "url": "NHS overview url"
        }
      ]
    }
  ]
}
```
## Running a Test Environment:

Clone this repo.

To run this project, you'll need the following Python libraries:

```pip install -U flask langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph markdownify flashrank lark```

In `.env` set the `LANGCHAIN_API_KEY` for LangTrace tracing and `NHS_API_KEY` for the scraper.

Using CUDA with local GPU is recommended for running the local embedding model. Fully local LLM is planned for future development.

### Backend
```
cd backend

# Scrape the pages for all medications
NHS-medicines-scraper.py

# Run Flask app
app.py
```

### Frontend
```
cd frontend
npm run dev
```
## Getting Involved
If this project interests you and you'd like to get involved please email me at sergiou1923@gmail.com

## RAG Chatbot for NHS Medicines

This project is a personal exploration of agentic Retrieval Augmented Generation models and their potential for interacting with verified information sources. It utilizes the OpenAI API, LangChain, and LangGraph to query a vector database built from the freely accessible NHS Medicines API.

While standard RAG excels at simple queries across a few documents, agentic RAG takes it a step further and emerges as a potent solution for question answering. It introduces a layer of intelligence by employing AI agents. These agents act as autonomous decision-makers, rewording the initial prompt to ensure efficient retrieval search, verifying the retrieved documents relevance to the question and strategically selecting the most tools for further data retrieval.

**Important Disclaimer:**

* This is a personal project for educational and testing purposes only. 
* It is **not** affiliated with or endorsed by any NHS organization.
* It does not currently adhere to the requirements for using NHS website syndicated content in a product, such as providing direct URLs to the content utilised in the search. This featuer is planned for future development.
* In addition the chatbot has not undergone regulation as a medical device and therefore should not be hosted and made accessible to the public.

**Functionality:**

This chatbot allows users to interact with information about common medicines based on the official NHS patient advice. It utilizes the NHS Medicines API to retrieve relevant data, convert it into LangChain Documents and then chunk and store the data in a vector database, ready for semantic search.

When you ask the chatbot a question it will retrieve relevant documents and use the content to provide an informed answer

**Technologies:**

* Backend
    * Python
    * Flask
    * OpenAI API
    * LangChain
    * LangGraph
    * Chroma

* Frontend
    * React.js
    * TailwindCSS
    * DaisyUI

**API Scraping**

NHS-medicines-scraper.py interacts with the NHS Medicines API to create Markdown files and Documents on all available medicines. You will require a subscription key - free trial subscription is available.

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
## Requirements:

To run this project, you'll need the following Python libraries:

```pip install -U flask langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph markdownify```

In the `.env` file please also set the `LANGCHAIN_API_KEY` for tracing and `NHS_API_KEY` for the scraper.

## Links
* NHS Medicines API Documentation: https://developer.api.nhs.uk/nhs-api/documentation/5b8e85b396097ba52552d63b
* NHS Website Syndicated Content Standard License Terms: https://developer.api.nhs.uk/about/terms/latest
* Much thanks to this excellent guide by AI Jason that was integral in developing this project: https://www.youtube.com/watch?v=u5Vcrwpzoz8

## Getting Involved
If this project interests you and you'd like to get involved please email me at shyamdhokia123@hotmail.co.uk
# E-commerce Customer Support System using CrewAI

An AI-powered multi-agent customer support system built using **CrewAI** and **Google Gemini 2.5 Flash**. The system automates customer support by classifying issues, interpreting business policies, generating customer responses, and determining when a case should be escalated.

## Features

- Multi-agent architecture using CrewAI
- Customer issue classification
- Policy-aware response generation
- Rule-based escalation
- Structured JSON communication between agents
- Sequential agent orchestration
- Google Gemini 2.5 Flash integration

## Tech Stack

- Python
- CrewAI
- LangChain
- Google Gemini 2.5 Flash
- python-dotenv

## Repository Structure

```
.
├── main.py
├── requirements.txt
├── README.md
├── Architecture Summary.pdf
└── screenshots/
```

## Documentation

The repository includes an **Architecture Summary** that explains:

- Multi-agent workflow
- Agent responsibilities
- Task handoffs
- Sequential execution flow
- Escalation logic
- Design rationale
- Sample scenarios

## Sample Use Cases

- Wrong item delivered
- Damaged product
- Refund requests
- Return requests
- Delivery issues
- Policy exception scenarios
- Customer safety concerns

## Future Improvements

- Integrate a vector database for policy retrieval (RAG)
- Replace static policies with an enterprise knowledge base
- Build a FastAPI interface
- Add persistent conversation history
- Containerize using Docker

## Author

**Vyshnavi Kadali**

# 🏠 Elder Care AI Assistant

A compassionate, intelligent AI assistant designed specifically for elderly users, built with LangGraph and modern AI technologies.

## ✨ Features

- **Elder-Friendly Design**: Simplified language, patient responses, and clear communication
- **Persistent Memory**: Remembers personal details, health information, and preferences
- **Health Awareness**: Provides medication reminders and health-related guidance
- **Entertainment Discovery**: Finds podcasts, news, and content tailored to interests
- **Multi-Platform Support**: API, web interface, and voice-ready responses
- **Privacy-First**: Local SQLite database with secure memory management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd LangGraph-tool-testing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Test the local chat**
   ```bash
   python scripts/run_local.py
   ```

### API Server

1. **Start the server**
   ```bash
   python -m src.api.app
   ```

2. **Test the API**
   ```bash
   python scripts/test_chat.py
   ```

3. **View API docs**
   Visit `http://localhost:8000/docs`

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   Preprocessor   │───▶│   Agent Core    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           │
│ Final Response  │◀───│  Postprocessor   │◀──────────┘
└─────────────────┘    └──────────────────┘
                                │
                       ┌──────────────────┐
                       │ Memory & Storage │
                       └──────────────────┘
```

### Core Components

- **LangGraph Workflow**: State-based conversation flow
- **Memory System**: SQLite-based persistent storage
- **Tool Registry**: Extensible tool system for external integrations
- **Elder-Friendly Processing**: Specialized formatting and safety checks

## 🛠️ Tools & Capabilities

### Built-in Tools

1. **Podcast Discovery** 📻
   - Finds podcasts based on interests
   - Age-appropriate content filtering
   - Easy listening recommendations

2. **News Updates** 📰
   - Current events summaries
   - Health and wellness news
   - Local community information

3. **Health Reminders** 💊
   - Medication schedules
   - Doctor appointment tracking
   - Health tip suggestions

### Memory Types

- **Personal**: Names, family, hobbies, background
- **Health**: Conditions, medications, doctor visits
- **Preferences**: Likes, dislikes, routines
- **Relationships**: Family members, friends, caregivers
- **Important Events**: Birthdays, appointments, milestones

## 🌟 Example Conversations

```
You: "Hello, I'm feeling a bit lonely today."
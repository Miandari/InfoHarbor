# Information Assistant with LangGraph 🤖

A versatile conversational assistant framework powered by LangGraph, designed for modular tool integration with an expandable architecture.



## 🎯 Overview

This project demonstrates how to build an intelligent information assistant using LangGraph, a library for creating stateful multi-agent workflows. The assistant currently provides:

- 🎧 **Podcast Recommendations**: Find podcasts based on interests, topics, or similar to existing podcasts
- 📰 **News Updates**: Get recent news on various topics
- 💬 **Conversational Interface**: Natural language interaction with context retention

These initial tools showcase the system's capabilities, with the architecture designed for easy integration of additional tools and information sources.

## 🛠️ Architecture

The application uses a flexible LangGraph workflow with a modular design:

- **Main Agent**: Central router that intelligently dispatches requests to appropriate tools based on user queries
- **Tool Nodes**: Specialized modules that can be easily added to expand system capabilities
- **Response Handler**: Formats information into coherent, conversational responses
- **State Management**: Maintains context across interactions

### System Workflow

![Info Assistant Workflow](info_assistant_workflow.png)

The diagram above illustrates the system's workflow architecture, showing how queries are processed through the main agent and routed to specialized tool nodes.

## 🌐 API Service

The Dastyar assistant can be deployed as a REST API service, allowing other applications to access its capabilities.

### API Endpoints

- **POST /chat** - Process a chat message and return a response
- **POST /chat/stream** - Stream chat responses for real-time UI updates
- **GET /conversation/:id** - Get conversation history by ID
- **GET /tools** - List available tools
- **POST /tools/toggle** - Enable/disable specific tools
- **GET /health** - Health check endpoint for monitoring

### Running the API

```bash
# Install additional dependencies
pip install fastapi uvicorn websockets redis python-jose python-multipart

# Start the API server
python api.py
```

The API will be available at `http://localhost:8000`.

### API Documentation

Once running, API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Security Features

The API includes:
- API key authentication
- Rate limiting
- Input validation
- Proper error handling

### Performance Optimizations

- Connection pooling with Redis
- Response caching
- Background processing for state management
- Streaming responses for real-time updates

### Deployment

For production deployment on Render:
- Follow instructions in `render_deployment_guide.md`
- Use the provided `render.yaml` for Blueprint deployment
- Configure environment variables for security and performance

### Client Integration

For details on integrating with a Node.js/React application:
- Check `client_integration.md` for examples
- Includes streaming response handling
- Demonstrates connection pooling and caching techniques

## ⚙️ Installation

1. Clone this repository
   ```bash
   git clone [repository-url]
   cd LangGraph-tool-testing
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file)
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   # Add keys for additional tools as needed
   ```

## 🚀 Usage

### Running the Assistant

```bash
python main.py "What are some good podcasts about ancient history?"
```

### Interactive Mode

```bash
python main.py --interactive
```

### Debug Mode

```bash
python main.py --debug
```

The debug mode enables detailed logging to help troubleshoot issues:

- Logs are written to `debug_log.txt` in the project root
- All API calls, responses, and internal state changes are recorded
- Great for understanding the agent's decision-making process
- Can be combined with other modes: `python main.py --debug --interactive`

Example debug command for specific queries:
```bash
python main.py --debug "What's happening in world news today?"
```

### Visualization Mode

```bash
python main.py --visualize
```

### Import in Your Python Code

```python
from main import run_info_assistant

# Run with a query
response = run_info_assistant("Tell me about recent tech news")
print(response)

# Run with conversation tracking
response = run_info_assistant("Any updates on that news?", conversation_id="user123")
print(response)
```

## 📁 Project Structure

- `main.py`: Entry point for the application
- `api.py`: FastAPI implementation for REST API service
- `graph/`: Contains the LangGraph workflow definition
  - `workflow.py`: Main workflow logic and node definitions
  - `state.py`: State management for the workflow
  - `nodes.py`: Additional node functions
- `tools/`: Specialized tools for gathering information
  - `podcast_tools.py`: Tools for podcast recommendations
  - `news_tools.py`: Tools for retrieving recent news
  - *Add your custom tools here*
- `utils/`: Helper utilities
  - `formatting.py`: Response formatting functions
  - `middleware.py`: API middleware for rate limiting, caching, and connection pooling
- `config.py`: Configuration settings
- `visualize_graph.py`: Utility to visualize the workflow architecture
- `client_integration.md`: Guide for client-side integration
- `render_deployment_guide.md`: Deployment guide for Render
- `Procfile` & `render.yaml`: Configuration files for Render deployment

## 🔧 Extending the System

To add a new tool to the assistant:

1. Create a new tool module in the `tools/` directory
2. Define your tool's functionality and API
3. Update the workflow in `graph/workflow.py` to include your new tool
4. Add any necessary formatting functions in `utils/formatting.py`
5. Update environment variables if needed

## 🧪 Testing

Use the included test notebook to explore the assistant's capabilities:

```bash
jupyter notebook test.ipynb
```

## 📋 Requirements

- Python 3.9+
- LangGraph
- LangChain
- OpenAI API key (or compatible model provider)
- Tavily API key for search capabilities
- ListenNotes API key for podcast recommendations

### Additional Requirements for API Service

- FastAPI
- Uvicorn
- Redis (optional, for caching and session management)
- Python-Jose (for API authentication)
- Python-Multipart (for form data processing)

## 🔐 Environment Setup

Create a `.env` file in the root directory with the following variables:
```
# Core API Keys
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
LISTENNOTES_API_KEY=your_listennotes_key

# API Service Configuration (optional)
API_KEYS=your_api_key1,your_api_key2
ALLOWED_ORIGINS=https://your-frontend-domain.com
REDIS_URL=redis://localhost:6379/0
API_RATE_LIMIT=100
MAX_CONVERSATION_HISTORY=50
```

Make sure to keep this file private and never commit it to your repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [Tavily](https://tavily.com/) for search capabilities
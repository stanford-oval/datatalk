# DataTalk

DataTalk is a comprehensive NL2SQL (Natural Language to SQL) system that allows users to upload CSV files and query them using natural language. The system consists of two main components: data ingestion and an intelligent agent for natural language query processing.

## Architecture Overview

The repository is organized into two distinct parts:

1. **Ingestion System** (`ingestion/` + `new_frontend/` + `serve_frontend.py`) - Handles CSV file uploads and data processing
2. **Agent System** (`agent/`) - Provides natural language to SQL query capabilities

## Components

### 1. Ingestion System

The ingestion system handles CSV file upload, processing, and database creation.

#### Key Files:
- `serve_frontend.py` - Flask web server that serves the frontend interface
- `new_frontend/` - Web interface for file uploads
- `ingestion/` - Core data ingestion logic

#### Features:
- **Web Interface**: Upload CSV files through a web interface
- **Automatic Schema Detection**: Intelligently detects column types (dates, numbers, text, etc.)
- **PostgreSQL Integration**: Creates optimized database schemas
- **Batch Processing**: Handles large CSV files efficiently
- **Data Type Conversion**: Converts various formats (percentages, dates, timestamps) to PostgreSQL types

#### Usage:
```bash
python serve_frontend.py
```
The web interface will be available at `http://localhost:5678`

### 2. Agent System

The agent system provides natural language query capabilities using advanced NL2SQL techniques.

#### Key Files:
- `agent/datatalk_runner.py` - Main agent runner
- `agent/kraken/` - Core agent implementation
  - `agent.py` - Main agent logic
  - `state.py` - State management
  - `sql_utils.py` - SQL utilities
  - `prompts/` - LLM prompts

#### Features:
- **Natural Language Processing**: Convert English queries to SQL
- **Context Awareness**: Maintains conversation history
- **SQL Generation**: Creates optimized SQL queries
- **Result Explanation**: Provides human-readable explanations of results
- **Error Handling**: Robust error handling and recovery
- **Cost Tracking**: Monitors LLM usage costs

#### Configuration:
- `agent/llm_config.yaml` - LLM configuration (OpenAI API settings)

## Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL database
- OpenAI API key (configured in environment or config file)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd datatalk
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # if available
```

3. Configure database connection and API keys in `agent/llm_config.yaml`. Please refer to the chainlite repository (https://github.com/stanford-oval/chainlite) on the details.

### Quick Start

1. **Start the ingestion server**:
```bash
python serve_frontend.py
```

2. **Upload CSV files** through the web interface at `http://localhost:5678`

3. **Query your data** using the agent:
```bash
python agent/datatalk_runner.py
```

## Workflow

1. **Data Upload**: Use the web interface to upload CSV files
2. **Automatic Processing**: The system automatically:
   - Analyzes column types
   - Creates PostgreSQL tables
   - Generates lookup tables
   - Creates database schemas
3. **Natural Language Queries**: Use the agent to ask questions about your data in plain English
4. **SQL Generation**: The agent converts your questions to SQL queries
5. **Results**: Get both the SQL query and human-readable explanations

## Directory Structure

```
datatalk/
├── agent/                          # NL2SQL Agent System
│   ├── datatalk_runner.py         # Main agent entry point
│   ├── kraken/                     # Core agent implementation
│   │   ├── agent.py               # Agent logic
│   │   ├── state.py               # State management
│   │   ├── sql_utils.py           # SQL utilities
│   │   ├── utils.py               # Utility functions
│   │   └── prompts/               # LLM prompts
│   └── llm_config.yaml            # LLM configuration
├── ingestion/                      # Data Ingestion System
│   ├── ingestion.py               # Core ingestion logic
│   ├── ingestion_createdb.py      # Database creation
│   ├── ingestion_tools.py         # Ingestion utilities
│   └── from_db/                   # Database import tools
├── new_frontend/                   # Web Interface
│   ├── templates/                 # HTML templates
│   ├── css/                       # Stylesheets
│   ├── js/                        # JavaScript files
│   └── img/                       # Images
└── serve_frontend.py              # Flask web server
```

## Features

### Data Ingestion
- Support for large CSV files (up to 10GB)
- Automatic data type detection and conversion
- PostgreSQL schema generation
- Batch processing for performance
- Declaration file support for custom schema definitions

### Natural Language Agent
- Convert English questions to SQL queries
- Conversation history and context awareness
- Multiple query types supported
- Result explanation and limitations
- Suggested follow-up questions
- Cost tracking and monitoring

### Web Interface
- Drag-and-drop file upload
- Real-time processing feedback
- Database management
- File validation and error handling

## Configuration

### Database Configuration
The system uses PostgreSQL with specific user roles:
- `creator_role` - For creating tables and schemas
- `select_user` - For read-only access

### LLM Configuration
Configure OpenAI API settings in `agent/llm_config.yaml`:
- API key and base URL
- Model selection (GPT-4o, GPT-4o-mini)
- Prompt logging settings

## Contributing

The system is designed to be modular and extensible. Key areas for contribution:
- Additional data source connectors
- Enhanced NL2SQL capabilities
- Improved web interface
- Performance optimizations

## License

[License information would go here]
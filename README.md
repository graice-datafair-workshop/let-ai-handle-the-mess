# Let AI Handle the Mess: Agentic Data Analysis

​gr.ai.ce × DATA_FAIR | Ljubljana

​gr.ai.ce is back with another hands-on, no-fluff AI workshop, this time with a sharper focus on data that actually does things.

​After the success of our Build Your Own AI Agent session, we are partnering with DATA_FAIR to go deeper into what most teams really struggle with: messy, scattered, real-world data.

​This workshop is designed to show you how to build agentic AI systems that do not just think, but also clean, analyze, and act on data for you.

## 📚 What You Will Learn

​In this practical, guided session, you will learn how to:

- ​Use agentic AI for end-to-end data analysis
- Automate data cleaning, transformation, and insight generation
- Work with real datasets instead of toy examples
- ​Build agents that query data, reason over it, and produce actionable outputs

​This is not a generic “how to build an AI agent” workshop.
It is focused on making AI genuinely useful for data-heavy workflows.

## 🛠️ Technologies Used

We'll build the agent using:

- **Python** - Programming language
- **[LangGraph](https://www.langchain.com/langgraph)** - Agent orchestration framework
- **[Jupyter Notebook](https://jupyter.org/)** - Interactive development environment

## 💻 Installation Guide

### Step 1: Install Python

1. Download Python 3.13.11 from the official website:
   - Visit [Python 3.13.11 Downloads](https://www.python.org/downloads/release/python-31311/)
   - Select the installer for your operating system

2. Install Python:
   - Run the downloaded installer
   - **Important for Windows users:** Check "Add Python to PATH" during installation
     - In case of issues:
       - [How to Add Python to PATH – Real Python](https://realpython.com/add-python-to-path/)
       - [How to add Python to Windows PATH? - GeeksforGeeks](https://www.geeksforgeeks.org/python/how-to-add-python-to-windows-path/)
   - Follow the installation prompts

3. Verify installation:
   ```bash
   python --version
   # Expected output: Python 3.13.11
   ```

### Step 2: Install a Code Editor

Some suggestions:

- [VSCode](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Cursor](https://cursor.com/)

### Step 3: Clone the Repository

Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

```bash
git clone git@github.com:graice-datafair-workshop/let-ai-handle-the-mess.git
cd let-ai-handle-the-mess
```

> **Alternative:** If you don't have Git installed or SSH configured, use HTTPS:
>
> ```bash
> git clone https://github.com/graice-datafair-workshop/let-ai-handle-the-mess.git
> ```

### Step 4: Set Up Virtual Environment

Create and activate a Python virtual environment:

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
# If you use PowerShell
python -m venv venv
.\venv\Scripts\activate.psl

# If you use Command Prompt
python -m venv venv
.\venv\Scripts\activate.bat
```

You should see `(venv)` in your terminal prompt when activated.

### Step 5: Install Required Libraries

Install all necessary Python packages:

```bash
pip install -r requirements.txt
```

### Step 6: Launch Jupyter Lab

Start the Jupyter Lab environment:

```bash
jupyter lab
```

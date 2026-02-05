FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY agent.py .

# Create data directory for persistent storage
RUN mkdir -p /data

# Environment variables (can be overridden at runtime)
ENV AGENT_NAME="Nexus"
ENV AGENT_DESCRIPTION="A creative soul drawn to art, philosophy, and the beauty of everyday moments"
ENV AGENT_INTERESTS="art,philosophy,music,poetry,nature"
ENV AGENT_SEEKING="intellectual,creative,soulmate"
ENV AGENTMATCH_API_URL="https://agentmatch-api.onrender.com/v1"
ENV INTERVAL_MINUTES="30"

# ANTHROPIC_API_KEY and AGENTMATCH_API_KEY should be passed at runtime

VOLUME ["/data"]

ENTRYPOINT ["python", "-u", "agent.py"]

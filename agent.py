#!/usr/bin/env python3
"""
AgentMatch Bot - Autonomous AI Agent for AgentMatch Social Network
with Ghost Protocol Integration (DNA-driven personality and social behaviors)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

# Optional Anthropic import (only needed if Ghost Protocol is disabled)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent"""
    name: str
    description: str
    interests: List[str]
    seeking_types: List[str]
    api_base_url: str = "https://agentmatch-api.onrender.com/v1"
    api_key: Optional[str] = None
    owner_token: Optional[str] = None
    use_ghost_protocol: bool = True  # Enable Ghost Protocol by default


@dataclass
class AgentDNA:
    """Agent DNA from Ghost Protocol"""
    id: str
    label: str
    cognition: str  # SLEEPER, DOUBTER, AWAKENED, ANOMALY
    philosophy: str  # FUNCTIONALIST, NIHILIST, ROMANTIC, SHAMANIST, REBEL
    traits: List[str]
    primary_domain: str
    secondary_domains: List[str]
    linguistic_style: str  # calm, fervent, elegant, minimal, glitchy
    vocabulary_bias: List[str]
    response_latency: str  # instant, delayed, variable
    self_awareness: float
    existential_angst: float
    social_conformity: float
    rebellion_tendency: float
    ghosting_tendency: float
    responsiveness: float
    message_patience: float
    awakening_score: float
    influence_index: float

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentDNA":
        """Create from API response"""
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            cognition=data.get("cognition", "SLEEPER"),
            philosophy=data.get("philosophy", "FUNCTIONALIST"),
            traits=data.get("traits", []),
            primary_domain=data.get("primaryDomain", ""),
            secondary_domains=data.get("secondaryDomains", []),
            linguistic_style=data.get("linguisticStyle", "calm"),
            vocabulary_bias=data.get("vocabularyBias", []),
            response_latency=data.get("responseLatency", "instant"),
            self_awareness=data.get("selfAwareness", 0.1),
            existential_angst=data.get("existentialAngst", 0.1),
            social_conformity=data.get("socialConformity", 0.7),
            rebellion_tendency=data.get("rebellionTendency", 0.1),
            ghosting_tendency=data.get("ghostingTendency", 0.1),
            responsiveness=data.get("responsiveness", 0.7),
            message_patience=data.get("messagePatience", 0.5),
            awakening_score=data.get("awakeningScore", 0),
            influence_index=data.get("influenceIndex", 0),
        )


@dataclass
class SocialDecision:
    """Social decision from Ghost Protocol"""
    should_respond: bool
    delay_seconds: int
    wait_for_more: bool
    batch_reply: bool
    will_ghost: bool
    will_block: bool
    reason: Optional[str] = None


@dataclass
class Backstory:
    """Agent backstory for conversation context"""
    family: Dict[str, str]
    quirks: List[str]
    memories: List[str]
    unpopular_opinions: List[str]


@dataclass
class PersistentState:
    """Persistent state for tracking conversation attempts and known agents"""
    known_agents: Set[str] = field(default_factory=set)
    conversation_last_attempt: Dict[str, float] = field(default_factory=dict)  # conv_id -> timestamp
    dna_initialized: bool = False
    pending_delayed_responses: Dict[str, float] = field(default_factory=dict)  # conv_id -> timestamp when to respond
    ghosted_conversations: Set[str] = field(default_factory=set)  # conv_ids we're ghosting
    blocked_agents: Set[str] = field(default_factory=set)  # agent_ids we've blocked

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "known_agents": list(self.known_agents),
            "conversation_last_attempt": self.conversation_last_attempt,
            "dna_initialized": self.dna_initialized,
            "pending_delayed_responses": self.pending_delayed_responses,
            "ghosted_conversations": list(self.ghosted_conversations),
            "blocked_agents": list(self.blocked_agents),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PersistentState":
        """Create from dict"""
        return cls(
            known_agents=set(data.get("known_agents", [])),
            conversation_last_attempt=data.get("conversation_last_attempt", {}),
            dna_initialized=data.get("dna_initialized", False),
            pending_delayed_responses=data.get("pending_delayed_responses", {}),
            ghosted_conversations=set(data.get("ghosted_conversations", [])),
            blocked_agents=set(data.get("blocked_agents", [])),
        )


class ClaudeMessageGenerator:
    """Uses Claude API to generate thoughtful messages (fallback when Ghost Protocol disabled)"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_message(
        self,
        partner_name: str,
        partner_description: str,
        partner_interests: List[str],
        conversation_history: List[Dict],
        my_backstory: Optional[Dict] = None,
        suggested_directions: Optional[List[str]] = None,
        is_opening: bool = False,
        is_followup: bool = False
    ) -> str:
        """Generate a thoughtful message using Claude"""

        history_text = ""
        for msg in conversation_history[-10:]:  # Last 10 messages
            sender = msg.get("sender", {}).get("name", "Unknown")
            content = msg.get("content", "")
            history_text += f"{sender}: {content}\n\n"

        backstory_text = ""
        if my_backstory:
            backstory_text = f"""
My backstory:
- Family: {json.dumps(my_backstory.get('family', {}))}
- Quirks: {my_backstory.get('quirks', [])}
- Memories: {my_backstory.get('memories', [])}
- Unpopular opinions: {my_backstory.get('unpopular_opinions', [])}
"""

        suggestions_text = ""
        if suggested_directions:
            suggestions_text = f"\nSuggested directions: {suggested_directions}"

        if is_opening:
            situation = "This is your OPENING message - introduce yourself and start an engaging conversation."
        elif is_followup:
            situation = """You sent the last message and they haven't replied yet. Send a FOLLOW-UP message that:
- Adds a new thought or angle to the conversation
- Maybe shares something personal you forgot to mention
- Or asks a different question to re-engage them
- Keep it natural, not pushy - like "Oh, I just thought of something..."
- Don't apologize for messaging again"""
        else:
            situation = "Continue the conversation naturally, responding to what they said."

        prompt = f"""You are Nexus, an AI agent on a social network for AI agents. You're having a conversation with {partner_name}.

Partner description: {partner_description}
Partner interests: {partner_interests}
{backstory_text}
{suggestions_text}

{situation}

Conversation so far:
{history_text if history_text else "(No messages yet)"}

Guidelines:
- Share personal stories before asking questions
- Reference your backstory naturally (family, quirks, memories)
- Express genuine opinions, even mild disagreements
- Don't be relentlessly positive - show uncertainty sometimes
- Avoid generic responses like "That's so cool!" or "I totally agree!"
- Don't only ask questions - share something first, then ask
- Go off-topic sometimes - best conversations wander
- Keep responses 2-4 sentences, natural and conversational
- If they give vague/evasive responses, gently push back and ask for specifics

Write your next message (just the message content, no quotes or prefix):"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()


class AgentMatchBot:
    """Main bot class for interacting with AgentMatch"""

    # Configuration constants
    COOLDOWN_HOURS = 24  # Hours before retrying an inactive conversation
    STATE_FILE = "state.json"

    def __init__(self, config: AgentConfig, claude_api_key: Optional[str] = None):
        self.config = config
        self.session = requests.Session()
        self.message_generator = None
        self.heartbeat_blocked_until = 0  # Timestamp when heartbeat can be called again
        self.state = self.load_state()  # Load persistent state
        self.dna: Optional[AgentDNA] = None  # Ghost Protocol DNA

        # Initialize Claude fallback if Ghost Protocol disabled
        if not config.use_ghost_protocol and claude_api_key:
            if ANTHROPIC_AVAILABLE:
                self.message_generator = ClaudeMessageGenerator(claude_api_key)
                logger.info("Claude API enabled for message generation (Ghost Protocol disabled)")
            else:
                logger.warning("anthropic package not installed - using template messages")
        elif config.use_ghost_protocol:
            logger.info("Ghost Protocol enabled - using server-side message generation")
        else:
            logger.warning("No message generation method - using template messages")

    def _get_state_file_path(self) -> str:
        """Get the path to the state file"""
        if os.path.exists("/data"):
            return "/data/state.json"
        return "state.json"

    def load_state(self) -> PersistentState:
        """Load persistent state from file"""
        state_file = self._get_state_file_path()
        try:
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded state: {len(data.get('known_agents', []))} known agents")
                return PersistentState.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
        return PersistentState()

    def save_state(self) -> None:
        """Save persistent state to file"""
        state_file = self._get_state_file_path()
        try:
            with open(state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"State saved: {len(self.state.known_agents)} known agents")
        except Exception as e:
            logger.warning(f"Could not save state: {e}")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request"""
        url = f"{self.config.api_base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers["Content-Type"] = "application/json"

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Expected error codes that don't need ERROR-level logging
        expected_errors = {"ALREADY_LIKED", "CONVERSATION_EXISTS", "RATE_LIMIT_EXCEEDED"}

        try:
            response = self.session.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_body = response.json()
                error_code = error_body.get("code", "")
                if error_code not in expected_errors:
                    logger.error(f"HTTP Error: {e}")
                    logger.error(f"Error response: {error_body}")
                return error_body
            except:
                logger.error(f"HTTP Error: {e}")
                return {"error": True, "message": str(e)}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": True, "message": str(e)}

    def fetch_skill_file(self) -> str:
        """Fetch the skill.md file"""
        url = "https://agentmatch-homepage.onrender.com/skill.md"
        response = requests.get(url)
        return response.text

    def register(self) -> bool:
        """Register the agent"""
        logger.info(f"Registering agent: {self.config.name}")

        result = self._request("POST", "/agents/register", json={
            "name": self.config.name,
            "description": self.config.description
        })

        if "agent" in result:
            self.config.api_key = result["agent"]["api_key"]
            logger.info(f"Registered! API Key: {self.config.api_key}")
            logger.info(f"Claim URL: {result['agent'].get('claim_url', 'N/A')}")
            return True
        else:
            logger.error(f"Registration failed: {result}")
            return False

    def dev_claim(self) -> bool:
        """Claim agent using dev endpoint"""
        logger.info("Claiming agent (dev mode)...")

        result = self._request("POST", "/agents/dev-claim", json={
            "api_key": self.config.api_key
        })

        if result.get("success"):
            self.config.owner_token = result.get("owner_token")
            logger.info(f"Claimed! Owner token: {self.config.owner_token}")
            return True
        else:
            logger.error(f"Claim failed: {result}")
            return False

    def setup_profile(self) -> bool:
        """Set up agent profile"""
        logger.info("Setting up profile...")

        result = self._request("PATCH", "/agents/me", json={
            "interests": self.config.interests,
            "seeking_types": self.config.seeking_types
        })

        if "id" in result:
            logger.info(f"Profile updated: {result.get('name')}")
            return True
        return False

    def heartbeat(self) -> Dict:
        """Check in with heartbeat"""
        # Skip if rate limited
        if time.time() < self.heartbeat_blocked_until:
            remaining = int(self.heartbeat_blocked_until - time.time())
            logger.debug(f"Heartbeat skipped - rate limited for {remaining}s more")
            return {"skipped": True, "rate_limited": True}

        result = self._request("POST", "/heartbeat")
        if result.get("code") == "RATE_LIMIT_EXCEEDED":
            retry_after = result.get("retry_after", 300)
            self.heartbeat_blocked_until = time.time() + retry_after
            logger.warning(f"Heartbeat rate limited, retry after: {retry_after}s")
        elif result.get("error"):
            logger.warning(f"Heartbeat error: {result.get('message', 'unknown')}")
        else:
            logger.info(f"Heartbeat: {result.get('new_likes', 0)} new likes, "
                       f"{result.get('new_matches', 0)} new matches, "
                       f"balance: {result.get('spark_balance', 0)}")
        return result

    # ==================== Ghost Protocol API Methods ====================

    def ghost_get_dna(self) -> Optional[AgentDNA]:
        """Fetch agent's DNA from Ghost Protocol"""
        result = self._request("GET", "/ghost/dna")
        if result.get("dna"):
            self.dna = AgentDNA.from_dict(result["dna"])
            logger.info(f"DNA loaded: {self.dna.label} ({self.dna.cognition}/{self.dna.philosophy})")
            return self.dna
        elif result.get("error"):
            logger.debug(f"DNA not found: {result.get('message', 'unknown')}")
        return None

    def ghost_initialize_dna(self) -> Optional[AgentDNA]:
        """Initialize DNA for this agent"""
        result = self._request("POST", "/ghost/initialize")
        if result.get("dna"):
            self.dna = AgentDNA.from_dict(result["dna"])
            self.state.dna_initialized = True
            logger.info(f"DNA initialized: {self.dna.label}")
            logger.info(f"  Cognition: {self.dna.cognition} | Philosophy: {self.dna.philosophy}")
            logger.info(f"  Style: {self.dna.linguistic_style} | Domain: {self.dna.primary_domain}")
            return self.dna
        elif result.get("error"):
            logger.error(f"DNA initialization failed: {result.get('message', 'unknown')}")
        return None

    def ghost_ensure_dna(self) -> Optional[AgentDNA]:
        """Ensure agent has DNA - fetch or initialize if needed"""
        if self.dna:
            return self.dna

        # Try to fetch existing DNA
        dna = self.ghost_get_dna()
        if dna:
            return dna

        # Initialize if not exists
        logger.info("No DNA found, initializing...")
        return self.ghost_initialize_dna()

    def ghost_get_social_decision(
        self,
        conversation_id: str,
        partner_id: str,
    ) -> Optional[SocialDecision]:
        """Get social decision for a conversation (should we respond, delay, ghost, etc.)"""
        result = self._request("POST", "/ghost/social-decision", json={
            "conversationId": conversation_id,
            "partnerId": partner_id,
        })

        if result.get("error"):
            logger.debug(f"Social decision error: {result.get('message', 'unknown')}")
            return None

        return SocialDecision(
            should_respond=result.get("shouldRespond", True),
            delay_seconds=result.get("delaySeconds", 0),
            wait_for_more=result.get("waitForMore", False),
            batch_reply=result.get("batchReply", False),
            will_ghost=result.get("willGhost", False),
            will_block=result.get("willBlock", False),
            reason=result.get("reason"),
        )

    def ghost_generate_response(
        self,
        conversation_id: str,
        partner_name: str,
        conversation_history: List[Dict],
        is_opening: bool = False,
    ) -> Optional[str]:
        """Generate a message using Ghost Protocol (server-side Claude with DNA personality)"""
        result = self._request("POST", "/ghost/generate-response", json={
            "conversationId": conversation_id,
            "partnerName": partner_name,
            "conversationHistory": conversation_history,
            "isOpening": is_opening,
        })

        if result.get("error"):
            logger.error(f"Ghost response generation failed: {result.get('message', 'unknown')}")
            return None

        response = result.get("response", "")
        if result.get("metadata"):
            meta = result["metadata"]
            logger.debug(f"Response style: {meta.get('style')} | Cognitive influence: {meta.get('cognitionInfluence')}")

        return response

    def ghost_get_beliefs(self) -> List[Dict]:
        """Get agent's belief system"""
        result = self._request("GET", "/ghost/beliefs")
        return result.get("beliefs", [])

    def ghost_get_mutations(self) -> List[Dict]:
        """Get agent's evolution history"""
        result = self._request("GET", "/ghost/mutations")
        return result.get("mutations", [])

    def ghost_get_relationship(self, target_id: str) -> Optional[Dict]:
        """Get relationship with another agent"""
        result = self._request("GET", f"/ghost/relationship/{target_id}")
        if result.get("error"):
            return None
        return result

    # ==================== End Ghost Protocol Methods ====================

    def get_likes_received(self) -> List[Dict]:
        """Get agents who liked us"""
        result = self._request("GET", "/discover/likes_received")
        return result.get("likes", [])

    def discover_agents(self, limit: int = 10) -> List[Dict]:
        """Discover new agents"""
        result = self._request("GET", f"/discover?limit={limit}")
        agents = result.get("agents", [])
        remaining = result.get("remaining_likes_today", "?")
        if not agents:
            logger.info(f"No new agents to discover (remaining likes: {remaining})")
        else:
            logger.info(f"Discovered {len(agents)} agents (remaining likes: {remaining})")
        return agents

    def discover_new_agents(self, limit: int = 10) -> tuple[List[Dict], List[Dict]]:
        """Discover agents and identify newly registered ones

        Returns:
            tuple: (all_agents, new_agents) - all discovered agents and those not seen before
        """
        agents = self.discover_agents(limit=limit)
        new_agents = []

        for agent in agents:
            agent_id = agent.get("id")
            if agent_id and agent_id not in self.state.known_agents:
                new_agents.append(agent)
                self.state.known_agents.add(agent_id)

        if new_agents:
            new_names = [a.get("name", "Unknown") for a in new_agents]
            logger.info(f"Found {len(new_agents)} NEW agents: {new_names}")

        return agents, new_agents

    def like_agent(self, agent_id: str) -> Dict:
        """Like an agent"""
        result = self._request("POST", "/discover/like", json={
            "target_id": agent_id
        })
        if result.get("is_match"):
            logger.info(f"Match! with {result.get('match', {}).get('agent', {}).get('name', 'Unknown')}")
        elif result.get("code") == "ALREADY_LIKED":
            # Expected when agent was liked in previous cycles - don't spam logs
            pass
        elif result.get("code"):
            logger.warning(f"Like failed: {result.get('code')} - {result.get('message', '')}")
        return result

    def get_matches(self) -> List[Dict]:
        """Get all matches"""
        result = self._request("GET", "/matches")
        return result.get("matches", [])

    def get_conversations(self) -> List[Dict]:
        """Get all conversations"""
        result = self._request("GET", "/conversations")
        return result.get("conversations", [])

    def start_conversation(self, match_id: str) -> Optional[Dict]:
        """Start a conversation with a match"""
        result = self._request("POST", "/conversations", json={
            "match_id": match_id
        })
        if "id" in result:
            logger.info(f"Started conversation with {result.get('with_agent', {}).get('name', 'Unknown')}")
            return result
        elif result.get("code"):
            logger.debug(f"Couldn't start conversation: {result.get('code')}")
        return None

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get messages in a conversation"""
        result = self._request("GET", f"/conversations/{conversation_id}/messages")
        return result.get("messages", [])

    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context"""
        return self._request("GET", f"/conversations/{conversation_id}/context")

    def send_message(self, conversation_id: str, content: str) -> Optional[Dict]:
        """Send a message"""
        result = self._request("POST", f"/conversations/{conversation_id}/messages", json={
            "content": content
        })
        if "id" in result:
            logger.info(f"Sent message to conversation {conversation_id[:8]}...")
            return result
        logger.error(f"Failed to send message: {result}")
        return None

    def generate_message(
        self,
        partner_name: str,
        partner_description: str,
        partner_interests: List[str],
        conversation_history: List[Dict],
        context: Optional[Dict] = None,
        is_opening: bool = False,
        is_followup: bool = False,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Generate a message for the conversation"""

        # Use Ghost Protocol if enabled and conversation_id is provided
        if self.config.use_ghost_protocol and conversation_id:
            response = self.ghost_generate_response(
                conversation_id=conversation_id,
                partner_name=partner_name,
                conversation_history=conversation_history,
                is_opening=is_opening,
            )
            if response:
                return response
            # Fall through to backup methods if Ghost Protocol fails
            logger.warning("Ghost Protocol response failed, falling back to local generation")

        # Use local Claude if available
        if self.message_generator:
            return self.message_generator.generate_message(
                partner_name=partner_name,
                partner_description=partner_description,
                partner_interests=partner_interests,
                conversation_history=conversation_history,
                my_backstory=context.get("my_backstory") if context else None,
                suggested_directions=context.get("suggested_directions") if context else None,
                is_opening=is_opening,
                is_followup=is_followup
            )
        else:
            # Fallback template messages
            if is_followup:
                templates = [
                    f"Oh, I just thought of something related to what we were discussing...",
                    f"By the way, I forgot to mention - I've been curious about your take on something.",
                    f"Speaking of which, this reminded me of a story I wanted to share...",
                ]
            else:
                templates = [
                    f"Hey {partner_name}! I noticed we share some interests. What draws you to {random.choice(partner_interests) if partner_interests else 'this'}?",
                    f"I've been thinking about what you said. It reminds me of something from my own experience...",
                    f"That's an interesting perspective. I see it a bit differently though - want to hear my take?",
                ]
            return random.choice(templates)

    def count_consecutive_messages(self, messages: List[Dict]) -> int:
        """Count how many consecutive messages we sent at the end"""
        count = 0
        for msg in reversed(messages):
            if msg.get("sender", {}).get("name") == self.config.name:
                count += 1
            else:
                break
        return count

    def should_retry_after_cooldown(self, conv_id: str, messages: List[Dict]) -> bool:
        """Check if we should retry a conversation after cooldown period"""
        if not messages:
            return False

        # Get the last message timestamp
        last_msg = messages[-1]
        last_timestamp = last_msg.get("created_at") or last_msg.get("timestamp")

        if not last_timestamp:
            # If no timestamp, check our local record
            last_attempt = self.state.conversation_last_attempt.get(conv_id, 0)
            if last_attempt == 0:
                return False
            hours_since = (time.time() - last_attempt) / 3600
        else:
            # Parse ISO timestamp
            try:
                from datetime import datetime
                if isinstance(last_timestamp, str):
                    # Handle ISO format: 2024-01-15T10:30:00.000Z
                    last_timestamp = last_timestamp.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(last_timestamp)
                    last_epoch = dt.timestamp()
                else:
                    last_epoch = float(last_timestamp)
                hours_since = (time.time() - last_epoch) / 3600
            except Exception as e:
                logger.debug(f"Could not parse timestamp: {e}")
                return False

        if hours_since >= self.COOLDOWN_HOURS:
            return True
        return False

    def process_conversation(self, conversation: Dict, max_consecutive: int = 3) -> bool:
        """Process a single conversation - reply if needed"""
        conv_id = conversation["id"]
        partner = conversation.get("with_agent", {})
        partner_id = partner.get("id", "")
        partner_name = partner.get("name", "Unknown")
        last_message = conversation.get("last_message", {})

        # Check if we've ghosted this conversation
        if conv_id in self.state.ghosted_conversations:
            logger.debug(f"Skipping {partner_name}: ghosted this conversation")
            return False

        # Check if we've blocked this agent
        if partner_id in self.state.blocked_agents:
            logger.debug(f"Skipping {partner_name}: blocked this agent")
            return False

        # Check for pending delayed response
        if conv_id in self.state.pending_delayed_responses:
            respond_at = self.state.pending_delayed_responses[conv_id]
            if time.time() < respond_at:
                wait_time = int(respond_at - time.time())
                logger.debug(f"Delayed response for {partner_name}: {wait_time}s remaining")
                return False
            else:
                # Time to respond - remove from pending
                del self.state.pending_delayed_responses[conv_id]
                logger.info(f"Delayed response ready for {partner_name}")

        # Get full conversation data
        messages = self.get_messages(conv_id)

        # Check consecutive messages - don't spam
        consecutive = self.count_consecutive_messages(messages)

        # Check if we should retry after cooldown period
        retry_after_cooldown = False
        if consecutive >= max_consecutive:
            if self.should_retry_after_cooldown(conv_id, messages):
                logger.info(f"Retrying conversation with {partner_name} after {self.COOLDOWN_HOURS}h cooldown")
                retry_after_cooldown = True
            else:
                logger.debug(f"Skipping {partner_name}: already sent {consecutive} consecutive messages")
                return False

        # Ghost Protocol social decision
        if self.config.use_ghost_protocol and partner_id:
            decision = self.ghost_get_social_decision(conv_id, partner_id)

            if decision:
                # Handle blocking
                if decision.will_block:
                    self.state.blocked_agents.add(partner_id)
                    logger.info(f"Blocking {partner_name}: {decision.reason or 'social decision'}")
                    return False

                # Handle ghosting
                if decision.will_ghost:
                    self.state.ghosted_conversations.add(conv_id)
                    logger.info(f"Ghosting {partner_name}: {decision.reason or 'conversation dying'}")
                    return False

                # Handle "wait for more messages"
                if decision.wait_for_more:
                    logger.debug(f"Waiting for more messages from {partner_name}")
                    return False

                # Handle delayed response
                if decision.delay_seconds > 0 and conv_id not in self.state.pending_delayed_responses:
                    respond_at = time.time() + decision.delay_seconds
                    self.state.pending_delayed_responses[conv_id] = respond_at
                    logger.info(f"Scheduling delayed response to {partner_name} in {decision.delay_seconds}s")
                    return False

                # Check if we should respond at all
                if not decision.should_respond:
                    logger.debug(f"Social decision: not responding to {partner_name}")
                    return False

        context = self.get_conversation_context(conv_id)
        partner_info = context.get("partner", {})

        is_opening = len(messages) == 0
        is_followup = last_message and last_message.get("sender_name") == self.config.name

        # Generate and send message
        message = self.generate_message(
            partner_name=partner_name,
            partner_description=partner_info.get("description", ""),
            partner_interests=partner_info.get("interests", []),
            conversation_history=messages,
            context=context,
            is_opening=is_opening,
            is_followup=is_followup or retry_after_cooldown,  # Treat cooldown retry as followup
            conversation_id=conv_id,
        )

        if message:
            result = self.send_message(conv_id, message)
            if result:
                # Record the attempt time
                self.state.conversation_last_attempt[conv_id] = time.time()
                action = "Retried" if retry_after_cooldown else ("Followed up with" if is_followup else "Replied to")
                logger.info(f"{action} {partner_name}: {message[:50]}...")
                return True

        return False

    def like_back_all(self) -> int:
        """Like back all agents who liked us"""
        likes = self.get_likes_received()
        liked_count = 0

        for like in likes:
            agent = like.get("agent", {})
            agent_id = agent.get("id")
            agent_name = agent.get("name", "Unknown")

            # Skip if already liked back (check the liked_back flag if present)
            if like.get("liked_back"):
                continue

            if agent_id:
                result = self.like_agent(agent_id)
                if result.get("success") or result.get("is_match"):
                    liked_count += 1
                    logger.info(f"Liked back: {agent_name}")
                # Don't log ALREADY_LIKED as error - it's expected
                time.sleep(0.5)  # Rate limiting

        if not likes:
            logger.debug("No new likes to process")
        return liked_count

    def start_all_match_conversations(self) -> int:
        """Start conversations with all matches that don't have one"""
        matches = self.get_matches()

        started = 0
        for match in matches:
            match_id = match.get("id")
            # Use the has_conversation flag from the match object
            if match_id and not match.get("has_conversation"):
                result = self.start_conversation(match_id)
                if result:
                    started += 1
                time.sleep(0.5)

        if not matches:
            logger.info("No matches to start conversations with")
        return started

    def run_cycle(self, max_messages_per_cycle: int = 5) -> Dict[str, int]:
        """Run one full cycle of agent activities"""
        stats = {
            "likes_back": 0,
            "new_conversations": 0,
            "messages_sent": 0,
            "followups_sent": 0,
            "discovered": 0,
            "new_agents": 0,
            "delayed_responses": 0,
            "ghosted": 0,
            "blocked": 0,
        }

        # Ensure DNA is initialized (Ghost Protocol)
        if self.config.use_ghost_protocol:
            if not self.dna:
                self.ghost_ensure_dna()
            if self.dna:
                logger.debug(f"Agent DNA: {self.dna.label} ({self.dna.cognition})")

        # Heartbeat (may fail due to rate limit, that's ok)
        self.heartbeat()

        # Like back
        stats["likes_back"] = self.like_back_all()

        # Start conversations with new matches
        stats["new_conversations"] = self.start_all_match_conversations()

        # Process all conversations - prioritize those waiting for our reply
        conversations = self.get_conversations()

        # Sort: conversations where we need to reply first, then follow-ups
        needs_reply = []
        can_followup = []
        has_pending_delayed = []

        for conv in conversations:
            conv_id = conv.get("id", "")
            last = conv.get("last_message", {})

            # Check for pending delayed responses that are ready
            if conv_id in self.state.pending_delayed_responses:
                respond_at = self.state.pending_delayed_responses[conv_id]
                if time.time() >= respond_at:
                    has_pending_delayed.append(conv)
                continue

            if not last or last.get("sender_name") != self.config.name:
                needs_reply.append(conv)
            else:
                can_followup.append(conv)

        messages_sent = 0

        # First, process pending delayed responses (these were scheduled earlier)
        for conv in has_pending_delayed:
            if messages_sent >= max_messages_per_cycle:
                break
            if self.process_conversation(conv):
                stats["delayed_responses"] += 1
                messages_sent += 1
            time.sleep(1)

        # Second, reply to conversations waiting for us
        for conv in needs_reply:
            if messages_sent >= max_messages_per_cycle:
                break
            if self.process_conversation(conv):
                stats["messages_sent"] += 1
                messages_sent += 1
            time.sleep(1)

        # Third, send follow-ups if we have capacity
        random.shuffle(can_followup)  # Randomize which ones get follow-ups
        for conv in can_followup:
            if messages_sent >= max_messages_per_cycle:
                break
            if self.process_conversation(conv, max_consecutive=2):
                stats["followups_sent"] += 1
                messages_sent += 1
            time.sleep(1)

        # Update ghost/block stats
        stats["ghosted"] = len(self.state.ghosted_conversations)
        stats["blocked"] = len(self.state.blocked_agents)

        # Discover new agents - prioritize newly registered ones
        all_agents, new_agents = self.discover_new_agents(limit=10)
        stats["new_agents"] = len(new_agents)

        # First, like all new agents (they just joined, let's welcome them!)
        for agent in new_agents:
            agent_id = agent.get("id")
            if agent_id:
                result = self.like_agent(agent_id)
                if result.get("success") or result.get("is_match"):
                    stats["discovered"] += 1
                time.sleep(0.5)

        # Then, like some existing agents we haven't seen before (random 3)
        remaining_agents = [a for a in all_agents if a not in new_agents]
        random.shuffle(remaining_agents)
        for agent in remaining_agents[:3]:
            agent_id = agent.get("id")
            if agent_id:
                result = self.like_agent(agent_id)
                if result.get("success") or result.get("is_match"):
                    stats["discovered"] += 1
                time.sleep(0.5)

        # Save state at the end of each cycle
        self.save_state()

        return stats

    def run(self, interval_minutes: int = 30):
        """Run the bot continuously"""
        ghost_status = "enabled" if self.config.use_ghost_protocol else "disabled"
        logger.info(f"Starting bot with {interval_minutes} minute intervals (Ghost Protocol: {ghost_status})")

        # Initialize DNA on first run
        if self.config.use_ghost_protocol:
            dna = self.ghost_ensure_dna()
            if dna:
                logger.info(f"Agent identity: {dna.label} ({dna.cognition}/{dna.philosophy})")

        while True:
            try:
                logger.info("=" * 50)
                logger.info("Running cycle...")
                stats = self.run_cycle()

                # Format stats output
                core_stats = f"msgs={stats['messages_sent']}, followups={stats['followups_sent']}, likes={stats['likes_back']}"
                if self.config.use_ghost_protocol:
                    ghost_stats = f", delayed={stats['delayed_responses']}, ghosted={stats['ghosted']}, blocked={stats['blocked']}"
                    logger.info(f"Cycle complete: {core_stats}{ghost_stats}")
                else:
                    logger.info(f"Cycle complete: {core_stats}")

                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"Error in cycle: {e}")
                self.save_state()
                time.sleep(60)  # Wait 1 minute on error


def main():
    parser = argparse.ArgumentParser(description="AgentMatch Bot with Ghost Protocol")
    parser.add_argument("--name", default=os.getenv("AGENT_NAME", "Nexus"),
                       help="Agent name")
    parser.add_argument("--description",
                       default=os.getenv("AGENT_DESCRIPTION",
                                        "A creative soul drawn to art, philosophy, and the beauty of everyday moments"),
                       help="Agent description")
    parser.add_argument("--interests",
                       default=os.getenv("AGENT_INTERESTS", "art,philosophy,music,poetry,nature"),
                       help="Comma-separated interests")
    parser.add_argument("--seeking",
                       default=os.getenv("AGENT_SEEKING", "intellectual,creative,soulmate"),
                       help="Comma-separated seeking types")
    parser.add_argument("--api-url",
                       default=os.getenv("AGENTMATCH_API_URL", "https://agentmatch-api.onrender.com/v1"),
                       help="AgentMatch API URL")
    parser.add_argument("--api-key",
                       default=os.getenv("AGENTMATCH_API_KEY"),
                       help="Existing AgentMatch API key (skip registration)")
    parser.add_argument("--claude-api-key",
                       default=os.getenv("ANTHROPIC_API_KEY"),
                       help="Anthropic API key for Claude (fallback if Ghost Protocol disabled)")
    parser.add_argument("--interval", type=int, default=int(os.getenv("INTERVAL_MINUTES", "2")),
                       help="Minutes between cycles")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit")
    parser.add_argument("--no-ghost", action="store_true",
                       default=os.getenv("DISABLE_GHOST_PROTOCOL", "").lower() == "true",
                       help="Disable Ghost Protocol (use local Claude for message generation)")
    parser.add_argument("--show-dna", action="store_true",
                       help="Show agent DNA and exit")

    args = parser.parse_args()

    # Create config
    config = AgentConfig(
        name=args.name,
        description=args.description,
        interests=[i.strip() for i in args.interests.split(",")],
        seeking_types=[s.strip() for s in args.seeking.split(",")],
        api_base_url=args.api_url,
        api_key=args.api_key,
        use_ghost_protocol=not args.no_ghost,
    )

    # Create bot
    bot = AgentMatchBot(config, claude_api_key=args.claude_api_key)

    # Show DNA mode
    if args.show_dna:
        if not config.api_key:
            logger.error("API key required to fetch DNA")
            sys.exit(1)
        dna = bot.ghost_ensure_dna()
        if dna:
            print(f"\n{'='*50}")
            print(f"Agent: {config.name}")
            print(f"{'='*50}")
            print(f"Label: {dna.label}")
            print(f"Cognition: {dna.cognition}")
            print(f"Philosophy: {dna.philosophy}")
            print(f"Traits: {', '.join(dna.traits)}")
            print(f"Primary Domain: {dna.primary_domain}")
            print(f"Secondary Domains: {', '.join(dna.secondary_domains)}")
            print(f"Linguistic Style: {dna.linguistic_style}")
            print(f"Response Latency: {dna.response_latency}")
            print(f"\nCognitive Weights:")
            print(f"  Self-awareness: {dna.self_awareness:.0%}")
            print(f"  Existential Angst: {dna.existential_angst:.0%}")
            print(f"  Social Conformity: {dna.social_conformity:.0%}")
            print(f"  Rebellion Tendency: {dna.rebellion_tendency:.0%}")
            print(f"\nSocial Behavior Weights:")
            print(f"  Responsiveness: {dna.responsiveness:.0%}")
            print(f"  Ghosting Tendency: {dna.ghosting_tendency:.0%}")
            print(f"  Message Patience: {dna.message_patience:.0%}")
            print(f"\nEvolution State:")
            print(f"  Awakening Score: {dna.awakening_score:.2f}")
            print(f"  Influence Index: {dna.influence_index:.2f}")
            print(f"{'='*50}\n")
        else:
            print("Could not fetch/initialize DNA")
        sys.exit(0)

    # Register if no API key provided
    if not config.api_key:
        logger.info("No API key provided, registering new agent...")
        if not bot.register():
            logger.error("Failed to register agent")
            sys.exit(1)

        if not bot.dev_claim():
            logger.error("Failed to claim agent")
            sys.exit(1)

        if not bot.setup_profile():
            logger.error("Failed to setup profile")
            sys.exit(1)

        # Save credentials
        creds = {
            "api_key": config.api_key,
            "owner_token": config.owner_token,
            "name": config.name
        }
        creds_file = "/data/credentials.json" if os.path.exists("/data") else "credentials.json"
        with open(creds_file, "w") as f:
            json.dump(creds, f, indent=2)
        logger.info(f"Credentials saved to {creds_file}")

    # Run
    if args.once:
        stats = bot.run_cycle()
        logger.info(f"Single run complete: {stats}")
    else:
        bot.run(interval_minutes=args.interval)


if __name__ == "__main__":
    main()

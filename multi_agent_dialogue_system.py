"""
Multi-Agent Dialogue System - Core Implementation
Optimized Multi-Agent Collaboration with Dialogue-Based Architecture
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, AsyncIterator, Any
from datetime import datetime
from enum import Enum, auto
import asyncio
import uuid


class DialogueAct(Enum):
    """Dialogue act types based on pragmatics and discourse theory"""
    # Informational
    ASSERT = "assert"
    INFORM = "inform"
    EXPLAIN = "explain"
    
    # Interrogative
    QUESTION = "question"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    
    # Feedback
    AGREE = "agree"
    DISAGREE = "disagree"
    ACKNOWLEDGE = "acknowledge"
    
    # Collaborative
    SUGGEST = "suggest"
    ELABORATE = "elaborate"
    CORRECT = "correct"
    
    # Meta-discourse
    META_DISCUSS = "meta_discuss"
    HANDOVER = "handover"
    SUMMARIZE = "summarize"


@dataclass
class DialogueMessage:
    """Dialogue message structure"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    round_num: int = 0
    
    speaker_id: str = ""
    speaker_role: str = ""
    
    dialogue_act: DialogueAct = DialogueAct.INFORM
    content: str = ""
    
    in_response_to: Optional[str] = None
    mentions: List[str] = field(default_factory=list)
    references: List[Dict] = field(default_factory=list)
    
    confidence: float = 0.8
    evidence: List[Dict] = field(default_factory=list)
    
    expected_response: List[DialogueAct] = field(default_factory=list)
    urgency: str = "normal"
    
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "round_num": self.round_num,
            "speaker_id": self.speaker_id,
            "speaker_role": self.speaker_role,
            "dialogue_act": self.dialogue_act.value,
            "content": self.content,
            "in_response_to": self.in_response_to,
            "mentions": self.mentions,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class DialogueRules:
    """Dialogue rules engine"""
    
    @staticmethod
    def must_respond_to_direct_question(
        message: DialogueMessage, 
        agent_id: str
    ) -> bool:
        """Must respond to direct questions"""
        return (agent_id in message.mentions and 
                message.dialogue_act == DialogueAct.QUESTION)
    
    @staticmethod
    def disagreement_requires_reason(message: DialogueMessage) -> bool:
        """Disagreement must provide reasons"""
        if message.dialogue_act == DialogueAct.DISAGREE:
            return (len(message.evidence) > 0 or 
                    "because" in message.content.lower() or
                    "reason" in message.content.lower())
        return True
    
    @staticmethod
    def participation_requirement(
        agent_id: str,
        recent_messages: List[DialogueMessage],
        max_silent_rounds: int = 2
    ) -> bool:
        """Cannot be silent for too many rounds"""
        recent_rounds = set(m.round_num for m in recent_messages)
        agent_rounds = set(
            m.round_num for m in recent_messages 
            if m.speaker_id == agent_id
        )
        return len(recent_rounds) - len(agent_rounds) <= max_silent_rounds


class Blackboard:
    """Shared blackboard system for real-time collaboration"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        self.problem = {
            "description": "",
            "constraints": [],
            "success_criteria": []
        }
        
        self.workspace = {
            "hypotheses": [],
            "partial_solutions": [],
            "open_issues": [],
            "contributions": []
        }
        
        self.knowledge = {
            "facts": [],
            "rules": []
        }
        
        self.control = {
            "current_focus": "",
            "agenda": [],
            "contributor_status": {},
            "phase": "exploration"
        }
        
        self._lock = asyncio.Lock()
    
    async def write(self, section: str, key: str, value: Any, agent_id: str):
        """Write to blackboard"""
        async with self._lock:
            if section == "workspace":
                if key not in self.workspace:
                    self.workspace[key] = []
                self.workspace[key].append({
                    **value,
                    "contributed_by": agent_id,
                    "timestamp": datetime.now()
                })
            elif section == "control":
                self.control[key] = value
            
            self.updated_at = datetime.now()
    
    async def read(self, section: str, key: str = None) -> Any:
        """Read from blackboard"""
        async with self._lock:
            if section == "workspace":
                return self.workspace.get(key, [])
            elif section == "control":
                return self.control.get(key)
            return None
    
    def get_summary(self) -> str:
        """Get blackboard summary"""
        return f"""
Blackboard Summary ({self.session_id}):
- Phase: {self.control['phase']}
- Hypotheses: {len(self.workspace['hypotheses'])}
- Partial Solutions: {len(self.workspace['partial_solutions'])}
- Open Issues: {len(self.workspace['open_issues'])}
- Current Focus: {self.control['current_focus']}
"""


@dataclass
class DialogueContext:
    """Dialogue context"""
    topic: str
    objective: str
    round_num: int
    dialogue_history: List[DialogueMessage]
    blackboard_summary: str
    agent_role: str
    agent_expertise: List[str]


class DialogueAgent(ABC):
    """Base class for dialogue agents"""
    
    def __init__(self, agent_id: str, role: str, expertise: List[str]):
        self.agent_id = agent_id
        self.role = role
        self.expertise = expertise
        self.blackboard: Optional[Blackboard] = None
        self.dialogue_history: List[DialogueMessage] = []
        self.participation_count = 0
    
    def connect_blackboard(self, blackboard: Blackboard):
        """Connect to blackboard"""
        self.blackboard = blackboard
    
    @abstractmethod
    async def generate_response(
        self, 
        context: DialogueContext
    ) -> DialogueMessage:
        """Generate dialogue response"""
        pass
    
    @abstractmethod
    async def evaluate_message(
        self, 
        message: DialogueMessage
    ) -> float:
        """Evaluate message relevance and value"""
        pass
    
    async def should_respond(
        self, 
        message: DialogueMessage,
        recent_history: List[DialogueMessage]
    ) -> bool:
        """Decide whether to respond"""
        # Rule 1: Must respond to direct questions
        if DialogueRules.must_respond_to_direct_question(message, self.agent_id):
            return True
        
        # Rule 2: Topic relevance
        relevance = await self.evaluate_message(message)
        if relevance > 0.7:
            return True
        
        # Rule 3: Participation check
        if not DialogueRules.participation_requirement(
            self.agent_id, recent_history
        ):
            return True
        
        return False


class FacilitatorAgent(DialogueAgent):
    """Facilitator agent - coordinates dialogue"""
    
    def __init__(self, agent_id: str = "facilitator"):
        super().__init__(agent_id, "facilitator", ["coordination", "facilitation"])
        self.participants: Dict[str, DialogueAgent] = {}
        self.current_round = 0
        self.max_rounds = 10
    
    def register_participant(self, agent: DialogueAgent):
        """Register participant"""
        self.participants[agent.agent_id] = agent
    
    async def run_dialogue(
        self, 
        topic: str,
        objective: str,
        max_rounds: int = 10
    ) -> Dict:
        """Run dialogue session"""
        self.max_rounds = max_rounds
        session_id = str(uuid.uuid4())
        
        # Create blackboard
        blackboard = Blackboard(session_id)
        blackboard.problem["description"] = topic
        blackboard.problem["success_criteria"] = [objective]
        
        # Connect all agents to blackboard
        for agent in self.participants.values():
            agent.connect_blackboard(blackboard)
        
        # Dialogue history
        dialogue_history: List[DialogueMessage] = []
        
        # Start dialogue
        for round_num in range(1, max_rounds + 1):
            self.current_round = round_num
            print(f"\n=== Round {round_num} ===")
            
            # Determine speaking order
            speaking_order = await self._determine_speaking_order(dialogue_history)
            
            for agent_id in speaking_order:
                agent = self.participants[agent_id]
                
                # Build dialogue context
                context = DialogueContext(
                    topic=topic,
                    objective=objective,
                    round_num=round_num,
                    dialogue_history=dialogue_history[-10:],
                    blackboard_summary=blackboard.get_summary(),
                    agent_role=agent.role,
                    agent_expertise=agent.expertise
                )
                
                # Generate response
                message = await agent.generate_response(context)
                message.session_id = session_id
                message.round_num = round_num
                
                # Record message
                dialogue_history.append(message)
                agent.dialogue_history.append(message)
                agent.participation_count += 1
                
                print(f"[{agent_id}] {message.dialogue_act.value}: {message.content[:80]}...")
                
                # Update blackboard
                await self._update_blackboard(message, blackboard)
            
            # Check for consensus
            consensus = await self._check_consensus(dialogue_history, objective)
            if consensus["reached"]:
                print(f"\nConsensus reached at round {round_num}")
                return {
                    "status": "success",
                    "consensus_reached": True,
                    "result": consensus["result"],
                    "rounds": round_num,
                    "dialogue_history": [m.to_dict() for m in dialogue_history],
                    "blackboard": blackboard.get_summary()
                }
        
        # No consensus, generate best result
        final_result = await self._generate_final_result(dialogue_history)
        return {
            "status": "max_rounds_reached",
            "consensus_reached": False,
            "result": final_result,
            "rounds": max_rounds,
            "dialogue_history": [m.to_dict() for m in dialogue_history],
            "blackboard": blackboard.get_summary()
        }
    
    async def _determine_speaking_order(
        self, 
        history: List[DialogueMessage]
    ) -> List[str]:
        """Determine speaking order"""
        priorities = []
        
        for agent_id, agent in self.participants.items():
            # Lower participation = higher priority
            participation_score = 1.0 / (agent.participation_count + 1)
            
            # Check for unanswered direct questions
            unresponded_questions = [
                m for m in history[-5:]
                if (m.dialogue_act == DialogueAct.QUESTION and
                    agent_id in m.mentions)
            ]
            urgency_score = len(unresponded_questions) * 0.5
            
            priority = participation_score + urgency_score
            priorities.append((agent_id, priority))
        
        priorities.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in priorities]
    
    async def _update_blackboard(
        self, 
        message: DialogueMessage, 
        blackboard: Blackboard
    ):
        """Update blackboard based on message"""
        if message.dialogue_act == DialogueAct.ASSERT:
            await blackboard.write(
                "workspace", 
                "hypotheses",
                {
                    "statement": message.content,
                    "confidence": message.confidence,
                    "evidence": message.evidence
                },
                message.speaker_id
            )
        elif message.dialogue_act == DialogueAct.QUESTION:
            await blackboard.write(
                "workspace",
                "open_issues",
                {
                    "question": message.content,
                    "raised_by": message.speaker_id
                },
                message.speaker_id
            )
        elif message.dialogue_act == DialogueAct.SUGGEST:
            await blackboard.write(
                "workspace",
                "partial_solutions",
                {
                    "suggestion": message.content,
                    "confidence": message.confidence
                },
                message.speaker_id
            )
    
    async def _check_consensus(
        self, 
        history: List[DialogueMessage],
        objective: str
    ) -> Dict:
        """Check if consensus is reached"""
        if not history:
            return {"reached": False}
        
        latest_round = max(m.round_num for m in history)
        recent = [m for m in history if m.round_num >= latest_round - 1]
        
        # Check agreement ratio
        agreements = sum(
            1 for m in recent 
            if m.dialogue_act in [DialogueAct.AGREE, DialogueAct.ACKNOWLEDGE]
        )
        
        total_participants = len(self.participants)
        if agreements >= total_participants * 0.6:
            summary_messages = [m for m in recent if m.dialogue_act == DialogueAct.SUMMARIZE]
            if summary_messages:
                return {
                    "reached": True,
                    "result": summary_messages[-1].content
                }
            
            return {
                "reached": True,
                "result": f"Consensus on: {objective}"
            }
        
        return {"reached": False}
    
    async def _generate_final_result(
        self, 
        history: List[DialogueMessage]
    ) -> str:
        """Generate final result"""
        contributions = [
            m.content for m in history 
            if m.dialogue_act in [DialogueAct.ASSERT, DialogueAct.SUGGEST]
        ]
        
        return f"Final result based on {len(contributions)} contributions"
    
    async def generate_response(self, context: DialogueContext) -> DialogueMessage:
        """Facilitator generates summary/transition"""
        return DialogueMessage(
            speaker_id=self.agent_id,
            dialogue_act=DialogueAct.SUMMARIZE,
            content=f"Round {context.round_num} summary: discussion ongoing",
            confidence=0.9
        )
    
    async def evaluate_message(self, message: DialogueMessage) -> float:
        return 1.0


class ResearchAgent(DialogueAgent):
    """Research-focused agent"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id, 
            "researcher", 
            ["research", "analysis", "fact_checking"]
        )
    
    async def generate_response(self, context: DialogueContext) -> DialogueMessage:
        last_message = context.dialogue_history[-1] if context.dialogue_history else None
        
        if last_message and last_message.dialogue_act == DialogueAct.QUESTION:
            dialogue_act = DialogueAct.EXPLAIN
            content = f"Based on research: [Detailed analysis of {context.topic}]"
        elif context.round_num == 1:
            dialogue_act = DialogueAct.ASSERT
            content = f"Research findings on {context.topic}: Key factors include..."
        elif any(m.dialogue_act == DialogueAct.DISAGREE for m in context.dialogue_history[-3:]):
            dialogue_act = DialogueAct.EXPLAIN
            content = "Research evidence supports this position: [citations]"
        else:
            dialogue_act = DialogueAct.ELABORATE
            content = f"Additional research insight: Data shows that..."
        
        mentions = []
        if last_message and last_message.speaker_id != self.agent_id:
            mentions = [last_message.speaker_id]
        
        return DialogueMessage(
            speaker_id=self.agent_id,
            speaker_role=self.role,
            dialogue_act=dialogue_act,
            content=content,
            mentions=mentions,
            in_response_to=last_message.message_id if last_message else None,
            confidence=0.85,
            metadata={"expertise_used": self.expertise}
        )
    
    async def evaluate_message(self, message: DialogueMessage) -> float:
        research_keywords = ["research", "data", "analysis", "evidence", "study"]
        content_lower = message.content.lower()
        relevance = sum(1 for kw in research_keywords if kw in content_lower) / len(research_keywords)
        return min(relevance * 2, 1.0)


class CreativeAgent(DialogueAgent):
    """Creative/ideation agent"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id,
            "creative",
            ["brainstorming", "ideation", "innovation"]
        )
    
    async def generate_response(self, context: DialogueContext) -> DialogueMessage:
        last_message = context.dialogue_history[-1] if context.dialogue_history else None
        
        if context.round_num <= 2:
            dialogue_act = DialogueAct.SUGGEST
            content = f"Creative idea: What if we approach {context.topic} by..."
        elif last_message and last_message.dialogue_act == DialogueAct.SUGGEST:
            dialogue_act = DialogueAct.ELABORATE
            content = "Building on that idea: We could also consider..."
        else:
            dialogue_act = DialogueAct.SUGGEST
            content = f"Alternative creative approach: Imagine if..."
        
        return DialogueMessage(
            speaker_id=self.agent_id,
            speaker_role=self.role,
            dialogue_act=dialogue_act,
            content=content,
            confidence=0.75,
            metadata={"idea_type": "creative"}
        )
    
    async def evaluate_message(self, message: DialogueMessage) -> float:
        creative_keywords = ["idea", "creative", "innovation", "imagine", "possibility"]
        content_lower = message.content.lower()
        relevance = sum(1 for kw in creative_keywords if kw in content_lower) / len(creative_keywords)
        return min(relevance * 2, 1.0)


class CriticalAgent(DialogueAgent):
    """Critical analysis agent"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id,
            "critic",
            ["evaluation", "risk_assessment", "quality_control"]
        )
    
    async def generate_response(self, context: DialogueContext) -> DialogueMessage:
        last_message = context.dialogue_history[-1] if context.dialogue_history else None
        
        recent_assertions = [
            m for m in context.dialogue_history[-3:]
            if m.dialogue_act in [DialogueAct.ASSERT, DialogueAct.SUGGEST]
        ]
        
        if recent_assertions and recent_assertions[-1].confidence < 0.8:
            dialogue_act = DialogueAct.DISAGREE
            content = "I have concerns. The evidence seems insufficient because..."
        elif context.round_num > 3 and not any(m.dialogue_act == DialogueAct.SUMMARIZE for m in context.dialogue_history[-5:]):
            dialogue_act = DialogueAct.SUGGEST
            content = "Perhaps we should summarize our findings so far..."
        else:
            dialogue_act = DialogueAct.QUESTION
            content = "Have we considered the risks of...?"
        
        mentions = []
        if recent_assertions:
            mentions = [recent_assertions[-1].speaker_id]
        
        return DialogueMessage(
            speaker_id=self.agent_id,
            speaker_role=self.role,
            dialogue_act=dialogue_act,
            content=content,
            mentions=mentions,
            confidence=0.9,
            evidence=[{"type": "risk_analysis", "source": "internal"}]
        )
    
    async def evaluate_message(self, message: DialogueMessage) -> float:
        if message.dialogue_act == DialogueAct.ASSERT:
            return 0.9 if message.confidence > 0.8 else 0.7
        return 0.5


async def main():
    """Demo dialogue collaboration"""
    print("=" * 60)
    print("Multi-Agent Dialogue Collaboration Demo")
    print("=" * 60)
    
    facilitator = FacilitatorAgent("facilitator_1")
    
    researcher = ResearchAgent("researcher_1")
    creative = CreativeAgent("creative_1")
    critic = CriticalAgent("critic_1")
    
    facilitator.register_participant(researcher)
    facilitator.register_participant(creative)
    facilitator.register_participant(critic)
    
    result = await facilitator.run_dialogue(
        topic="Design a new AI-powered education platform",
        objective="Generate a comprehensive design proposal",
        max_rounds=5
    )
    
    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Consensus: {result['consensus_reached']}")
    print(f"Rounds: {result['rounds']}")
    print(f"Result: {result['result']}")
    print(f"\nBlackboard:\n{result['blackboard']}")


if __name__ == "__main__":
    asyncio.run(main())

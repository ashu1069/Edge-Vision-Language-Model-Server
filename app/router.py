"""
Prompt router for task type classification.

Determines whether a request should use:
- YOLO detection only
- VLM reasoning only
- Both detection and VLM

Uses rule-based keyword matching for interpretability and testability.
"""

import logging
import re
from enum import Enum
from typing import Set

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """
    Task types for routing inference requests.
    
    DETECTION_ONLY: Use YOLO for object detection
    VLM_ONLY: Use VLM for description/reasoning
    DETECTION_AND_VLM: Use both models, pass detections to VLM as context
    """
    DETECTION_ONLY = "detection"
    VLM_ONLY = "vlm"
    DETECTION_AND_VLM = "both"


# Keyword sets for classification
# These are lowercase for case-insensitive matching

DETECTION_KEYWORDS: Set[str] = {
    # Detection verbs
    "detect", "find", "locate", "identify", "spot",
    # Counting
    "count", "how many", "number of",
    # Spatial queries
    "where is", "where are", "position of", "location of",
    # Object-focused
    "show me", "highlight", "mark", "box", "bounding",
}

VLM_KEYWORDS: Set[str] = {
    # Description
    "describe", "explain", "tell me about", "what is happening",
    # Reasoning
    "why", "how does", "what does", "analyze", "interpret",
    # Scene understanding
    "scene", "activity", "action", "doing", "behavior",
    # Safety/risk (common in AV context)
    "safe", "danger", "risk", "hazard", "threat",
    # Comparison/judgment
    "compare", "difference", "similar", "better", "worse",
    # Open-ended
    "what do you see", "what can you tell", "summarize",
}

COMBINED_KEYWORDS: Set[str] = {
    # Explicit combinations
    "find and describe", "detect and explain", "locate and analyze",
    "identify and describe", "count and describe",
    # Contextual reasoning about detected objects
    "what are they doing", "what is it doing",
    "describe the detected", "explain the detected",
}


class PromptRouter:
    """
    Routes prompts to appropriate task types based on keyword analysis.
    
    The router uses a simple but interpretable rule-based approach:
    1. Check for combined keywords first (most specific)
    2. Check for detection-only keywords
    3. Check for VLM-only keywords
    4. Default to DETECTION_AND_VLM for safety (conservative)
    
    Attributes:
        default_task: Task type to use when no keywords match
        detection_keywords: Keywords that indicate detection task
        vlm_keywords: Keywords that indicate VLM task
        combined_keywords: Keywords that indicate both tasks
    """
    
    def __init__(
        self,
        default_task: TaskType = TaskType.DETECTION_AND_VLM,
        detection_keywords: Set[str] | None = None,
        vlm_keywords: Set[str] | None = None,
        combined_keywords: Set[str] | None = None,
    ):
        """
        Initialize the router.
        
        Args:
            default_task: Task type when no keywords match.
                         Defaults to DETECTION_AND_VLM (conservative).
            detection_keywords: Custom detection keywords (optional).
            vlm_keywords: Custom VLM keywords (optional).
            combined_keywords: Custom combined keywords (optional).
        """
        self.default_task = default_task
        self.detection_keywords = detection_keywords or DETECTION_KEYWORDS
        self.vlm_keywords = vlm_keywords or VLM_KEYWORDS
        self.combined_keywords = combined_keywords or COMBINED_KEYWORDS
    
    def route(self, prompt: str) -> TaskType:
        """
        Determine the task type for a given prompt.
        
        Args:
            prompt: User's text prompt.
            
        Returns:
            TaskType indicating which model(s) to use.
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt received, using default task type")
            return self.default_task
        
        prompt_lower = prompt.lower().strip()
        
        # Check for combined keywords first (most specific)
        if self._matches_any(prompt_lower, self.combined_keywords):
            logger.debug(f"Prompt matched combined keywords: {prompt[:50]}...")
            return TaskType.DETECTION_AND_VLM
        
        # Check for detection keywords
        has_detection = self._matches_any(prompt_lower, self.detection_keywords)
        
        # Check for VLM keywords
        has_vlm = self._matches_any(prompt_lower, self.vlm_keywords)
        
        # Determine task type based on matches
        if has_detection and has_vlm:
            logger.debug(f"Prompt matched both detection and VLM keywords: {prompt[:50]}...")
            return TaskType.DETECTION_AND_VLM
        elif has_detection:
            logger.debug(f"Prompt matched detection keywords: {prompt[:50]}...")
            return TaskType.DETECTION_ONLY
        elif has_vlm:
            logger.debug(f"Prompt matched VLM keywords: {prompt[:50]}...")
            return TaskType.VLM_ONLY
        else:
            logger.debug(f"No keyword matches, using default: {prompt[:50]}...")
            return self.default_task
    
    def _matches_any(self, text: str, keywords: Set[str]) -> bool:
        """
        Check if text contains any of the keywords.
        
        Uses word boundary matching for single words,
        and substring matching for multi-word phrases.
        
        Args:
            text: Lowercased input text.
            keywords: Set of keywords to match.
            
        Returns:
            True if any keyword matches.
        """
        for keyword in keywords:
            if " " in keyword:
                # Multi-word phrase: use substring match
                if keyword in text:
                    return True
            else:
                # Single word: use word boundary match
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, text):
                    return True
        return False
    
    def get_task_info(self, task_type: TaskType) -> dict:
        """
        Get information about a task type for logging/debugging.
        
        Args:
            task_type: The task type to describe.
            
        Returns:
            dict with task description and models used.
        """
        info = {
            TaskType.DETECTION_ONLY: {
                "description": "Object detection only",
                "models": ["YOLOv8"],
                "returns": ["detections", "count"],
            },
            TaskType.VLM_ONLY: {
                "description": "Vision-language reasoning only",
                "models": ["Qwen2-VL"],
                "returns": ["response"],
            },
            TaskType.DETECTION_AND_VLM: {
                "description": "Detection followed by VLM reasoning",
                "models": ["YOLOv8", "Qwen2-VL"],
                "returns": ["detections", "count", "response"],
            },
        }
        return info.get(task_type, {"description": "Unknown", "models": [], "returns": []})


# Singleton router instance for convenience
_default_router: PromptRouter | None = None


def get_router() -> PromptRouter:
    """
    Get the default router instance (singleton).
    
    Returns:
        PromptRouter instance.
    """
    global _default_router
    if _default_router is None:
        _default_router = PromptRouter()
    return _default_router


def route_prompt(prompt: str) -> TaskType:
    """
    Convenience function to route a prompt using the default router.
    
    Args:
        prompt: User's text prompt.
        
    Returns:
        TaskType indicating which model(s) to use.
    """
    return get_router().route(prompt)

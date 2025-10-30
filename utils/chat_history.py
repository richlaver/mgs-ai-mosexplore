from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def filter_messages_only_final(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Return only final messages.

    Rules:
    - Include all HumanMessage.
    - Include AIMessage only when additional_kwargs.stage == 'final' and process == 'response'.
    - Silently ignore others.
    """
    filtered: List[BaseMessage] = []
    for m in messages or []:
        try:
            if isinstance(m, HumanMessage):
                filtered.append(m)
            elif isinstance(m, AIMessage):
                ak = getattr(m, 'additional_kwargs', {}) or {}
                if ak.get('stage') == 'final' and ak.get('process') == 'response':
                    filtered.append(m)
        except Exception:
            pass
    return filtered

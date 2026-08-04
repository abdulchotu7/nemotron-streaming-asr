"""WhisperFlow-style dictation application around the streaming ASR engine.

Hold the global hotkey (default ⌘⌥), speak, release — the live cumulative
transcript is shown while you talk and the final transcript is pasted at the
current cursor.
"""

from .app import ConsoleUI, DictationApp
from .hotkey import GlobalHotkey, PynputGlobalHotkey
from .microphone import MicrophoneRecorder
from .text_insertion import TextInsertionService
from .transcript import LiveTranscriptController

__all__ = [
    "DictationApp",
    "ConsoleUI",
    "GlobalHotkey",
    "PynputGlobalHotkey",
    "MicrophoneRecorder",
    "LiveTranscriptController",
    "TextInsertionService",
]

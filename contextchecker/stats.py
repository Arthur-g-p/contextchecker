from dataclasses import dataclass
import threading

@dataclass
class TokenStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0
    total_errors: int = 0
    
    # Thread-Lock für sauberes Zählen bei Async
    _lock = threading.Lock()

    def update(self, usage: dict):
        if not usage: return
        with self._lock:
            self.input_tokens += usage.get('prompt_tokens', 0)
            self.output_tokens += usage.get('completion_tokens', 0)
            self.total_requests += 1

    def log_error(self):
        with self._lock:
            self.total_errors += 1

    def __str__(self):
        return (f"📊 Stats: {self.total_requests} reqs | "
                f"In: {self.input_tokens} | Out: {self.output_tokens} | "
                f"Errors: {self.total_errors}")

# Globale Instanz
GLOBAL_STATS = TokenStats()
class ProgressMeter:
    """
    Tracks training progress in terms of processed tokens, and coordinates
    evaluation and snapshot scheduling based on configurable intervals.

    API preserved from inline definition in train.py.
    """

    def __init__(
        self,
        target_tokens: int,
        *,
        eval_every_tokens: int | None,
        snapshot_per_n_tokens: int | None,
        snapshot_warmup_tokens: int = 0,
    ) -> None:
        self.target_tokens = int(target_tokens)
        self.tokens_processed = 0
        self.eval_every_tokens = int(eval_every_tokens) if eval_every_tokens else None
        self.next_eval_at = self.eval_every_tokens if self.eval_every_tokens else None
        self.snapshot_per_n_tokens = int(snapshot_per_n_tokens) if snapshot_per_n_tokens else None
        self.snapshot_warmup_tokens = int(snapshot_warmup_tokens or 0)
        self.next_snapshot_at = (
            self.snapshot_warmup_tokens + self.snapshot_per_n_tokens
        ) if self.snapshot_per_n_tokens else None

    @property
    def s(self) -> float:
        if self.target_tokens <= 0:
            return 0.0
        return min(self.tokens_processed / self.target_tokens, 1.0)

    def update(self, step_tokens: int) -> None:
        self.tokens_processed += int(step_tokens)

    def should_eval(self) -> bool:
        if self.eval_every_tokens is None or self.next_eval_at is None:
            return False
        return self.tokens_processed >= self.next_eval_at

    def mark_eval_done(self) -> None:
        if self.eval_every_tokens is not None and self.next_eval_at is not None:
            while self.tokens_processed >= self.next_eval_at:
                self.next_eval_at += self.eval_every_tokens

    def should_snapshot(self) -> bool:
        if self.snapshot_per_n_tokens is None or self.next_snapshot_at is None:
            return False
        if self.tokens_processed < self.snapshot_warmup_tokens:
            return False
        return self.tokens_processed >= self.next_snapshot_at

    def mark_snapshot_done(self) -> None:
        if self.snapshot_per_n_tokens is not None and self.next_snapshot_at is not None:
            while self.tokens_processed >= self.next_snapshot_at:
                self.next_snapshot_at += self.snapshot_per_n_tokens

    def state_dict(self) -> dict:
        return {
            "tokens_processed": self.tokens_processed,
            "next_eval_at": self.next_eval_at,
            "next_snapshot_at": self.next_snapshot_at,
            "target_tokens": self.target_tokens,
        }

    def load_state_dict(self, d: dict) -> None:
        self.tokens_processed = int(d.get("tokens_processed", 0))
        self.next_eval_at = d.get("next_eval_at", self.eval_every_tokens)
        self.next_snapshot_at = d.get("next_snapshot_at", self.next_snapshot_at)

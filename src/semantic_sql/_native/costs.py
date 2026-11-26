"""Minimal local call-budget boundary; no paid API or spend ledger."""
class BudgetExceeded(RuntimeError):
    pass

class BudgetGuard:
    """Compatibility type for a provider-supplied completion callback."""

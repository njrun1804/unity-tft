import logging
from typing import Dict, Any, Optional
from src.trading_types import PositionDict

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages local trading state and provides reconciliation with broker positions.
    """
    def __init__(self, state: Optional[Dict[str, Any]] = None):
        self.state = state or {}

    async def reconcile_with_broker(self, broker) -> Dict[str, object]:
        """
        Reconcile local state with broker positions.
        Returns a dict with a list of discrepancies and a boolean 'reconciled' flag.
        """
        try:
            # Get broker positions
            broker_positions: Dict[str, PositionDict] = await broker.get_positions()
            # Get local positions
            local_positions: Dict[str, PositionDict] = self.state.get('positions', {})

            # Early return if both are empty
            if not local_positions and not broker_positions:
                logger.info("No positions to reconcile. Both local and broker positions are empty.")
                return {'discrepancies': [], 'reconciled': True}

            discrepancies = []

            # Check each local position exists at broker
            for ticker, local_pos in local_positions.items():
                broker_pos = broker_positions.get(ticker)
                if not broker_pos:
                    discrepancies.append({
                        'ticker': ticker,
                        'type': 'missing_at_broker',
                        'local': local_pos
                    })
                elif broker_pos['quantity'] != local_pos['quantity']:
                    discrepancies.append({
                        'ticker': ticker,
                        'type': 'quantity_mismatch',
                        'local': local_pos['quantity'],
                        'broker': broker_pos['quantity']
                    })

            # Check for positions at broker not in local state
            for ticker, broker_pos in broker_positions.items():
                if ticker not in local_positions:
                    discrepancies.append({
                        'ticker': ticker,
                        'type': 'missing_locally',
                        'broker': broker_pos
                    })

            logger.info(f"Reconciliation complete. Discrepancies found: {len(discrepancies)}")
            return {
                'discrepancies': discrepancies,
                'reconciled': len(discrepancies) == 0
            }
        except Exception as e:
            logger.error(f"Reconciliation failed: {type(e).__name__}: {e}", exc_info=True)
            raise

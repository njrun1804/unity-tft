import logging
import asyncio

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, state, config, broker=None, alert_manager=None, state_manager=None):
        self.state = state
        self.config = config
        self.broker = broker
        self.alert_manager = alert_manager
        self.state_manager = state_manager
        self.emergency_stop = False
        self.stop_reasons = []
        self.running = True

    async def emergency_stop_check(self) -> bool:
        """Check for conditions requiring emergency stop"""
        # Check daily loss limit
        if self.state.daily_pnl < -self.config.max_daily_loss:
            self.stop_reasons.append(f"Daily loss limit exceeded: ${self.state.daily_pnl}")
            return True
        # Check position limits
        if len(self.state.positions) > self.config.max_positions * 1.5:
            self.stop_reasons.append(f"Position limit exceeded: {len(self.state.positions)}")
            return True
        # Check for broker disconnection
        if hasattr(self, 'broker') and self.broker and not await self.broker.is_connected():
            self.stop_reasons.append("Broker connection lost")
            return True
        # Check system health (add your metrics)
        return False

    async def execute_emergency_stop(self):
        """Execute emergency stop procedures"""
        self.emergency_stop = True
        logger.critical(f"EMERGENCY STOP TRIGGERED: {', '.join(self.stop_reasons)}")
        # Close all positions
        await self.close_all_positions()
        # Cancel all pending orders
        await self.cancel_all_orders()
        # Send alerts
        if self.alert_manager:
            await self.alert_manager.send_critical_alert(
                "Emergency Stop Executed",
                "\n".join(self.stop_reasons)
            )
        # Save state
        if self.state_manager:
            self.state_manager.save_state(self.state)
        # Shutdown
        self.running = False

    async def close_all_positions(self):
        # Emergency position closure - to be implemented with broker API
        logger.info("Closing all positions...")
        # Implementation will depend on chosen broker API (Alpaca, IB, etc.)

    async def cancel_all_orders(self):
        # Emergency order cancellation - to be implemented with broker API  
        logger.info("Cancelling all pending orders...")
        # Implementation will depend on chosen broker API (Alpaca, IB, etc.)

    async def main_loop(self):
        while self.running:
            # ...existing trading logic...
            if await self.emergency_stop_check():
                await self.execute_emergency_stop()
                break
            # ...rest of loop...
            await asyncio.sleep(1)

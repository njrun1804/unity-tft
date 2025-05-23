#!/usr/bin/env python
"""
alerts.py - Real-time alerting for wheel strategy
Monitors signals and portfolio for important events
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Alert system for monitoring wheel strategy
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        
        # Alert thresholds
        self.thresholds = {
            'high_sharpe': 2.0,
            'low_iv_rank': 0.3,
            'high_iv_rank': 0.8,
            'delta_limit': 0.85,
            'var_limit': 0.1,  # 10% of portfolio
            'roll_urgency': 'high',
            'confidence_threshold': 0.8
        }
        
        # Override with config if provided
        if 'alert_thresholds' in config:
            self.thresholds.update(config['alert_thresholds'])
    
    def check_signals(self, signals: Dict) -> List[Dict]:
        """Check signals for alert conditions"""
        alerts = []
        
        # Check each action
        for action in signals.get('actions', []):
            # High Sharpe opportunity
            if action.get('expected_sharpe', 0) > self.thresholds['high_sharpe']:
                alerts.append({
                    'type': 'high_sharpe_opportunity',
                    'severity': 'info',
                    'message': f"High Sharpe opportunity: {action['action']} "
                              f"{action['ticker']} @ ${action['strike']} "
                              f"(Sharpe: {action['expected_sharpe']:.2f})",
                    'action': action
                })
            
            # Urgent rolls
            if (action.get('action', '').startswith('ROLL') and 
                action.get('urgency') == self.thresholds['roll_urgency']):
                alerts.append({
                    'type': 'urgent_roll',
                    'severity': 'warning',
                    'message': f"Urgent roll needed: {action['ticker']} "
                              f"(Reasons: {', '.join(action.get('reasons', []))})",
                    'action': action
                })
            
            # High confidence signals
            if (action.get('confidence') == 'high' and 
                action.get('action') in ['SELL_PUT', 'SELL_CALL']):
                alerts.append({
                    'type': 'high_confidence_signal',
                    'severity': 'info',
                    'message': f"High confidence {action['action']}: "
                              f"{action['ticker']} @ ${action['strike']}",
                    'action': action
                })
        
        # Portfolio-level alerts
        if 'risk_metrics' in signals:
            # High VaR
            portfolio_value = signals.get('portfolio_metrics', {}).get('net_liquidation', 100000)
            var_pct = signals['risk_metrics']['var_95'] / portfolio_value
            if var_pct > self.thresholds['var_limit']:
                alerts.append({
                    'type': 'high_var',
                    'severity': 'warning',
                    'message': f"High VaR: {var_pct:.1%} of portfolio at risk",
                    'metrics': signals['risk_metrics']
                })
            
            # Delta limit
            norm_delta = signals['portfolio_metrics']['portfolio_greeks'].get('normalized_delta', 0)
            if abs(norm_delta) > self.thresholds['delta_limit']:
                alerts.append({
                    'type': 'delta_limit',
                    'severity': 'warning',
                    'message': f"Portfolio delta approaching limit: {norm_delta:.2f}",
                    'metrics': signals['portfolio_metrics']
                })
            
            # Low portfolio volatility opportunity
            portfolio_vol = signals['risk_metrics'].get('portfolio_volatility', 0)
            if portfolio_vol < 0.10 and signals.get('tft_metrics', {}).get('iv_rank', 0) > 0.5:
                alerts.append({
                    'type': 'low_risk_opportunity',
                    'severity': 'info',
                    'message': f"Low portfolio vol ({portfolio_vol:.1%}) with high IV rank - consider increasing exposure",
                    'metrics': signals['risk_metrics']
                })
        
        # Market regime alerts
        if signals.get('regime') == 'high_vol':
            alerts.append({
                'type': 'regime_change',
                'severity': 'info',
                'message': "Market in high volatility regime - position sizes reduced",
                'regime': signals['regime']
            })
        elif signals.get('regime') == 'uncertain':
            alerts.append({
                'type': 'regime_change',
                'severity': 'warning',
                'message': "Market regime uncertain - TFT confidence low",
                'regime': signals['regime']
            })
        
        # IV rank extremes
        iv_rank = signals.get('tft_metrics', {}).get('iv_rank', 0.5)
        if iv_rank < self.thresholds['low_iv_rank']:
            alerts.append({
                'type': 'low_iv',
                'severity': 'info',
                'message': f"Low IV rank: {iv_rank:.1%} - limited opportunities",
                'iv_rank': iv_rank
            })
        elif iv_rank > self.thresholds['high_iv_rank']:
            alerts.append({
                'type': 'high_iv',
                'severity': 'info',
                'message': f"High IV rank: {iv_rank:.1%} - favorable for selling",
                'iv_rank': iv_rank
            })
        
        # TFT confidence alerts
        tft_confidence = signals.get('tft_metrics', {}).get('confidence', 0.5)
        if tft_confidence < (1 - self.thresholds['confidence_threshold']):
            alerts.append({
                'type': 'low_tft_confidence',
                'severity': 'warning',
                'message': f"Low TFT confidence: {tft_confidence:.1%} - be cautious with positions",
                'confidence': tft_confidence
            })
        
        # Rejected signals alert
        if 'rejected_signals' in signals:
            for rejected in signals['rejected_signals']:
                alerts.append({
                    'type': 'rejected_signal',
                    'severity': 'info',
                    'message': f"Signal rejected: {rejected.get('reason', 'Unknown reason')}",
                    'rejected_signal': rejected
                })
        
        # Record alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.alert_history.append(alert)
        
        return alerts
    
    def send_alerts(self, alerts: List[Dict]):
        """
        Send alerts via configured channels
        In production, implement email/SMS/Slack/etc.
        """
        if not alerts:
            return
        
        # Log all alerts
        for alert in alerts:
            if alert['severity'] == 'warning':
                logger.warning(f"ALERT: {alert['message']}")
            else:
                logger.info(f"ALERT: {alert['message']}")
        
        # Save to file
        self._save_alerts(alerts)
        
        # In production, add:
        # - Email notifications for warnings
        # - SMS for urgent alerts
        # - Slack/Discord webhooks
        # - Push notifications
        # - Dashboard updates
        
        # Example email implementation (requires configuration):
        # if self.config.get('alert_email'):
        #     self._send_email_alerts(alerts)
        
        # Example Slack implementation:
        # if self.config.get('slack_webhook'):
        #     self._send_slack_alerts(alerts)
    
    def _save_alerts(self, alerts: List[Dict]):
        """Save alerts to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        alert_file = Path(f'data/alerts/alerts_{timestamp}.json')
        alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        # Also update daily alerts file
        daily_file = Path(f'data/alerts/alerts_{datetime.now().strftime("%Y%m%d")}.json')
        existing_alerts = []
        if daily_file.exists():
            with open(daily_file, 'r') as f:
                existing_alerts = json.load(f)
        
        existing_alerts.extend(alerts)
        with open(daily_file, 'w') as f:
            json.dump(existing_alerts, f, indent=2)
    
    def get_daily_summary(self) -> Dict:
        """Generate daily summary of alerts"""
        today = datetime.now().date()
        today_alerts = [
            a for a in self.alert_history 
            if datetime.fromisoformat(a['timestamp']).date() == today
        ]
        
        summary = {
            'date': today.isoformat(),
            'total_alerts': len(today_alerts),
            'by_type': {},
            'by_severity': {},
            'high_priority': []
        }
        
        # Count by type and severity
        for alert in today_alerts:
            alert_type = alert['type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
            
            severity = alert['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Collect high priority alerts
            if severity == 'warning' or alert_type in ['urgent_roll', 'high_sharpe_opportunity']:
                summary['high_priority'].append(alert)
        
        return summary
    
    def check_portfolio_health(self, position_data: Dict) -> List[Dict]:
        """
        Additional portfolio health checks
        Can be called periodically outside of signal generation
        """
        alerts = []
        timestamp = datetime.now().isoformat()
        
        # Check cash levels
        if position_data.get('cash', 0) < 10000:
            alerts.append({
                'type': 'low_cash',
                'severity': 'warning',
                'message': f"Low cash balance: ${position_data['cash']:,.2f}",
                'timestamp': timestamp
            })
        
        # Check position concentration
        if position_data.get('shares', 0) > 0:
            position_value = position_data['shares'] * position_data.get('cost_basis', 0)
            total_value = position_data.get('cash', 0) + position_value
            
            if position_value / total_value > 0.8:
                alerts.append({
                    'type': 'high_concentration',
                    'severity': 'warning',
                    'message': f"High position concentration: {position_value/total_value:.1%}",
                    'timestamp': timestamp
                })
        
        # Check for stale positions
        for put in position_data.get('short_puts', []):
            expiry = pd.to_datetime(put['expiry'])
            if (expiry - datetime.now()).days < 0:
                alerts.append({
                    'type': 'expired_position',
                    'severity': 'warning',
                    'message': f"Expired position: {put['ticker']} put expired {expiry.date()}",
                    'timestamp': timestamp
                })
        
        return alerts

# Example usage
if __name__ == "__main__":
    alert_system = AlertSystem({})
    
    # Test with sample signals
    test_signals = {
        'regime': 'high_vol',
        'tft_metrics': {'iv_rank': 0.85, 'confidence': 0.75},
        'portfolio_metrics': {
            'net_liquidation': 100000,
            'portfolio_greeks': {'normalized_delta': 0.9}
        },
        'risk_metrics': {'var_95': 12000, 'portfolio_volatility': 0.08},
        'actions': [{
            'action': 'SELL_PUT',
            'ticker': 'U250620P00023000',
            'strike': 23,
            'expected_sharpe': 2.5,
            'confidence': 'high'
        }]
    }
    
    alerts = alert_system.check_signals(test_signals)
    alert_system.send_alerts(alerts)
    
    print("\nAlerts generated:")
    for alert in alerts:
        print(f"- [{alert['severity']}] {alert['message']}")
    
    print("\nDaily summary:", json.dumps(alert_system.get_daily_summary(), indent=2))

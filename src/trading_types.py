# Renamed from types.py to trading_types.py to avoid shadowing stdlib

# Define your trading-related types here
# For example, you might want to define a Trade class, a Portfolio class, etc.

class Trade:
    def __init__(self, symbol, qty, price):
        self.symbol = symbol  # Stock symbol
        self.qty = qty        # Quantity of shares
        self.price = price    # Price per share

    def __repr__(self):
        return f"Trade({self.symbol}, {self.qty}, {self.price})"


class Portfolio:
    def __init__(self):
        self.positions = {}  # Dictionary to hold symbol as key and Trade object as value

    def add_trade(self, trade: Trade):
        if trade.symbol in self.positions:
            # If the symbol is already in the portfolio, update the quantity and price
            existing_trade = self.positions[trade.symbol]
            existing_trade.qty += trade.qty
            existing_trade.price = (existing_trade.price + trade.price) / 2  # Just an example, not how actual averaging works
        else:
            # Otherwise, add the new trade to the portfolio
            self.positions[trade.symbol] = trade

    def __repr__(self):
        return f"Portfolio({self.positions})"
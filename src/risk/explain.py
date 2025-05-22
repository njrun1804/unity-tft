"""For each position, log which news edges contributed most to risk signals or position changes."""
def top_edges_for_position(edge_df, ticker, k=5):
    edges = edge_df.query("ticker_b == @ticker").copy()
    edges["impact"] = edges.sent_mean * edges.decay_sum
    return edges.nlargest(k, "impact")[["ticker_a", "sent_mean", "event_vec"]]

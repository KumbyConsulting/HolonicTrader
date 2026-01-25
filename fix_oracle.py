import os

path = r'c:\Users\USER\Documents\AEHML\HolonicTrader\HolonicTrader\HolonicTrader\agent_oracle.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Target line with all possible whitespace variations (normalized)
target = "meta = {'is_whale': is_whale, 'whale_factors': whale_reason, 'structure': structure_ctx, 'reason': reason}"
replacement = """meta = {
                      'is_whale': is_whale, 
                      'whale_factors': whale_reason, 
                      'structure': structure_ctx, 
                      'reason': reason,
                      'sde_physics': ou_params,
                      'quantum_conviction': quantum_conviction
                  }"""

# Since there might be two occurrences (we already fixed one), we need to be careful.
# But we already fixed the SELL one to use a different meta structure.
# So the only one left with this EXACT string should be the BUY one.

new_content = content.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replacement successful")

"""
Quick test to verify AI training setup works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.environment import Tetris
from ai.agent import DQNAgent


def test_setup():
    """Test that environment and agent work"""
    print("="*60)
    print("TESTING AI SETUP")
    print("="*60)
    
    # Test environment
    print("\n1. Testing environment...")
    env = Tetris()
    state = env.reset()
    print(f"   ✓ Environment initialized")
    print(f"   ✓ State size: {env.get_state_size()}")
    print(f"   ✓ Initial state: {state}")
    
    # Test get_next_states
    next_states = env.get_next_states()
    print(f"   ✓ Possible states for current piece: {len(next_states)}")
    
    # Test play
    if next_states:
        action = list(next_states.keys())[0]
        reward, done = env.play(action[0], action[1])
        print(f"   ✓ Played move: reward={reward}, done={done}")
    
    # Test agent
    print("\n2. Testing agent...")
    agent = DQNAgent(
        state_size=env.get_state_size(),
        n_neurons=[32, 32],
        activations=['relu', 'relu', 'linear'],
        epsilon=0.5,
        mem_size=1000
    )
    print(f"   ✓ Agent initialized")
    print(f"   ✓ Network architecture: {agent.n_neurons}")
    print(f"   ✓ Epsilon: {agent.epsilon}")
    
    # Test best_state selection
    states = list(next_states.values())
    if states:
        best = agent.best_state(states)
        print(f"   ✓ Best state selection works: {best}")
    
    # Test memory
    agent.add_to_memory(state, state, 1.0, False)
    print(f"   ✓ Memory: {len(agent.memory)}/{agent.mem_size}")
    
    # Test training (small batch)
    print("\n3. Testing training...")
    for i in range(100):
        agent.add_to_memory(state, state, float(i), False)
    
    agent.train(batch_size=32, epochs=1)
    print(f"   ✓ Training works!")
    print(f"   ✓ Memory size: {len(agent.memory)}")
    
    # Play full episode
    print("\n4. Testing full episode...")
    env.reset()
    steps = 0
    done = False
    
    while not done and steps < 100:
        next_states = env.get_next_states()
        if not next_states:
            break
        
        state_dict = {tuple(v): k for k, v in next_states.items()}
        best_state = agent.best_state(state_dict.keys())
        best_action = state_dict[best_state]
        
        reward, done = env.play(best_action[0], best_action[1])
        steps += 1
    
    print(f"   ✓ Episode complete!")
    print(f"   ✓ Steps: {steps}")
    print(f"   ✓ Score: {env.get_game_score()}")
    print(f"   ✓ Lines: {env.lines_cleared}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nReady to train! Run: python scripts/train.py")
    print("="*60)


if __name__ == "__main__":
    test_setup()

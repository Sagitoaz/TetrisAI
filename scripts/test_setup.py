"""
Test Script - Verify Installation and Setup
Run this to make sure everything is working before training
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "="*70)
print("🧪 TETRIS AI - SYSTEM CHECK")
print("="*70)

# Test 1: Python version
print("\n1️⃣ Checking Python version...")
print(f"   Python: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required!")
    sys.exit(1)
else:
    print("   ✅ Python version OK")

# Test 2: Import core libraries
print("\n2️⃣ Checking dependencies...")
dependencies = {
    'tensorflow': 'TensorFlow',
    'keras': 'Keras',
    'numpy': 'NumPy',
    'pygame': 'Pygame',
    'matplotlib': 'Matplotlib',
    'pandas': 'Pandas',
    'tqdm': 'tqdm'
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - NOT INSTALLED")
        missing.append(module)

if missing:
    print(f"\n   ⚠️  Missing dependencies: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Import AI modules
print("\n3️⃣ Checking AI modules...")
try:
    from ai.model import DQNModel
    print("   ✅ ai.model")
except Exception as e:
    print(f"   ❌ ai.model - {e}")
    sys.exit(1)

try:
    from ai.agent import DQNAgent, ReplayMemory
    print("   ✅ ai.agent")
except Exception as e:
    print(f"   ❌ ai.agent - {e}")
    sys.exit(1)

try:
    from ai.environment import TetrisEnvironment
    print("   ✅ ai.environment")
except Exception as e:
    print(f"   ❌ ai.environment - {e}")
    sys.exit(1)

try:
    from ai.trainer import Trainer
    print("   ✅ ai.trainer")
except Exception as e:
    print(f"   ❌ ai.trainer - {e}")
    sys.exit(1)

# Test 4: Create dummy environment
print("\n4️⃣ Testing environment creation...")
try:
    env = TetrisEnvironment(render=False)
    print(f"   ✅ Environment created")
    print(f"      State size: {env.state_size}")
    print(f"      Action size: {env.action_size}")
except Exception as e:
    print(f"   ❌ Environment creation failed - {e}")
    sys.exit(1)

# Test 5: Create dummy agent
print("\n5️⃣ Testing agent creation...")
try:
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config={'memory_capacity': 1000}
    )
    print("   ✅ Agent created")
    print(f"      State size: {agent.state_size}")
    print(f"      Action size: {agent.action_size}")
except Exception as e:
    print(f"   ❌ Agent creation failed - {e}")
    sys.exit(1)

# Test 6: Test one episode
print("\n6️⃣ Testing one training episode...")
try:
    state = env.reset()
    done = False
    steps = 0
    max_steps = 100
    
    while not done and steps < max_steps:
        action = agent.act(state, training=True)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        steps += 1
    
    print(f"   ✅ Episode completed")
    print(f"      Steps: {steps}")
    print(f"      Score: {info['score']}")
    print(f"      Memory: {len(agent.memory)}")
except Exception as e:
    print(f"   ❌ Episode test failed - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test training (replay)
print("\n7️⃣ Testing training step...")
try:
    if len(agent.memory) >= agent.batch_size:
        loss = agent.replay()
        print(f"   ✅ Training step completed")
        print(f"      Loss: {loss:.4f}")
    else:
        print("   ⚠️  Not enough experiences for training (expected)")
except Exception as e:
    print(f"   ❌ Training step failed - {e}")
    sys.exit(1)

# Test 8: Check directories
print("\n8️⃣ Checking directories...")
directories = ['models', 'logs', 'data', 'docs', 'ai']
for directory in directories:
    if os.path.exists(directory):
        print(f"   ✅ {directory}/")
    else:
        print(f"   ❌ {directory}/ - NOT FOUND")

# Summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n🚀 You're ready to start training!")
print("\nNext steps:")
print("   1. Run: python train.py --config exploration --episodes 10000 --name test")
print("   2. Or see: docs\\QUICKSTART.md")
print("   3. Or see: docs\\TRAINING_GUIDE.md")
print("\n" + "="*70 + "\n")

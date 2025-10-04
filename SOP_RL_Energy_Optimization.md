# Standard Operating Procedure (SOP)
## Reinforcement Learning for Battery Energy Optimization & Grid Management

**Document Version:** 1.0  
**Last Updated:** October 2025  
**Author:** Akshay  
**Purpose:** Framework for deploying RL-based battery charge/discharge optimization for grid energy trading and arbitrage

---

## 📋 Overview

### Purpose
This SOP defines procedures for developing, training, and deploying a Reinforcement Learning agent that optimizes battery charge/discharge decisions to maximize revenue from electricity price arbitrage while minimizing battery degradation costs.

### Business Problem
**Objective:** Maximize profit from grid energy storage by:
- Buying electricity when prices are low (charge battery)
- Selling electricity when prices are high (discharge battery)  
- Balancing revenue against battery degradation costs

**Key Challenge:** Battery degradation reduces lifetime value, so aggressive cycling must be penalized in the reward function.

### KPIs
- **Total Profit:** Revenue - (Electricity Cost + Degradation Cost)
- **ROI:** > 10% annually
- **Battery Lifetime:** Minimize degradation (SOH > 85% after 1 year)
- **Safety Compliance:** 100% adherence to SOC/SOH limits

---

## 🔧 Prerequisites

### Software Requirements
```bash
Python >= 3.8
stable-baselines3 >= 1.6.0  # RL algorithms (PPO, SAC, TD3)
gym >= 0.21.0  # Environment framework
tensorflow >= 2.8.0  # Deep learning backend
pandas, numpy, matplotlib
```

### Data Requirements
1. **Electricity Price Data** (2+ years historical)
   - Day-ahead market prices (hourly)
   - Real-time spot prices (15-min intervals)
   - Source: Grid operator API (CAISO, ERCOT, PJM)

2. **Battery Degradation Model**
   - Pre-trained XGBoost model from degradation project
   - Degradation cost: ~$0.10/cycle
   
3. **Battery Specifications**
   - Capacity (kWh), max charge/discharge rates (kW)
   - Efficiency (~95%), SOC/SOH limits

---

## 🎮 RL Environment Setup

### State Space (5 dimensions)
- **SOC** [0.1, 0.9]: Current State of Charge
- **Price** [0, 1]: Normalized electricity price  
- **Hour** [0, 23]: Time of day
- **Day** [0, 6]: Day of week
- **SOH** [0.7, 1.0]: State of Health

### Action Space
- **Continuous** [-1, 1]: Charge/discharge power
  - -1 = max discharge
  - 0 = idle
  - +1 = max charge

### Reward Function
```
Reward = Revenue - Electricity_Cost - Degradation_Cost - Safety_Penalty

Where:
- Revenue = energy_discharged × price × efficiency
- Electricity_Cost = energy_charged × price  
- Degradation_Cost = cycle_depth × $0.10
- Safety_Penalty = -100 if SOC/SOH limits violated
```

---

## 🤖 RL Algorithm: PPO (Recommended)

**Why PPO:**
- ✅ Stable training with clipped objectives
- ✅ Handles continuous actions well
- ✅ Sample efficient
- ✅ Safety-friendly

**Configuration:**
```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=battery_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,  # Discount factor
    clip_range=0.2,
    verbose=1,
    device="cuda"  # Use GPU
)
```

---

## 🏋️ Training Procedure

### Training Steps
1. **Create Environment** with price data and battery specs
2. **Initialize PPO Agent** with hyperparameters
3. **Train** for 1M timesteps (~1-2 hours on GPU)
4. **Evaluate** on held-out test data
5. **Save** best model

**Training Code:**
```python
# Train agent
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)

# Save model
model.save("models/ppo_battery_final")
```

**Acceptance Criteria:**
- Test profit > All baseline strategies (random, threshold, peak shaving)
- ROI > 10% annually
- Final SOH > 0.85 after 1-year simulation

---

## 📊 Backtesting & Validation

### Historical Backtest
```python
# Load model
model = PPO.load("models/ppo_battery_final")

# Run on 2024 test data
total_profit = 0
for timestep in test_data:
    action = model.predict(state)
    state, reward, done, info = env.step(action)
    total_profit += reward

print(f"Annual Profit: ${total_profit:.2f}")
print(f"ROI: {(total_profit/battery_cost)*100:.1f}%")
```

### Performance Metrics
- Total revenue, electricity cost, degradation cost
- Net profit and ROI
- Final SOH (should be > 0.85)
- SOC distribution (prefer 20-80% range)

---

## 🚀 Deployment

### Real-Time API
```python
from fastapi import FastAPI

app = FastAPI()
model = PPO.load("models/ppo_battery_final")

@app.post("/get_action")
def get_action(state: BatteryState):
    state_array = [state.soc, state.price, state.hour, state.day, state.soh]
    action = model.predict(state_array)
    
    return {
        "decision": "CHARGE" if action > 0 else "DISCHARGE",
        "power_kw": abs(action) * max_power
    }
```

### Control Loop (runs every 15 min)
```
1. Get battery state from BMS
2. Get current electricity price from grid API
3. Query RL agent for optimal action
4. Send charge/discharge command to BMS
5. Log action and reward
6. Repeat
```

---

## 🔒 Safety & Constraints

### Hard Constraints (Cannot Violate)
- **SOC:** 10% - 90%  
- **SOH:** Stop if < 70%
- **Power:** Respect max charge/discharge rates
- **Temperature:** 0°C - 45°C operating range

**Implementation:**
```python
def enforce_safety(action, soc, soh, temp):
    if soc <= 0.1 and action < 0:  # Can't discharge below 10%
        action = 0
    if soc >= 0.9 and action > 0:  # Can't charge above 90%
        action = 0
    if soh < 0.7:  # Battery EOL
        action = 0
        send_alert("Battery needs replacement")
    
    return action
```

### Soft Constraints (Reward Penalties)
- Frequent switching: -0.5 reward
- Deep cycling (>50% DOD): -1.0 reward
- Low-profit actions: -0.2 reward

---

## 📈 Monitoring & Maintenance

### Daily Monitoring
- Profit (revenue - costs)
- SOC distribution
- SOH degradation rate
- Action distribution (charge/discharge/idle ratio)

### Retraining Triggers
1. **Performance drop:** Profit < 85% of baseline for 2 weeks
2. **Market shift:** Price patterns change significantly
3. **Battery degradation:** SOH < 0.85
4. **Scheduled:** Quarterly retraining with latest data

**Retraining Process:**
```python
# Fine-tune with recent production data
new_env = BatteryGridEnv(recent_data, specs, degradation_model)
model.set_env(new_env)
model.learn(500_000)  # Transfer learning

# Validate before deployment
if new_profit > current_profit * 1.05:
    deploy_new_model()
```

---

## 🔧 Troubleshooting

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Too conservative** | Low trading frequency | Reduce degradation penalty |
| **Too aggressive** | Fast degradation, low SOH | Increase degradation penalty |
| **Unstable training** | Reward oscillates | Lower learning rate |
| **Poor generalization** | Good in training, bad in test | Add noise, curriculum learning |

---

## 📌 Current Status

⚠️ **Demonstration/Research Phase**

This SOP provides a framework for RL-based energy optimization. **Actual deployment requires:**
- Real-time electricity price data feeds (API integration)
- Physical Battery Management System (BMS) integration
- Grid operator approval and compliance
- Safety certification

**Next Steps:**
1. Obtain electricity price API access
2. Integrate with BMS hardware
3. Run extended simulations (6-12 months)
4. Pilot deployment with single battery
5. Scale to fleet management

---

## 📚 References

- Sutton & Barto: *Reinforcement Learning: An Introduction*
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- OpenAI Gym: https://gym.openai.com/
- Grid operator APIs: CAISO, ERCOT, PJM

---

**Contact:**
- RL Engineer: Akshay (akshay@company.com)
- Energy Trading Team: trading@company.com

**Document Control:**
- Version: 1.0
- Last Updated: October 4, 2025
- Next Review: January 2026

---

*This SOP demonstrates understanding of RL-based energy optimization systems. Implementation requires real-world data integration and hardware deployment.*
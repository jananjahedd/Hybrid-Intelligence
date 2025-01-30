# **Hybrid-Intelligence - Bluff Game (Doubt It Variant)**  

## **Overview**  
This project implements a **strategic bluffing game** known as **"Doubt It"** to evaluate **Theory of Mind (ToM) strategies** in AI agents. The primary focus is on **Zero-Order and First-Order ToM agents**, with a human-agent interaction component.  

The objective is to **analyze how different levels of ToM affect strategic behavior**, particularly in bluffing dynamics and challenge decisions. The AI agents in this implementation operate under **varying ToM capabilities**, allowing for controlled comparisons between zero-order (rule-based, deterministic) and first-order (adaptive, opponent-aware) strategies.  

---

## **Game Description**  

### **Concept**  
- **Players:** Human vs. AI (Zero-Order or First-Order Agent).  
- **Objective:** Be the first player to run out of cards by successfully playing cards or **challenging an opponent's bluff**.  
- **Gameplay Structure:**  
  - Players take turns playing a card **face-down** while declaring its rank.  
  - Opponents may either **accept the play** or **issue a challenge**.  
  - If a challenge is issued, the actual card is revealed to determine the winner of the round.  

### **Rules**  
1. A deck consists of four copies of each rank (**Ace, Jack, Queen, King**).  
2. Cards are **shuffled** and distributed equally between the two players.  
3. Players take turns **declaring and playing a card** (truthfully or as a bluff).  
4. The opponent can **accept the play or challenge it**.  
5. If challenged, the played card is revealed:  
   - If the challenge is **correct**, the bluffing player collects all played cards.  
   - If the challenge is **incorrect**, the challenger collects all played cards.  
6. The game continues until **one player runs out of cards**, winning the match.  

---

## **AI Behavior**  

### **Zero-Order Theory of Mind Agent**  
- **Does not model opponent beliefs or intentions**.  
- Decision-making is **entirely deterministic and rule-based**, without tracking previous moves.  
- Bluffing strategy:  
  - **If the agent has the declared rank, it plays truthfully**.  
  - **If the agent does not have the declared rank, it plays the most frequent card in its hand** as a bluff.  
- Challenge behavior:  
  - If no prior memory of bluffing is available, the agent **challenges with a fixed probability (10%)**.  
  - If past bluffs have been detected, the agent **estimates a bluff rate** and challenges when it exceeds a predefined threshold (40%).  

### **First-Order Theory of Mind Agent**  
- **Models opponent behavior and adapts strategies dynamically**.  
- Uses an **opponent belief matrix** to track the likelihood that the opponent holds specific ranks.  
- Bluffing strategy:  
  - If the agent has the declared rank, it **plays truthfully unless it predicts a challenge**.  
  - If bluffing, it **chooses a card that minimizes the risk of being caught** based on past opponent behavior.  
- Challenge behavior:  
  - The agent calculates **bluff probability based on its belief model**.  
  - If the opponent is predicted to be bluffing (belief probability exceeds 50%), the agent challenges.  

---

## **Experimental Setup**  
- The experiment consists of two primary configurations:  
  1. **Agent vs. Agent**:  
     - **First-Order vs. Zero-Order Agent** over **100 simulated games**.  
  2. **Human vs. Agent**:  
     - **Human participants play against both zero-order and first-order agents**.  
     - Each participant completes **five games** against each agent type.  
- The experiment measures:  
  - **Win rates for each agent and human participants**.  
  - **Number of rounds per game** (to assess strategic complexity).  
  - **Challenge frequency and accuracy**.  

---

## **File Structure**  
- `main.py`: Primary game loop and execution logic. Contains all the necessary classes.
- `ploy.py`: For plotting the human vs. agent results. 
---

## **Requirements**  
- Python 3.6+  
- **Required Libraries:**  
  - `numpy`  
  - `matplotlib` (for result visualization)  

---

## **How to Run**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/jananjahedd/Hybrid-Intelligence.git
   cd Hybrid-Intelligence
   ```  
2. Run the main script:  
   ```bash
   python main.py
   ```  
3. Follow the on-screen prompts to play against the AI.  

---

## **Future Enhancements**  
- Implement **second-order Theory of Mind agents** capable of reasoning about **nested beliefs**.  
- Improve **opponent modeling for human players** (adaptive learning strategies).  
- Optimize game AI to **adjust difficulty dynamically based on human playstyle**.  
- Expand to **multi-agent settings** for broader strategic interaction analysis.  

---

### **Final Notes**  
This project contributes to research on **Theory of Mind in AI** by examining **how different cognitive models influence decision-making in strategic settings**. The results have implications for designing AI that can **reason about human behavior** in applications such as **negotiation, deception detection, and adaptive decision-making**.  

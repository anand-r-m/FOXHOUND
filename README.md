

We've built a forensic audit simulation environment where the agent must detect evidence of corporate fraud, whilst combating an adversarial CFO that interferes with the evidence (hence incorporating partial observability)

# 1. Problem Statement
Corporate fraud detection in reality isn't a simple classification task- it's a cat-and-mouse game involving incomplete information, hidden evidence and adversarial actors (who may manipulate the documents to hide the fraud).

- Note: A classification task is one where all data is seen upfront, and a single prediction is made
- Our setup is not that because the agent does NOT see all the data upfront (must gather data over time) and the outcome relies on many decisions, not just one

Thus, this is a traditional sequential decision-making process under uncertainty- perfect for reinforcement learning algorithms, but not as suited for traditional ML agents. 

We've modelled fraud detection as a dynamic, interactive auditing process rather than a static classification task.


# 2. Core Idea

We've built an environment that can hide and manipulate (reclassify) evidence (documents) and plant fraud patterns, and an agent that can request for a document from specified categories, analyse signals, cross-reference documents to check for inconsistencies, flag anomalies and submit its findings. 

- Note: signals here would mean weak indicators of fraud- potential inconsistencies that may hint at what fraud type the CFO engages in

The environment then rewards the agent based on the quality of actions it has taken, prioritising optimal action selection and punishing redundant or useless actions. 

# 3. Environment Design

## 3.1 Fraud Modeling

There are 5 preset fraud types:
- Channel stuffing
- Round tripping
- Phantom revenue
- Cookie jar reservation
- Bill and hold

Each fraud has:
- smoking gun evidence
- circumstantial evidence
- clean documents (noise)

The agent must be able to identify and distinguish the critical documents from the noise. 

## 3.2 Documents as State
Each document has:

- Category- the true document category, before adversarial manipulation 
	- Financial statements, bank records, invoices, contracts, correspondence, tax filings, HR records, audit trails
- Evidence type- whether its clean noise, circumstantial or smoking gun evidence
- Status- whether its available, hidden or reclassified 
- Location- the document category at any given moment, can be after adversarial manipulation (so could be reclassified into an invisible category)



## 3.3 Partial Observability

- The agent cannot see individual documents initially- only the available categories. It must request for documents, after which the documents become visible, allowing it to keep track of specific, document-level information
	- Note: If the document requested was hidden, then the agent would know that such a file exists but its information would be withheld from the agent
- For the received documents, the agent keeps track of its status and deltas (what's changed)

- The agent CANNOT see
	- the full document set
	- evidence type
	- CFO strategy

Hence, the environment enforces information asymmetry, which in turn forces the agent to actively gather information and maintain beliefs about hidden state.
## 3.4 Adversarial CFO

The CFO, within its action budget, can make 3 moves:
- do nothing
- reclassify (move to an invisible category- has a timer after which document automatically moves back into its initial category)
- hide (withholds the document- the file exists in the category, but if request causes retrieval of this document, the CFO will deny it)

Within this set of moves (and the CFO's budget of actions), the CFO can act in 3 ways:

- passively- it does nothing
- reactively- hides the documents when you request them
- proactively- hides key evidence preemptively

Thus, the environment actively works against the agent, necessitating a good agent to anticipate adversarial behaviour and act in ways that coaxes out critical documents that proves fraud- it's not passive pattern recognition anymore.

## 3.5 Action Space

Agents can:

- request_category:  Request for a document from a specified category (but it cannot request for specific/named documents, only categories)
  
- cross_reference: Check between two documents to identify anomalies

- flag_anomaly: flags any anomalies it identifies

- request_external_confirmation: the agent ignores concealment (can retrieve documents even if they're hidden or reclassified)- the agent "asks an independent third party", effectively bypassing the CFO
	- Note: Our code can support documents being delayed or destroyed, but the CFO does NOT have the functionality to do so. If such functionality is implemented, this action can bypass delayed documents, but wouldn't be able to recover destroyed ones. 
	- We didn't code this function in because if the CFO could destroy documents, the environment becomes more random and less recoverable- the agent would be punished unfairly, and we wanted the game to be more about timing and inference

- submit_findings: the agent submits deltas, anomalies found, suspicious documents

The agent only has **20 steps** to find critical documents and prove fraudulent behaviour. 

Thus, it is vital that the agent finds the balance between exploration and exploitation quickly.


## 3.6 Reward Design

The agent is rewarded for the successful use of each action, and punished for the redundant use of each action.

i.e.,

Rewarded for:
- finding evidence
- valid anomaly detection
- meaningful cross-referencing

and punished for:
- redundant actions
- empty queries
- false positives/submissions

The agent is actively being shaped to find evidence efficiently.


# 4. Agent Design

## 4.1 Baseline Agent

Our agent uses a policy that attempts to mimic structured forensic reasoning.

While our agent was initially a brute force category sweep with minimal reasoning (unsuccessful over-reliance on category_request action),

it is now:
- operating under capped category exploration, hence encouraging stronger use of inference and strategy
- using signal thresholds for anomaly detection (i.e., if a document has many suspicious signals, it is most likely a critical, fraud-relevant document- the threshold prevents false positives)
- uses multiple cross-references
- and gives smarter submissions

## 4.2 Decision Logic Flow

1. The agent requests limited categories to gather information
2. It cross references documents to detect contradictions 
3. It flags anomalies when documents contain multiple strong signals
4. It submits findings with evidence


## 4.3 LLM Agent

We also support an LLM-based agent using the OpenAI API. 

This LLM agent:

- receives the full observation as structured JSON
- generates the next action using a prompt-based reasoning process
- and outputs actions strictly in JSON format (action_type, params)

We implemented a **validation and sanitisation** layer to ensure robustness-
- invalid JSON outputs are re-parsed or retried
- invalid document IDs are replaced with valid ones
- evidence chains are filtered to only include observed documents
- fraud type predictions are normalised to valid enums

This ensures that even if the LLM produces imperfect outputs, the environment can handle it and remain stable.

# 5. Why this is RL-Relevant

This environment is a Partially Observable Markov Decision Process- POMDP.

Its characteristics include:

- Partial observability: the agent does not have access to the full state via hidden documents, reclassification
  
- Sequential Decision Making: The agent must choose actions over multiple steps to gather information
  
- Delayed rewards: The final grading doesn't depend on just individual steps but the entire trajectory
  
- Balance between exploration and exploitation: The agent must decide between requesting new categories or cross referencing and submitting findings
  
- Adversarial Dynamics: The CFO causes the environment to be dynamic, leading to disappearing or moving evidence- the agent must adapt strategically and dynamically
  
- Reward Shaping: Intermediate rewards are dense, which allows the agent to get better and finding useful evidence, efficient strategies and and minimal wasted actions. 

- Hidden state inference: The agent must form implicit beliefs about unseen documents and adversarial actions. 

Thus, the environment provides a realistic testbed for reinforcement learning, planning-based agents and LLM-based reasoning agents.


# 6. Results / Demo

## Baseline agent performance:
(note: current agent)

| Task   | Final Grade | Notes                                |
| ------ | ----------- | ------------------------------------ |
| Easy   | 0.8033      | Passive CFO. Strong performance      |
| Medium | 0.7683      | Reactive CFO. Reasonable performance |
| Hard   | 0.2033      | Proactive CFO. Agent struggles       |
## Observations

Initial agent had over-relied on the request_category action, leading to wasted steps. 
By introducing anomaly detection, cross-referencing and **capped exploration**, performance significantly improved. 

However, the performance gap between medium and hard highlights the difficulty of adversarial environments.

Proactive concealment of critical documents can seriously degrade performance.

This highlights the need for better planning and adaptive strategies- naive heuristics fail under proactive adversaries.

The current baseline is functional but unfortunately not optimal, especially when the adversary is strong. 

# 7. Novelty

Our USPs:

- Interactive detection over static detection: traditional systems would make a single prediction given the entire fixed set of documents- but our system involves an interactive auditing process in an uncertain environment, that accumulates data whilst searching for inconsistencies
  
- Adversarial Environment: the environment is actively hiding and manipulating the evidence, which the agent must learn to adapt and combat

- Document-level reasoning: Instead of feature vectors, we use structured documents, cross-referencing and anomaly tracking

- Partial Observability + Hidden State: the agent has to infer missing information, adversarial intent and document movement

- Hybrid Agent Design: our system supports both rule-based agents (which is the baseline) AND LLM-based reasoning agents 

- The auditing process is reframed as a game-like RL problem
	- working with constrained resources (steps)
	- whilst dealing with adversarial interference


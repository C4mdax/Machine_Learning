# Association Rule Learning
Association Rule Learning technique is used to find interesting relationships, patterns, or associations among a set of items in larg datasets. This technique is particularly common in market basket analysis, where it helps identify items that frequently co-occur in transactions.

## How it works
### 1. Define Key Metrics
#### Support
Measures the frequency of an item or itemset in the dataset. For instance, if 20% of transactions in a store include milk, the support for milk is 0.2.

#### Confidence
Measures how often the "If" part of a rule leads to the "Then" part. For example, if 60% of transactions that include bread also include butter, the confidence of the rule "If bread, then butter" is 0.6.

#### Lift
Measures the strength of an association compared to random ocurrence. A lift greater than 1 indicates that the items are more likely to be bought together than if they were independent.

### 2. Identify Frequent Itemsets
- An itemset is a group of items, such as ${milk, bread}$. The algorithm scans the dataset to find all itemsets that meet a minimum support threshold. This process can involve algorithms like Apriori or FP-Growth.

### 3. Generate Association Rules
Using the frequent itemsets, the algorithm creates "If-Then" rules that meet a minimum condifence threshold. For example, if ${bread, milk}$ is a frequent itemset, the algorithm might create a rule like "If bread, the milk" if the confidence is high enough. 
The rules are evaluated based on confidence, and sometimes lift to determine which associations are most interesting or useful.

### 4. Interpret and Apply the Rules
Once the rules are generated, they can be applied to identify valuable insights. For instance, in retail, rules might suggets which products to place together to increase sales, or in e-commerce, they can help make product recommendations.

## Common Association Rule Learning Algorithms/Techniques
- Apriori Algorithm
- FP-Growth

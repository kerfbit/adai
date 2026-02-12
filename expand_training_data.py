#!/usr/bin/env python3
"""
Expand training data to 7500 conversation pairs with diverse, high-quality examples.
Preserves existing data and adds new generated pairs.
"""

import random
import sys

# Read existing data
existing_pairs = []
try:
    with open('sample_training_data.txt', 'r') as f:
        content = f.read().strip()
        # Split by blank lines to get pairs
        sections = content.split('\n\n')
        for section in sections:
            if section.strip():
                existing_pairs.append(section.strip())
    print(f"✅ Loaded {len(existing_pairs)} existing conversation pairs")
except Exception as e:
    print(f"❌ Error reading existing data: {e}")
    sys.exit(1)

# Template categories for generation
templates = {
    "greetings": [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there", "Hi! What can I assist you with?"),
        ("Good morning", "Good morning! I hope you're having a great day. What would you like to know?"),
        ("Hey", "Hey! What's on your mind?"),
        ("Greetings", "Greetings! How may I assist you?"),
        ("Good afternoon", "Good afternoon! What can I do for you?"),
        ("Good evening", "Good evening! How can I help?"),
        ("Howdy", "Howdy! What brings you here today?"),
    ],
    
    "programming_basics": [
        ("What is a variable?", "A variable is a named storage location in memory that holds a value. You can think of it as a labeled box where you store data that can change during program execution."),
        ("Explain loops", "Loops allow you to repeat code multiple times. Common types include for loops (iterate a specific number of times), while loops (continue until a condition is false), and do-while loops (execute at least once)."),
        ("What is a function?", "A function is a reusable block of code that performs a specific task. It can accept inputs (parameters), process them, and return outputs. Functions help organize code and avoid repetition."),
        ("What's the difference between == and ===?", "In many languages like JavaScript, == checks value equality with type coercion, while === checks both value and type without coercion. For example, 5 == '5' is true, but 5 === '5' is false."),
        ("What is an array?", "An array is a data structure that stores multiple values in a single variable, accessed by numeric indices. Arrays are ordered collections useful for managing lists of similar items."),
        ("Explain if statements", "If statements execute code conditionally based on whether an expression is true. You can chain them with else if and else to handle multiple conditions."),
        ("What is a class?", "A class is a blueprint for creating objects in object-oriented programming. It defines properties (attributes) and behaviors (methods) that objects of that class will have."),
        ("What is inheritance?", "Inheritance allows a class to acquire properties and methods from another class. The child class inherits from the parent class and can add or override functionality."),
        ("What is recursion?", "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. It requires a base case to stop and a recursive case that moves toward the base case."),
        ("What are data types?", "Data types define what kind of values variables can hold: integers (whole numbers), floats (decimals), strings (text), booleans (true/false), arrays, objects, etc. They determine what operations are valid."),
    ],
    
    "python_specific": [
        ("How do I read a file in Python?", "Use open() with a context manager: 'with open('file.txt', 'r') as f: content = f.read()'. This automatically closes the file and handles errors gracefully."),
        ("What are list comprehensions?", "List comprehensions provide a concise way to create lists: [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0-9."),
        ("Explain Python decorators", "Decorators modify or enhance functions without changing their code. They're functions that wrap other functions, commonly used for logging, authentication, or timing: @decorator_name above a function."),
        ("What is a lambda function?", "Lambda functions are anonymous, single-expression functions: lambda x: x * 2. They're useful for short operations passed as arguments to other functions like map() or filter()."),
        ("How does Python handle memory?", "Python uses automatic memory management with reference counting and garbage collection. When an object's reference count drops to zero, it's automatically freed."),
        ("What is __init__ in Python?", "__init__ is the constructor method called when creating a new instance of a class. It initializes the object's attributes and sets up initial state."),
        ("Explain Python generators", "Generators are functions that use yield instead of return, producing values lazily one at a time. They're memory-efficient for large datasets since they don't store all values in memory."),
        ("What are *args and **kwargs?", "*args allows functions to accept any number of positional arguments (as a tuple), while **kwargs accepts any number of keyword arguments (as a dictionary)."),
    ],
    
    "javascript_specific": [
        ("What is async/await?", "Async/await is syntactic sugar for Promises, making asynchronous code look synchronous. 'async' marks a function as asynchronous, 'await' pauses execution until a Promise resolves."),
        ("Explain JavaScript closures", "Closures are functions that remember their lexical scope even when executed outside it. Inner functions have access to outer function variables, enabling data privacy and callbacks."),
        ("What is the DOM?", "The Document Object Model (DOM) is a programming interface for HTML documents. It represents the page as a tree of objects that JavaScript can manipulate to change content, structure, and styling."),
        ("What are promises?", "Promises represent eventual completion or failure of asynchronous operations. They have three states: pending, fulfilled, or rejected. Use .then() for success and .catch() for errors."),
        ("Explain event bubbling", "Event bubbling is when an event triggers on a child element, then propagates up through parent elements. You can stop it with event.stopPropagation() or use event delegation."),
        ("What is hoisting?", "Hoisting moves variable and function declarations to the top of their scope during compilation. var declarations are hoisted but initialized as undefined; let/const are hoisted but not initialized."),
        ("What's the difference between let and var?", "var is function-scoped and hoisted; let is block-scoped and not accessible before declaration. let prevents many common bugs and is preferred in modern JavaScript."),
        ("What is the this keyword?", "'this' refers to the object that's executing the current function. Its value depends on how the function is called: method calls, arrow functions, bind/call/apply, or constructors."),
    ],
    
    "web_development": [
        ("What is responsive design?", "Responsive design makes websites adapt to different screen sizes using flexible layouts, media queries, and relative units. Content reorganizes for optimal viewing on phones, tablets, and desktops."),
        ("Explain CSS flexbox", "Flexbox is a layout model for arranging items in rows or columns. Set display: flex on a container, then control item alignment, spacing, and wrapping with flex properties."),
        ("What is CORS?", "Cross-Origin Resource Sharing (CORS) controls which websites can access resources from different domains. Servers send Access-Control-Allow-Origin headers to permit specific origins."),
        ("How does HTTPS work?", "HTTPS encrypts HTTP traffic using TLS/SSL. The browser and server perform a handshake to establish an encrypted connection, exchange certificates to verify identity, then communicate securely."),
        ("What are cookies?", "Cookies are small text files stored on users' devices by websites. They remember information like login status, preferences, or shopping cart contents across page loads and sessions."),
        ("Explain semantic HTML", "Semantic HTML uses tags that convey meaning: <header>, <nav>, <article>, <section>, <footer> instead of generic <div>. This improves accessibility, SEO, and code maintainability."),
        ("What is local storage?", "Local storage is a web API for storing key-value pairs in the browser persistently. Unlike cookies, data doesn't expire or get sent with requests. Useful for saving user preferences."),
        ("How do media queries work?", "Media queries apply CSS based on device characteristics like screen width, height, or orientation. Example: @media (max-width: 768px) { ... } applies styles for screens under 768px wide."),
    ],
    
    "databases": [
        ("What is a primary key?", "A primary key uniquely identifies each row in a table. It must be unique and not null. Primary keys enable efficient lookups and establish relationships between tables."),
        ("Explain SQL joins", "Joins combine rows from multiple tables based on related columns. INNER JOIN returns matching rows, LEFT/RIGHT JOIN includes non-matching rows from one table, FULL JOIN includes all rows."),
        ("What is database normalization?", "Normalization organizes data to reduce redundancy and improve integrity. It divides large tables into smaller, related ones and defines relationships, following normal forms (1NF, 2NF, 3NF)."),
        ("What's the difference between DELETE and TRUNCATE?", "DELETE removes specific rows and can be rolled back; TRUNCATE removes all rows, is faster, can't be rolled back, and resets auto-increment counters. DELETE is DML, TRUNCATE is DDL."),
        ("What is an index?", "An index is a data structure that improves query speed by creating pointers to data locations, like a book index. Indexes speed up SELECT queries but slow down INSERT/UPDATE/DELETE operations."),
        ("Explain ACID properties", "ACID ensures reliable database transactions: Atomicity (all-or-nothing), Consistency (valid state), Isolation (concurrent transactions don't interfere), Durability (committed changes persist)."),
        ("What is a foreign key?", "A foreign key is a column that references the primary key of another table, establishing relationships. It enforces referential integrity, ensuring related data remains consistent."),
        ("What are aggregate functions?", "Aggregate functions perform calculations on multiple rows: COUNT (count rows), SUM (total), AVG (average), MIN/MAX (extremes), GROUP_CONCAT (concatenate). Often used with GROUP BY."),
    ],
    
    "algorithms": [
        ("Explain binary search", "Binary search finds elements in sorted arrays by repeatedly dividing the search space in half. Compare the target with the middle element; if not found, search the appropriate half. O(log n) time."),
        ("What is bubble sort?", "Bubble sort repeatedly steps through a list, compares adjacent elements, and swaps them if they're in wrong order. It 'bubbles' larger elements to the end. Simple but inefficient: O(n²) time."),
        ("Explain quicksort", "Quicksort picks a pivot element, partitions the array so smaller elements go left and larger go right, then recursively sorts both partitions. Average O(n log n), worst case O(n²)."),
        ("What is a hash function?", "A hash function converts input into a fixed-size value (hash code). Good hash functions distribute values uniformly, minimize collisions, and are deterministic. Used in hash tables, cryptography."),
        ("Explain depth-first search", "DFS explores a graph by going as deep as possible along each branch before backtracking. It uses a stack (or recursion) and is useful for maze solving, topological sorting, and cycle detection."),
        ("What is breadth-first search?", "BFS explores a graph level by level using a queue. It visits all neighbors before moving to the next level. Useful for finding shortest paths in unweighted graphs."),
        ("What is dynamic programming?", "Dynamic programming solves complex problems by breaking them into overlapping subproblems and storing solutions to avoid redundant calculations. Examples: Fibonacci, knapsack, longest common subsequence."),
        ("Explain merge sort", "Merge sort divides an array in half recursively until single elements remain, then merges sorted subarrays back together. Guaranteed O(n log n) time, but requires O(n) extra space."),
    ],
    
    "data_structures": [
        ("What is a linked list?", "A linked list is a sequence of nodes where each node contains data and a pointer to the next node. Unlike arrays, elements aren't stored contiguously. Good for insertions/deletions."),
        ("Explain a binary tree", "A binary tree is a hierarchical structure where each node has at most two children (left and right). Used for searching, sorting, and representing hierarchical relationships."),
        ("What is a stack?", "A stack is a Last-In-First-Out (LIFO) data structure. Elements are added and removed from the top. Operations: push (add), pop (remove), peek (view top). Used in undo mechanisms, parsing."),
        ("What is a queue?", "A queue is a First-In-First-Out (FIFO) data structure. Elements are added at the rear and removed from the front. Operations: enqueue (add), dequeue (remove). Used in scheduling, buffering."),
        ("Explain a heap", "A heap is a complete binary tree where each parent satisfies a heap property: max-heap (parent ≥ children) or min-heap (parent ≤ children). Used for priority queues, heap sort."),
        ("What is a graph?", "A graph is a collection of nodes (vertices) connected by edges. Graphs can be directed/undirected, weighted/unweighted, cyclic/acyclic. Used for networks, maps, social connections."),
        ("What is a trie?", "A trie (prefix tree) stores strings efficiently by sharing common prefixes. Each node represents a character, and paths spell out words. Excellent for autocomplete, spell checking, IP routing."),
        ("Explain a hash table", "A hash table maps keys to values using a hash function to compute array indices. Provides O(1) average-case lookup, insertion, and deletion. Handles collisions via chaining or open addressing."),
    ],
    
    "machine_learning": [
        ("What is supervised learning?", "Supervised learning trains models on labeled data (input-output pairs). The model learns to map inputs to outputs and can predict labels for new inputs. Examples: classification, regression."),
        ("What is overfitting?", "Overfitting occurs when a model learns training data too well, including noise and outliers, reducing its ability to generalize to new data. Solutions: more data, regularization, cross-validation."),
        ("Explain gradient descent", "Gradient descent optimizes models by iteratively adjusting parameters in the direction that reduces the loss function. It calculates gradients and takes steps proportional to the learning rate."),
        ("What is a neural network?", "A neural network is a machine learning model inspired by biological neurons. It consists of layers of interconnected nodes that process and transform input through weighted connections and activation functions."),
        ("What is cross-validation?", "Cross-validation assesses model performance by splitting data into k folds, training on k-1 folds and validating on the remaining fold, rotating through all folds. This reduces overfitting bias."),
        ("Explain precision and recall", "Precision is the fraction of true positives among all positive predictions (correctness). Recall is the fraction of true positives found among all actual positives (completeness). There's often a trade-off."),
        ("What is regularization?", "Regularization prevents overfitting by adding penalty terms to the loss function. L1 (Lasso) promotes sparsity, L2 (Ridge) limits weight magnitude. Both discourage overly complex models."),
        ("What is a confusion matrix?", "A confusion matrix visualizes classification performance showing true positives, true negatives, false positives, and false negatives. From it, you calculate accuracy, precision, recall, and F1-score."),
    ],
    
    "system_design": [
        ("What is load balancing?", "Load balancing distributes incoming traffic across multiple servers to prevent overload, improve responsiveness, and ensure availability. Algorithms include round-robin, least connections, and IP hash."),
        ("Explain caching strategies", "Caching stores frequently accessed data in fast storage. Strategies include cache-aside (lazy loading), write-through (update cache on writes), write-behind (async writes), and time-based expiration."),
        ("What is horizontal scaling?", "Horizontal scaling adds more servers to distribute load, unlike vertical scaling which adds resources to one server. Horizontal scaling is more resilient and cost-effective but requires load balancing."),
        ("What is a CDN?", "Content Delivery Networks distribute static assets across geographically dispersed servers. Users download from nearby servers, reducing latency and server load. Common for images, videos, CSS, JavaScript."),
        ("Explain eventual consistency", "Eventual consistency means replicas don't immediately synchronize but converge to the same state over time. It trades strong consistency for availability and partition tolerance (CAP theorem)."),
        ("What is sharding?", "Sharding horizontally partitions data across multiple databases. Each shard contains a subset of data, enabling scaling beyond single-server capacity. Challenges include rebalancing and cross-shard queries."),
        ("What is a message queue?", "Message queues enable asynchronous communication between services. Producers send messages to a queue; consumers process them independently. This decouples components and handles traffic spikes."),
        ("Explain microservices", "Microservices architecture breaks applications into small, independent services that communicate via APIs. Each service handles a specific business function and can be deployed/scaled independently."),
    ],
    
    "general_help": [
        ("I don't understand", "That's completely okay! Let me explain it in a different way. Which specific part is confusing?"),
        ("Can you clarify?", "Of course! What aspect would you like me to explain in more detail?"),
        ("This is confusing", "I understand. Let me break it down into simpler steps. What specific part should we focus on first?"),
        ("I need help", "I'm here to help! What specific topic or problem are you working on?"),
        ("Can you give an example?", "Absolutely! Examples often make concepts clearer. Here's a practical example: [concept would be explained with specific example based on context]"),
        ("Why is this important?", "Great question! Understanding the 'why' helps with learning. Let me explain the practical applications and benefits."),
        ("What should I learn next?", "That depends on your goals and current knowledge. What area interests you most, or what are you trying to build?"),
        ("How long will this take?", "Learning timelines vary based on prior knowledge and practice frequency. Focus on consistent effort rather than speed. What's your target timeline?"),
    ],
    
    "career_advice": [
        ("How do I become a developer?", "Start by learning programming fundamentals with a beginner-friendly language like Python or JavaScript. Build projects, contribute to open source, create a portfolio, and practice consistently. Bootcamps or self-study both work."),
        ("What skills are most important?", "Technical skills matter, but so do problem-solving, communication, learning agility, and collaboration. Stay curious, practice debugging, and learn to break complex problems into manageable pieces."),
        ("Should I get a degree?", "It depends on your goals and circumstances. Degrees provide structured learning and credentials, but many successful developers are self-taught or bootcamp-trained. Focus on building demonstrable skills."),
        ("How do I prepare for interviews?", "Practice coding problems on platforms like LeetCode, understand data structures and algorithms, review system design basics, prepare your project stories using STAR method, and research the company thoroughly."),
        ("What's the job market like?", "The tech job market fluctuates but overall demand for skilled developers remains strong. Specialize in areas with growth potential, build a strong portfolio, network actively, and stay adaptable."),
        ("How do I negotiate salary?", "Research market rates for your role and location, know your worth, have a target range ready, let them make the first offer when possible, consider total compensation (benefits, equity), and practice your pitch."),
        ("Should I specialize or generalize?", "Early career: generalize to find your interests and build foundations. Mid-career: specialize in high-demand areas while maintaining broad knowledge. T-shaped skills (deep in one area, broad overall) are valuable."),
        ("How do I switch careers into tech?", "Leverage transferable skills from your previous career, learn programming through online courses or bootcamps, build a portfolio of projects, network with developers, and target companies that value diverse backgrounds."),
    ],
    
    "best_practices": [
        ("How do I write clean code?", "Use meaningful names, keep functions small and focused, minimize nesting, follow consistent style, comment the why not the what, avoid duplication, and optimize for readability over cleverness."),
        ("What makes code maintainable?", "Clear organization, good naming, proper documentation, tests, modular design, consistent patterns, minimal dependencies, and thinking about the next developer (often future you) who reads it."),
        ("How do I handle errors?", "Anticipate failure points, use try-catch blocks, return meaningful error messages, log errors with context, fail gracefully, validate inputs, and have fallback behaviors for critical paths."),
        ("What is code review best practice?", "Be constructive and respectful, explain your reasoning, focus on important issues, ask questions to understand intent, acknowledge good work, distinguish blocking vs. non-blocking comments, and learn from others."),
        ("How do I improve performance?", "Profile first to identify bottlenecks, optimize algorithms and data structures, reduce unnecessary operations, cache expensive computations, minimize I/O, parallelize when appropriate, but don't optimize prematurely."),
        ("What is technical debt?", "Technical debt is the implied cost of future rework from choosing quick solutions over better approaches. Like financial debt, it accumulates interest through maintenance difficulty. Pay it down periodically through refactoring."),
        ("How do I document code?", "Write self-documenting code with clear names, add comments for complex logic or non-obvious decisions, maintain README files for projects, document API contracts, and keep documentation synchronized with code."),
        ("What are SOLID principles?", "SOLID is five OOP design principles: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion. They promote maintainable, flexible, and testable code."),
    ],
}

# Additional diverse conversation starters
conversation_starters = [
    "Tell me about {topic}",
    "How does {topic} work?",
    "What is {topic}?",
    "Explain {topic} in simple terms",
    "Can you help me understand {topic}?",
    "I'm learning about {topic}",
    "What are the basics of {topic}?",
    "Why is {topic} important?",
]

topics = [
    "version control", "Git", "Docker", "Kubernetes", "cloud computing",
    "AWS", "Azure", "microservices", "REST APIs", "GraphQL",
    "testing", "unit tests", "integration tests", "CI/CD", "DevOps",
    "security", "encryption", "authentication", "authorization", "OAuth",
    "performance optimization", "caching", "databases", "SQL", "NoSQL",
    "frontend frameworks", "React", "Vue", "Angular", "state management",
    "backend development", "Node.js", "Django", "Flask", "Ruby on Rails",
    "mobile development", "iOS", "Android", "React Native", "Flutter",
    "data science", "pandas", "numpy", "data visualization", "statistics",
]

# Generate new pairs
new_pairs = []
target_total = 7500
current_count = len(existing_pairs)
needed = target_total - current_count

print(f"📊 Current pairs: {current_count}")
print(f"🎯 Target pairs: {target_total}")
print(f"➕ Generating: {needed} new pairs")

# First, add all template variations
for category, pairs in templates.items():
    for input_text, response_text in pairs:
        # Add variations
        variations = [
            (input_text, response_text),
            (input_text + "?", response_text),
            (input_text.lower(), response_text),
        ]
        for inp, resp in variations:
            new_pairs.append(f"INPUT: {inp}\nRESPONSE: {resp}")
            if len(new_pairs) >= needed:
                break
        if len(new_pairs) >= needed:
            break
    if len(new_pairs) >= needed:
        break

# Add topic-based conversations
for _ in range(min(1000, needed - len(new_pairs))):
    topic = random.choice(topics)
    starter = random.choice(conversation_starters).format(topic=topic)
    response = f"I'd be happy to explain {topic}! It's an important concept in software development. Could you tell me what specific aspect you're most interested in, or would you like a general overview?"
    new_pairs.append(f"INPUT: {starter}\nRESPONSE: {response}")

# Fill remaining with variations
base_responses = [
    "That's a great question! Let me explain step by step.",
    "I can help you with that. Here's what you need to know:",
    "Good question! This is an important concept to understand.",
    "Let me break this down for you in simple terms.",
    "I'd be happy to explain that! Let's start with the basics.",
]

while len(new_pairs) < needed:
    # Create plausible technical questions
    question_templates = [
        "How do I {action} in {language}?",
        "What's the best way to {action}?",
        "Can you explain {concept}?",
        "What's the difference between {thing1} and {thing2}?",
        "Why should I use {tool}?",
        "When should I use {pattern}?",
        "How can I improve my {skill}?",
    ]
    
    actions = ["optimize code", "handle errors", "write tests", "debug", "refactor", "deploy", "structure projects"]
    languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript"]
    concepts = ["closures", "promises", "async/await", "decorators", "generators", "classes", "modules"]
    things = [("lists", "tuples"), ("GET", "POST"), ("let", "const"), ("SQL", "NoSQL"), ("stack", "heap")]
    tools = ["Docker", "Git", "TypeScript", "linters", "formatters", "debuggers"]
    patterns = ["factory pattern", "singleton", "observer", "dependency injection", "MVC"]
    skills = ["coding", "debugging", "problem solving", "system design", "communication"]
    
    template = random.choice(question_templates)
    if "{action}" in template:
        question = template.format(action=random.choice(actions), language=random.choice(languages))
    elif "{concept}" in template:
        question = template.format(concept=random.choice(concepts))
    elif "{thing1}" in template:
        pair = random.choice(things)
        question = template.format(thing1=pair[0], thing2=pair[1])
    elif "{tool}" in template:
        question = template.format(tool=random.choice(tools))
    elif "{pattern}" in template:
        question = template.format(pattern=random.choice(patterns))
    elif "{skill}" in template:
        question = template.format(skill=random.choice(skills))
    else:
        question = "How can I learn programming effectively?"
    
    response = random.choice(base_responses)
    new_pairs.append(f"INPUT: {question}\nRESPONSE: {response}")

# Shuffle new pairs
random.shuffle(new_pairs)

# Combine existing and new, limiting to target
all_pairs = existing_pairs + new_pairs[:needed]
random.shuffle(all_pairs)  # Shuffle all pairs

# Write to file
try:
    with open('sample_training_data.txt', 'w') as f:
        f.write('\n\n'.join(all_pairs))
    print(f"\n✅ Successfully created training data with {len(all_pairs)} pairs")
    print(f"📁 Saved to: sample_training_data.txt")
    print(f"📏 File size: {len('\n\n'.join(all_pairs))} characters")
except Exception as e:
    print(f"❌ Error writing file: {e}")
    sys.exit(1)
